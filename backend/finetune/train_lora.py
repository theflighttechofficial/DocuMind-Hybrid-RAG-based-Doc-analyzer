"""LoRA fine-tune of a small instruct model on finetune/data/train.jsonl, on CPU.

Loss is computed on the assistant answer only (prompt tokens are masked), so
the model learns *how to answer from passages* — citation format, refusal
wording, brevity — rather than memorizing SQuAD passages.

    python finetune/train_lora.py --dry-run     # time 3 steps and exit
    python finetune/train_lora.py               # GPU defaults: 1.5B base, 300 steps
    python finetune/train_lora.py --base Qwen/Qwen2.5-0.5B-Instruct --name documind-qwen2.5-0.5b

Runs on CUDA when available (bf16), else CPU (fp32; ~30 s per example on a
4-core laptop, too slow for more than a smoke test).

Writes finetune/out/<name>/adapter (LoRA weights), finetune/out/<name>/merged
(full fp16 safetensors, ready for `ollama create`), and train_log.json.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

HERE = Path(__file__).parent


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def encode(tok, messages, max_len):
    """input_ids with labels = -100 everywhere except the assistant reply."""
    prompt_ids = tok.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    full_ids = tok.apply_chat_template(messages, tokenize=True)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("chat template prefix mismatch")
    full_ids = full_ids[:max_len]
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return torch.tensor(full_ids), torch.tensor(labels[: len(full_ids)])


@torch.no_grad()
def val_loss(model, data):
    model.eval()
    total, n = 0.0, 0
    for ids, labels in data:
        out = model(input_ids=ids[None].to(model.device), labels=labels[None].to(model.device))
        total += out.loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--name", default="documind-qwen2.5-1.5b",
                    help="also the Ollama model name; set OLLAMA_MODEL to this in backend/.env")
    ap.add_argument("--steps", type=int, default=300, help="optimizer steps")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=1280)
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--val-size", type=int, default=40)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-grad-checkpointing", action="store_true",
                    help="faster, needs more memory (fine for 0.5B on an 8 GB GPU)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    random.seed(0)
    out_dir = HERE / "out" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    print(f"device {device} dtype {dtype}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype).to(device)
    model.config.use_cache = False
    if not args.no_grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable / 1e6:.2f}M")

    train = [encode(tok, r["messages"], args.max_len) for r in load_jsonl(HERE / "data" / "train.jsonl")]
    val = [encode(tok, r["messages"], args.max_len) for r in load_jsonl(HERE / "data" / "val.jsonl")][: args.val_size]
    random.shuffle(train)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    total_micro = args.steps * args.grad_accum
    warmup = max(1, args.steps // 20)

    def lr_at(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, args.steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    log = {"args": vars(args), "trainable_params": trainable, "steps": []}
    if not args.dry_run:
        v0 = val_loss(model, val)
        print(f"step 0 val_loss {v0:.4f}")
        log["steps"].append({"step": 0, "val_loss": v0})

    model.train()
    t0 = time.time()
    running = 0.0
    for micro in range(total_micro if not args.dry_run else 3 * args.grad_accum):
        ids, labels = train[micro % len(train)]
        loss = model(input_ids=ids[None].to(device), labels=labels[None].to(device)).loss / args.grad_accum
        loss.backward()
        running += loss.item()
        if (micro + 1) % args.grad_accum == 0:
            step = (micro + 1) // args.grad_accum
            for g in opt.param_groups:
                g["lr"] = args.lr * lr_at(step - 1)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad()
            elapsed = time.time() - t0
            print(f"step {step}/{args.steps} loss {running:.4f} lr {opt.param_groups[0]['lr']:.2e} "
                  f"{elapsed / step:.1f}s/step eta {(args.steps - step) * elapsed / step / 60:.0f}min", flush=True)
            entry = {"step": step, "train_loss": running, "elapsed_s": round(elapsed)}
            running = 0.0
            if not args.dry_run and (step % args.val_every == 0 or step == args.steps):
                entry["val_loss"] = val_loss(model, val)
                print(f"step {step} val_loss {entry['val_loss']:.4f}", flush=True)
                model.save_pretrained(out_dir / "adapter")
            log["steps"].append(entry)
            (out_dir / "train_log.json").write_text(json.dumps(log, indent=2))

    if args.dry_run:
        return

    model.save_pretrained(out_dir / "adapter")
    merged = model.merge_and_unload()
    merged = merged.to("cpu", torch.float16)
    merged.config.use_cache = True
    merged.save_pretrained(out_dir / "merged", safe_serialization=True)
    tok.save_pretrained(out_dir / "merged")
    print("saved", out_dir / "merged")


if __name__ == "__main__":
    sys.exit(main())
