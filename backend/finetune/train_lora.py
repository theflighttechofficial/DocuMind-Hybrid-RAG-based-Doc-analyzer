"""LoRA / QLoRA fine-tune of an instruct model on finetune/data/train.jsonl.

Loss is computed on the assistant answer only (prompt tokens are masked), so
the model learns *how to answer from passages* — citation format, refusal
wording, brevity — rather than memorizing SQuAD passages.

    python finetune/train_lora.py --dry-run     # time 3 steps and exit
    python finetune/train_lora.py               # Qwen3-4B, QLoRA, light run: 60 steps x 4 examples
    python finetune/train_lora.py --base Qwen/Qwen2.5-0.5B-Instruct --name documind-qwen2.5-0.5b --no-4bit

Runs on CUDA when available, else CPU (fp32; ~30 s per example for 0.5B on a
4-core laptop; a 4B model does not fit in CPU training memory at all).
On CUDA, models of 3B+ parameters load in 4-bit (QLoRA, bitsandbytes) so
Qwen3-4B trains in 8 GB of VRAM; the adapter is then merged into a bf16 copy
of the base on the CPU (needs ~10 GB free RAM) for `ollama create`.

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
    ap.add_argument("--base", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--name", default="documind-qwen3-4b-lora",
                    help="also the Ollama model name; set OLLAMA_MODEL to this in backend/.env")
    # deliberately light: 240 examples only teach the answer format (citations,
    # refusal sentence, brevity) without overwriting what the base model knows
    ap.add_argument("--steps", type=int, default=60, help="optimizer steps")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=1280)
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--val-size", type=int, default=40)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-grad-checkpointing", action="store_true",
                    help="faster, needs more memory (fine for 0.5B on an 8 GB GPU)")
    ap.add_argument("--no-4bit", action="store_true", help="train in bf16 even for 3B+ models on CUDA")
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
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.base)
    approx_params = cfg.num_hidden_layers * 12 * cfg.hidden_size ** 2 + cfg.vocab_size * cfg.hidden_size
    use_4bit = device.type == "cuda" and not args.no_4bit and approx_params > 3e9
    if use_4bit:
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        model = AutoModelForCausalLM.from_pretrained(args.base, device_map={"": 0}, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=not args.no_grad_checkpointing)
        print("loaded in 4-bit (QLoRA)")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype).to(device)
    model.config.use_cache = False
    if not args.no_grad_checkpointing and not use_4bit:
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
            # a dry run must not overwrite the real run's log
            log_name = "dry_run_log.json" if args.dry_run else "train_log.json"
            (out_dir / log_name).write_text(json.dumps(log, indent=2))

    if args.dry_run:
        return

    model.save_pretrained(out_dir / "adapter")
    if use_4bit:
        # merging into 4-bit weights loses precision: reload the base in bf16 on CPU
        from peft import PeftModel
        del model, opt
        torch.cuda.empty_cache()
        base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, low_cpu_mem_usage=True)
        model = PeftModel.from_pretrained(base, out_dir / "adapter")
    merged = model.merge_and_unload()
    merged = merged.to("cpu", torch.float16)
    merged.config.use_cache = True
    merged.save_pretrained(out_dir / "merged", safe_serialization=True)
    tok.save_pretrained(out_dir / "merged")
    print("saved", out_dir / "merged")


if __name__ == "__main__":
    sys.exit(main())
