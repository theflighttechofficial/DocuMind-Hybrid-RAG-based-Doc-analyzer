"""Light, no-GPU adaptation of a stock Ollama model to DocuMind: bakes the
answering rules (system prompt) and a handful of worked examples (few-shot
chat turns) into a named Ollama model. No weights change; the model sees the
examples ahead of every conversation.

Examples come from SQuAD v2 validation (same source and format as the LoRA
training data in build_sft_data.py), never from DocuMind's demo documents or
eval set. Passages are cut to ~45 words to keep the added prompt small: on a
CPU every extra prompt token costs latency.

    python finetune/build_fewshot_model.py --base qwen3:4b-instruct --name documind-qwen3-4b
"""
import argparse
import os
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from app.qa.generator import NOT_FOUND_MESSAGE, system_instruction  # noqa: E402
from finetune.build_sft_data import answer_sentence, build_prompt, load_squad  # noqa: E402

HERE = Path(__file__).parent
PASSAGE_WORDS = 45


def short(text: str) -> str:
    w = text.split()
    return text if len(w) <= PASSAGE_WORDS else " ".join(w[:PASSAGE_WORDS]) + " …"


def pick_examples(seed: int) -> tuple[list, list]:
    df = load_squad("validation")
    rng = random.Random(seed)
    groups = [g for _, g in df.groupby("title") if g["context"].nunique() >= 2]
    rng.shuffle(groups)
    answerable, unanswerable = [], []
    for g in groups:
        paras = list(dict.fromkeys(g["context"]))
        for _, row in g.sample(frac=1, random_state=seed).iterrows():
            has = len(row["answers"]["text"]) > 0
            other = next(p for p in paras if p != row["context"])
            section = row["title"].replace("_", " ")
            if has and len(answerable) < 2:
                ans, start = row["answers"]["text"][0], int(row["answers"]["answer_start"][0])
                sent = answer_sentence(row["context"], start, ans)
                if not sent or len(row["context"][: start + len(ans)].split()) > PASSAGE_WORDS:
                    continue
                gold_first = len(answerable) == 1  # vary the cited number: [2] then [1]
                passages = [row["context"], other] if gold_first else [other, row["context"]]
                n = 1 if gold_first else 2
                prompt = build_prompt(row["question"], [(section, short(p)) for p in passages])
                answerable.append((prompt, f"{sent.rstrip('.!?')} [{n}]."))
                break
            if not has and not unanswerable:
                prompt = build_prompt(row["question"], [(section, short(row["context"])), (section, short(other))])
                unanswerable.append((prompt, NOT_FOUND_MESSAGE))
                break
        if len(answerable) == 2 and unanswerable:
            break
    return answerable, unanswerable


def quote(text: str) -> str:
    return '"""' + text.replace('"""', "'''") + '"""'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="qwen3:4b-instruct")
    ap.add_argument("--name", default="documind-qwen3-4b")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--refusal-example", action="store_true",
                    help="also add a worked refusal; off by default because on qwen3:4b-instruct it "
                         "made vague questions ('what is this?') get refused")
    args = ap.parse_args()

    lines = [
        f"# Built by finetune/build_fewshot_model.py from {args.base}.",
        f"FROM {args.base}",
        f"SYSTEM {quote(system_instruction(False))}",
        "PARAMETER temperature 0.1",
        "PARAMETER num_ctx 4096",
    ]
    answerable, unanswerable = pick_examples(args.seed)
    examples = [answerable[0], unanswerable[0], answerable[1]] if args.refusal_example else answerable
    for prompt, answer in examples:
        lines.append(f"MESSAGE user {quote(prompt)}")
        lines.append(f"MESSAGE assistant {quote(answer)}")
    out = HERE / "out" / args.name
    out.mkdir(parents=True, exist_ok=True)
    modelfile = out / "Modelfile"
    # LF only, see build_ollama_model.py
    modelfile.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(modelfile.read_text(encoding="utf-8")[-1500:])
    subprocess.run(["ollama", "create", args.name, "-f", str(modelfile)], check=True)


if __name__ == "__main__":
    main()
