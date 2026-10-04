"""Builds supervised fine-tuning data that teaches a small local model
DocuMind's exact answering contract:

  * read numbered passages in the same format app.qa.context_builder emits,
  * answer only from them, with inline [n] citations to the passage used,
  * reply with the exact NOT_FOUND_MESSAGE when the passages don't answer,
  * no "Based on the provided documents" preamble.

Source: SQuAD v2 (Wikipedia; human-written questions, extractive answers and
human-written unanswerable questions). None of DocuMind's demo documents or
eval questions are used, so the eval set stays a held-out test.

Each example holds 3-4 passages from the same Wikipedia article: the gold
paragraph plus same-article distractors (hard negatives), shuffled, so the
model has to pick the right passage number. Unanswerable questions get the
article's paragraphs with no answer present.

    python finetune/build_sft_data.py --train 1600 --val 120
Output: finetune/data/{train,val}.jsonl with {"messages": [...]} chat records.
"""
import argparse
import glob
import json
import os
import random
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from app.qa.generator import NOT_FOUND_MESSAGE, system_instruction  # noqa: E402

HERE = Path(__file__).parent
MAX_PASSAGE_WORDS = 170  # keeps 4 passages + prompt inside ~1k tokens


def load_squad(split: str) -> pd.DataFrame:
    from huggingface_hub import snapshot_download
    root = snapshot_download("rajpurkar/squad_v2", repo_type="dataset", allow_patterns=["squad_v2/*"])
    files = glob.glob(os.path.join(root, "squad_v2", f"{split}-*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def answer_sentence(context: str, start: int, answer: str) -> str | None:
    """The sentence of the paragraph that contains the answer span."""
    sents = [(m.start(), m.end()) for m in re.finditer(r"[^.!?]+(?:[.!?]+|$)", context)]
    for s, e in sents:
        if s <= start < e:
            sent = context[s:e].strip()
            if answer in sent and 6 <= len(sent.split()) <= 45:
                return sent
    return None


def truncate(text: str, words: int) -> str:
    w = text.split()
    return text if len(w) <= words else " ".join(w[:words]) + " …"


def build_prompt(question: str, passages: list[tuple[str, str]]) -> str:
    # mirrors app.qa.generator._build_prompt + context_builder.build_context
    blocks = [f"[{i}] (Page {i} · {section})\n{text}" for i, (section, text) in enumerate(passages, start=1)]
    context = "\n\n".join(blocks)
    return (f"Numbered passages:\n{context}\n\n"
            f"Question: {question}\n\nAnswer using only the passages above, citing them inline as [n].")


def make_examples(df: pd.DataFrame, n: int, rng: random.Random, unanswerable_share: float) -> list[dict]:
    by_title = {t: g for t, g in df.groupby("title")}
    paragraphs = {t: list(dict.fromkeys(g["context"])) for t, g in by_title.items()}
    titles = [t for t in by_title if len(paragraphs[t]) >= 4]
    system = system_instruction(False)
    out, seen = [], set()
    while len(out) < n:
        title = rng.choice(titles)
        g = by_title[title]
        row = g.iloc[rng.randrange(len(g))]
        if row["id"] in seen:
            continue
        seen.add(row["id"])
        answers = row["answers"]
        answerable = len(answers["text"]) > 0
        want_unanswerable = rng.random() < unanswerable_share
        if answerable == want_unanswerable:
            continue
        section = title.replace("_", " ")
        k = rng.choice([3, 4])
        others = [p for p in paragraphs[title] if p != row["context"]]
        if answerable:
            ans, start = answers["text"][0], int(answers["answer_start"][0])
            gold = row["context"]
            # keep the answer inside the truncated gold passage
            if len(gold[: start + len(ans)].split()) > MAX_PASSAGE_WORDS:
                continue
            passages = [gold] + rng.sample(others, k - 1)
            rng.shuffle(passages)
            gold_n = passages.index(gold) + 1
            sent = answer_sentence(gold, start, ans)
            target = (f"{sent.rstrip('.!?')} [{gold_n}]." if sent and rng.random() < 0.7
                      else f"{ans.rstrip('.')} [{gold_n}].")
            if target[0].islower():
                target = target[0].upper() + target[1:]
        else:
            # SQuAD v2 unanswerable questions are written against this paragraph;
            # include it so the model learns to refuse near-miss passages
            passages = [row["context"]] + rng.sample(others, k - 1)
            rng.shuffle(passages)
            target = NOT_FOUND_MESSAGE
        passages = [(section, truncate(p, MAX_PASSAGE_WORDS)) for p in passages]
        out.append({"messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": build_prompt(row["question"], passages)},
            {"role": "assistant", "content": target},
        ], "answerable": answerable})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=1600)
    ap.add_argument("--val", type=int, default=120)
    ap.add_argument("--unanswerable-share", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    (HERE / "data").mkdir(exist_ok=True)
    for split, n, source in [("train", args.train, "train"), ("val", args.val, "validation")]:
        rows = make_examples(load_squad(source), n, rng, args.unanswerable_share)
        path = HERE / "data" / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{path}: {len(rows)} examples, {sum(not r['answerable'] for r in rows)} unanswerable")


if __name__ == "__main__":
    main()
