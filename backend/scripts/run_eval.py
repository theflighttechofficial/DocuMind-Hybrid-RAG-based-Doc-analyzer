"""End-to-end evaluation of a running DocuMind backend.

Talks to the live API over HTTP, exactly like the frontend does:
  1. logs in as a dedicated eval user (registers it on first run),
  2. deletes that user's previous copies of the demo documents and uploads
     fresh ones from demo-data/, waiting until processing finishes,
  3. runs every question in demo-data/sample_evaluation_set.json through
     /search (per retrieval method) and /ask (with conversation chaining for
     follow-ups, and a workspace for cross-document questions),
  4. writes the full per-question record and a summary to
     backend/eval_results/<timestamp>.json.

Usage (backend must be running):
    python scripts/run_eval.py --label baseline
    python scripts/run_eval.py --label phase1 --compare eval_results/<baseline>.json
    python scripts/run_eval.py --skip-upload        # reuse already-processed docs
"""
import argparse
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import requests

ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = ROOT / "demo-data"
EVAL_SET = DEMO_DIR / "sample_evaluation_set.json"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "eval_results"

EVAL_EMAIL = "eval@documind.local"
EVAL_PASSWORD = "documind-eval-2026"
TOP_K = 5

# Must match app.qa.generator.NOT_FOUND_MESSAGE (checked loosely so small
# wording changes in later phases don't break refusal detection).
NOT_FOUND_MARKERS = ("couldn't find sufficient information", "could not find sufficient information")
UNAVAILABLE_MARKER = "temporarily unavailable"


# ---------------------------------------------------------------- text utils

def norm(text: str) -> str:
    """Lowercase alphanumeric tokens joined by single spaces. Digit groups like
    300,000 become 300000 so number formatting doesn't decide correctness."""
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text or "")
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def contains(haystack: str, needle: str) -> bool:
    h, n = f" {norm(haystack)} ", f" {norm(needle)} "
    return n.strip() != "" and n in h


def token_f1(pred: str, truth: str) -> float:
    stop = {"the", "a", "an", "of", "to", "in", "is", "are", "was", "were", "and", "or", "for", "on", "at"}
    p = [t for t in norm(pred).split() if t not in stop]
    t = [x for x in norm(truth).split() if x not in stop]
    if not p or not t:
        return 0.0
    common = sum(min(p.count(w), t.count(w)) for w in set(p))
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(t)
    return 2 * prec * rec / (prec + rec)


_judge = None


def semantic_similarity(a: str, b: str) -> float:
    """Fixed judge model (all-MiniLM-L6-v2) regardless of what the app uses for
    retrieval, so this metric stays comparable across phases."""
    global _judge
    if _judge is None:
        from sentence_transformers import SentenceTransformer
        _judge = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    va, vb = _judge.encode([a, b], normalize_embeddings=True)
    return float((va * vb).sum())


def is_refusal(answer: str, grounded) -> bool:
    low = (answer or "").lower()
    return grounded is False or any(m in low for m in NOT_FOUND_MARKERS)


# ---------------------------------------------------------------- API client

class Api:
    def __init__(self, base: str):
        self.base = base.rstrip("/")
        self.s = requests.Session()

    def _url(self, path):
        return f"{self.base}/api{path}"

    def call(self, method, path, **kw):
        kw.setdefault("timeout", 600)
        r = self.s.request(method, self._url(path), **kw)
        if r.status_code >= 400:
            raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text[:300]}")
        return r.json() if r.content else None

    def login(self):
        try:
            tok = self.call("POST", "/auth/login", json={"email": EVAL_EMAIL, "password": EVAL_PASSWORD})
        except RuntimeError:
            tok = self.call("POST", "/auth/register", json={"email": EVAL_EMAIL, "password": EVAL_PASSWORD})
        self.s.headers["Authorization"] = f"Bearer {tok['access_token']}"


def upload_documents(api: Api, doc_files: dict) -> dict:
    existing = api.call("GET", "/documents")
    wanted_names = set(doc_files.values())
    for d in existing:
        if d["name"] in wanted_names:
            api.call("DELETE", f"/documents/{d['id']}")

    # One at a time, so per-document processing time is measured in isolation.
    ids, timings = {}, {}
    for key, filename in doc_files.items():
        started = time.time()
        with open(DEMO_DIR / filename, "rb") as fh:
            doc = api.call("POST", "/documents", files={"file": (filename, fh)})
        ids[key] = doc["id"]
        deadline = started + 1800
        while time.time() < deadline:
            d = api.call("GET", f"/documents/{doc['id']}")
            if d["processing_status"] in ("ready", "failed"):
                break
            time.sleep(1)
        else:
            raise RuntimeError(f"Timed out waiting for {filename}")
        timings[key] = round(time.time() - started, 1)
        print(f"  {filename}: {d['processing_status']} in {timings[key]}s ({d['page_count']} pages)")
        if d["processing_status"] == "failed":
            raise RuntimeError(f"Processing failed for {filename}: {d.get('error_message')}")
    return {"ids": ids, "processing_seconds": timings}


def existing_documents(api: Api, doc_files: dict) -> dict:
    by_name = {d["name"]: d for d in api.call("GET", "/documents") if d["processing_status"] == "ready"}
    missing = [f for f in doc_files.values() if f not in by_name]
    if missing:
        raise RuntimeError(f"--skip-upload but these are not ready for the eval user: {missing}")
    return {"ids": {k: by_name[f]["id"] for k, f in doc_files.items()}, "processing_seconds": {}}


# ---------------------------------------------------------------- scoring

def retrieval_scores(texts: list[str], quotes: list[str]) -> dict:
    """Relevance = passage contains a gold evidence quote."""
    rel = [any(contains(t, q) for q in quotes) for t in texts[:TOP_K]]
    found = [q for q in quotes if any(contains(t, q) for t in texts[:TOP_K])]
    first = next((i for i, r in enumerate(rel) if r), None)
    return {
        "hit": 1.0 if found else 0.0,
        "recall": len(found) / len(quotes),
        "precision": (sum(rel) / len(rel)) if rel else 0.0,
        "mrr": 1.0 / (first + 1) if first is not None else 0.0,
    }


def ask_with_retry(api: Api, path: str, payload: dict, sleep_s: float, retries: int = 4):
    for attempt in range(retries):
        t0 = time.time()
        res = api.call("POST", path, json=payload)
        latency = time.time() - t0
        if UNAVAILABLE_MARKER not in (res.get("answer") or "").lower():
            return res, latency, None
        wait = sleep_s * (attempt + 2) * 5
        print(f"    LLM unavailable, retrying in {wait:.0f}s")
        time.sleep(wait)
    return res, latency, "llm_unavailable"


def mean(values):
    values = [v for v in values if v is not None]
    return round(statistics.fmean(values), 4) if values else None


def run(args):
    eval_set = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    doc_files = eval_set["documents"]
    questions = eval_set["questions"]
    if args.only:
        wanted = set(args.only.split(","))
        questions = [q for q in questions if q["id"] in wanted or q.get("follow_up_of") in wanted]

    api = Api(args.base_url)
    api.login()
    config = {}
    try:
        config = api.call("GET", "/system/config")
    except RuntimeError:
        pass  # older backends don't expose it

    print("Documents:")
    docs = existing_documents(api, doc_files) if args.skip_upload else upload_documents(api, doc_files)
    ids = docs["ids"]

    workspace_ids = {}

    def workspace_for(keys):
        key = tuple(sorted(keys))
        if key not in workspace_ids:
            name = "eval: " + " + ".join(key)
            for w in api.call("GET", "/workspaces"):
                if w["name"] == name:
                    api.call("DELETE", f"/workspaces/{w['id']}")
            w = api.call("POST", "/workspaces", json={"name": name})
            api.call("POST", f"/workspaces/{w['id']}/documents", json={"document_ids": [ids[k] for k in keys]})
            workspace_ids[key] = w["id"]
        return workspace_ids[key]

    methods = args.methods.split(",")
    conversations = {}
    records = []

    for q in questions:
        rec = {k: q.get(k) for k in ("id", "type", "question", "answer", "must_contain", "evidence", "documents", "follow_up_of")}
        rec["answerable"] = q.get("answerable", True)
        multi = len(q["documents"]) > 1
        doc_id = ids[q["documents"][0]]
        print(f"[{q['id']}] {q['question']}")

        # --- retrieval via /search (single-document, standalone questions only:
        # /search has no conversation context, so follow-ups are scored through
        # the evidence /ask returns instead)
        rec["retrieval"] = {}
        if q["evidence"] and not multi and q["type"] != "follow-up":
            for method in methods:
                body = {"query": q["question"], "method": method, "top_k": TOP_K}
                try:
                    results = api.call("POST", f"/documents/{doc_id}/search", json=body)
                except RuntimeError as exc:
                    rec["retrieval"][method] = {"error": str(exc)}
                    continue
                texts = [r["text"] for r in results]
                scores = retrieval_scores(texts, q["evidence"])
                scores["pages"] = [r["page"] for r in results]
                rec["retrieval"][method] = scores

        if args.no_qa:
            records.append(rec)
            continue

        # --- answer via /ask
        if multi:
            path = f"/workspaces/{workspace_for(q['documents'])}/ask"
            payload = {"question": q["question"]}
        else:
            path = f"/documents/{doc_id}/ask"
            payload = {"question": q["question"]}
            if q.get("follow_up_of") and q["follow_up_of"] in conversations:
                payload["conversation_id"] = conversations[q["follow_up_of"]]
        res, latency, err = ask_with_retry(api, path, payload, args.sleep)
        if res.get("conversation_id"):
            conversations[q["id"]] = res["conversation_id"]

        answer = res.get("answer") or ""
        evidence_texts = [e.get("full_text") or e.get("text") or "" for e in res.get("evidence", [])]
        refused = is_refusal(answer, res.get("grounded"))
        rec.update({
            "prediction": answer, "grounded": res.get("grounded"), "refused": refused,
            "confidence": res.get("confidence"), "latency_s": round(latency, 2), "error": err,
            "evidence_returned": [
                {"page": e.get("page"), "document": e.get("document_name"), "text": (e.get("text") or "")[:300]}
                for e in res.get("evidence", [])
            ],
        })
        if rec["answerable"]:
            rec["answer_correct"] = 0.0 if refused else float(all(contains(answer, m) for m in q["must_contain"]))
            rec["f1"] = token_f1(answer, q["answer"])
            rec["semantic_similarity"] = semantic_similarity(answer, q["answer"])
            rec["context_hit"] = retrieval_scores(evidence_texts, q["evidence"])["hit"] if q["evidence"] else None
        else:
            rec["refusal_correct"] = float(refused)
        mark = "ok" if rec.get("answer_correct", rec.get("refusal_correct")) else "MISS"
        print(f"    {mark}  {answer[:110]!r}")
        records.append(rec)
        if args.sleep:
            time.sleep(args.sleep)

    summary = summarize(records, methods)
    out = {
        "label": args.label, "timestamp": datetime.now().isoformat(timespec="seconds"),
        "base_url": args.base_url, "backend_config": config,
        "documents": docs, "summary": summary, "questions": records,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}{'-' + args.label if args.label else ''}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print_summary(summary, args.compare)
    print(f"\nSaved {path}")


def summarize(records, methods):
    retrieval = {}
    for m in methods:
        rows = [r["retrieval"][m] for r in records if m in r.get("retrieval", {}) and "error" not in r["retrieval"][m]]
        retrieval[m] = {k: mean([row[k] for row in rows]) for k in ("hit", "recall", "precision", "mrr")}
        retrieval[m]["n"] = len(rows)

    answerable = [r for r in records if r["answerable"] and "prediction" in r and not r.get("error")]
    unanswerable = [r for r in records if not r["answerable"] and "prediction" in r and not r.get("error")]
    by_type = {}
    for r in answerable:
        by_type.setdefault(r["type"], []).append(r["answer_correct"])

    qa = {
        "answer_accuracy": mean([r["answer_correct"] for r in answerable]),
        "token_f1": mean([r["f1"] for r in answerable]),
        "semantic_similarity": mean([r["semantic_similarity"] for r in answerable]),
        "context_hit": mean([r["context_hit"] for r in answerable]),
        "false_refusal_rate": mean([float(r["refused"]) for r in answerable]),
        "refusal_accuracy": mean([r["refusal_correct"] for r in unanswerable]),
        "accuracy_by_type": {t: mean(v) for t, v in sorted(by_type.items())},
        "mean_latency_s": mean([r["latency_s"] for r in records if "latency_s" in r]),
        "n_answerable": len(answerable), "n_unanswerable": len(unanswerable),
        "n_errors": sum(1 for r in records if r.get("error")),
    }
    return {"retrieval": retrieval, "qa": qa}


def _flat(summary):
    flat = {}
    for m, v in summary["retrieval"].items():
        for k, x in v.items():
            if k != "n":
                flat[f"retrieval.{m}.{k}"] = x
    for k, v in summary["qa"].items():
        if isinstance(v, dict):
            for t, x in v.items():
                flat[f"qa.by_type.{t}"] = x
        elif not k.startswith("n_"):
            flat[f"qa.{k}"] = v
    return flat


def print_summary(summary, compare_path=None):
    prev = {}
    if compare_path:
        prev = _flat(json.loads(Path(compare_path).read_text(encoding="utf-8"))["summary"])
    print("\n" + "=" * 64)
    for key, value in _flat(summary).items():
        line = f"{key:42s} {value if value is not None else '-':>8}"
        if key in prev and prev[key] is not None and value is not None:
            delta = value - prev[key]
            line += f"   (was {prev[key]:.4f}, {'+' if delta >= 0 else ''}{delta:.4f})"
        print(line)
    q = summary["qa"]
    print(f"answerable={q['n_answerable']} unanswerable={q['n_unanswerable']} errors={q['n_errors']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--label", default="")
    p.add_argument("--methods", default="bm25,semantic,hybrid")
    p.add_argument("--skip-upload", action="store_true")
    p.add_argument("--no-qa", action="store_true", help="retrieval metrics only (no LLM calls)")
    p.add_argument("--only", help="comma-separated question ids")
    p.add_argument("--sleep", type=float, default=1.0, help="seconds between /ask calls (LLM rate limits)")
    p.add_argument("--compare", help="previous results JSON to diff against")
    sys.exit(run(p.parse_args()))
