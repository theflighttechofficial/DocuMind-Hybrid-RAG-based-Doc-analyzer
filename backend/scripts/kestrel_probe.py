"""Uploads the KESTREL test PDF for the eval user (replacing an older copy)
and saves the /ask JSON for two probe questions.

    python scripts/kestrel_probe.py <label> [--skip-upload]
"""
import json, sys, time
from pathlib import Path
import requests

B = "http://127.0.0.1:8000/api"
PDF = Path(__file__).resolve().parents[4] / "KESTREL_Technical_Review_2025.pdf"
QUESTIONS = ["what is this?", "What is the maximum payload of the KESTREL-7?"]

s = requests.Session()
s.headers["Authorization"] = "Bearer " + s.post(B + "/auth/login", json={"email": "eval@documind.local", "password": "documind-eval-2026"}).json()["access_token"]
docs = [d for d in s.get(B + "/documents").json() if d["name"] == PDF.name]
if "--skip-upload" in sys.argv and docs:
    doc_id = docs[0]["id"]
else:
    for d in docs:
        s.delete(f"{B}/documents/{d['id']}")
    with open(PDF, "rb") as fh:
        doc_id = s.post(B + "/documents", files={"file": (PDF.name, fh)}).json()["id"]
    while s.get(f"{B}/documents/{doc_id}").json()["processing_status"] not in ("ready", "failed"):
        time.sleep(1)
out = {q: s.post(f"{B}/documents/{doc_id}/ask", json={"question": q}).json() for q in QUESTIONS}
for r in out.values():
    r.pop("conversation_id", None)
    for e in r.get("evidence", []):
        e.pop("spans", None)  # bounding boxes: long and not what this probe checks
        e.pop("chunk_id", None)
path = Path(__file__).resolve().parents[1] / "eval_results" / "examples" / f"kestrel_{sys.argv[1]}.json"
path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(out, indent=2, ensure_ascii=False))
