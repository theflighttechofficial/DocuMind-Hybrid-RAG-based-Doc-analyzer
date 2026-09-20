# Continuing DocuMind AI with Claude Code

This project's MVP tier is complete and working end-to-end: upload, extraction,
NLP analysis (NER/POS/keywords/statistics), chunking, embeddings, FAISS + BM25 +
hybrid retrieval, grounded QA with conversation memory, hierarchical summarization,
semantic document comparison, an entity co-occurrence knowledge graph, and a real
evaluation dashboard (Precision@K/Recall@K/MRR/F1/ROUGE) — all backed by genuine
computation, no fabricated data.

Paste the prompt below into Claude Code (in this project's root directory) to
continue with the "SHOULD HAVE" and "NICE TO HAVE" tiers from the original spec.

---

## Prompt to paste into Claude Code

```
I have a working DocuMind AI project (FastAPI backend in backend/, React+Vite
frontend in frontend/). Read README.md first for full architecture context, then
continue building the remaining features from the original spec, in this order:

PHASE 1 — Multi-document workspaces and cross-document QA
- Add a `workspaces` table (id, name, created_at) and a `workspace_documents`
  join table.
- Add API endpoints: create workspace, add/remove documents, and a
  `/api/workspaces/{id}/ask` endpoint that retrieves from ALL documents in the
  workspace (run retrieval per-document, merge and re-rank by score, tag each
  evidence item with which document it came from).
- Add a Workspaces page in the frontend: create a workspace, multi-select
  documents into it, and a workspace-scoped version of the Ask interface that
  shows which document each evidence card came from.

PHASE 2 — Reranking stage
- In backend/app/retrieval/, add a reranker.py using a cross-encoder model
  (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2 via sentence-transformers) that
  re-scores the top ~20 hybrid retrieval candidates before truncating to top_k.
  Wire it into hybrid.py as an optional step, and expose a toggle for it in the
  Evaluation dashboard so reranked vs non-reranked metrics can be compared.

PHASE 3 — OCR fallback for scanned PDFs
- In backend/app/nlp/extraction.py, detect when a PDF page's extracted text is
  empty/near-empty (scanned page) and fall back to OCR via pytesseract + pdf2image
  for those pages specifically. Update the processing pipeline's error message
  to distinguish "no OCR available" from "truly empty" documents. Add
  pytesseract/pdf2image to requirements.txt and note the system-level tesseract
  dependency in README.md.

PHASE 4 — Streaming LLM responses
- Update qa/llm_client.py to support streaming via Gemini's streaming API.
- Add a streaming variant of the /ask endpoint using Server-Sent Events or
  chunked responses.
- Update the frontend Ask.jsx to render tokens as they arrive instead of
  waiting for the full response.

PHASE 5 — Auth and per-user document isolation
- Add a `users` table and simple JWT-based auth (register/login endpoints).
- Add `user_id` to the documents table and filter all document queries by the
  authenticated user.
- Add a login/register page to the frontend and store the JWT in memory
  (not localStorage, to avoid XSS token theft), attaching it to all API calls.

PHASE 6 — Polish
- Add a light theme toggle (the current design is dark-only) using the same
  token system in tailwind.config.js — add a light-mode palette and a
  prefers-color-scheme-aware default with a manual override switch in Settings.
- Add a Settings page (currently missing) with the theme toggle and
  Gemini API key management (masked input, "test connection" button).
- Write a docker-compose.yml that runs backend + frontend together for
  one-command setup.

Work through these phases in order. After each phase, run the backend's existing
patterns (compile-check, then a quick end-to-end smoke test using the demo data
in /demo-data) before moving to the next phase. Ask me before starting Phase 5
(auth) since it changes the request/response contract for every existing endpoint.
```
