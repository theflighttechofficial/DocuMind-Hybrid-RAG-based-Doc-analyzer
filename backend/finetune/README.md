# DocuMind local answer model

DocuMind answers questions with a local LLM served by [Ollama](https://ollama.com).
No document text is sent to any external API. The answer model is a LoRA
fine-tune of Qwen2.5-Instruct, trained to follow DocuMind's answering rules:

- read numbered passages in the exact format the backend sends;
- answer only from those passages, with inline `[n]` citations;
- reply with the exact refusal sentence when the passages don't answer;
- start with the answer, with no "Based on the provided documents" opener.

The backend uses `OLLAMA_MODEL` (default `documind-qwen2.5-1.5b`). If that
model is not installed, it falls back to the stock `qwen2.5:1.5b-instruct`.
`GET /api/system/config` shows which model is active.

## Training data

`build_sft_data.py` builds 1,600 training and 120 validation chats from SQuAD v2.
SQuAD v2 has human-written questions over Wikipedia paragraphs, including
questions that the paragraph cannot answer.

- Each example shows 3–4 passages from one article: the gold paragraph and
  same-article distractors, in random order.
- The target is the answer sentence (or the short answer) with the gold
  passage number, for example `Walvis Bay [2].`
- About 27% of the examples are unanswerable. Their target is the exact
  refusal sentence.
- The DocuMind demo documents and the eval set are **not** used. The
  63-question eval set stays a held-out test.

## Run on a CUDA machine (RTX 5050, 8 GB)

Run all commands from `backend/`.

1. Install a PyTorch build for your GPU. RTX 50-series (Blackwell) cards need a
   CUDA 12.8 build:

   ```powershell
   pip install torch --index-url https://download.pytorch.org/whl/cu128
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

   Keep `numpy==1.26.4` (faiss-cpu and spaCy need it).

2. Install the rest and the base models:

   ```powershell
   pip install -r requirements.txt
   ollama pull qwen2.5:1.5b-instruct    # fallback model, also the "stock" comparison
   ```

3. Build the training data (downloads SQuAD v2 from Hugging Face once):

   ```powershell
   python finetune/build_sft_data.py --unanswerable-share 0.42
   ```

4. Time a few steps, then train:

   ```powershell
   python finetune/train_lora.py --dry-run
   python finetune/train_lora.py            # Qwen2.5-1.5B, 300 steps x 8 examples
   ```

   The script uses CUDA in bf16 automatically. It logs validation loss every 50
   steps to `finetune/out/<name>/train_log.json`, and saves the adapter at each
   validation. For a 0.5B model on 8 GB, add `--no-grad-checkpointing` for speed.

5. Package the model for Ollama and point the backend at it:

   ```powershell
   python finetune/build_ollama_model.py --name documind-qwen2.5-1.5b --quantize q8_0
   # backend/.env:  OLLAMA_MODEL=documind-qwen2.5-1.5b
   ```

6. Verify against stock with the held-out eval. Run the backend once per model:

   ```powershell
   $env:OLLAMA_MODEL="qwen2.5:1.5b-instruct"; python -m uvicorn app.main:app --port 8000
   python scripts/run_eval.py --label stock-1.5b

   $env:OLLAMA_MODEL="documind-qwen2.5-1.5b"; python -m uvicorn app.main:app --port 8000
   python scripts/run_eval.py --label finetuned-1.5b --skip-upload --compare eval_results/<stock-1.5b>.json
   ```

   Keep the fine-tune only if it beats stock on answer accuracy and refusal
   accuracy.

## Results so far (CPU-only laptop: i7-1165G7, 16 GB, no GPU)

- Pipeline smoke test passed for train → merge → `ollama create` → backend
  `/ask`: 1 training step on Qwen2.5-0.5B, packaged as `documind-smoke`
  (531 MB, q8_0). After the template fix it answered
  `"The maximum payload of the KESTREL-7 is 4.5 kg [2]."`
- CPU training speed is about 30 s per example for 0.5B. A full run takes many
  hours, so the real training run moves to the GPU machine.
- Stock `qwen2.5:1.5b-instruct` on a 13-question eval subset
  (`eval_results/20261004-213836-local-1.5b-stock-subset.json`):
  - answer accuracy 8/11;
  - refusal accuracy 2/2;
  - shown evidence contains the answer: 1.00;
  - about 20 s per answer.

  It missed two table lookups and one follow-up, and wrote inline citations
  only sometimes.
- 0.5B models (stock and 1-step fine-tune) echo the passages back for a vague
  question like "what is this?". Stock 1.5B answers it, so 1.5B is the default.

## Pitfalls already fixed

- **CRLF Modelfile on Windows.** It put `\r\n` into every ChatML turn, and the
  model echoed its input. `build_ollama_model.py` now writes LF only.
- **Long parent sections overflow small context windows.** Each passage sent to
  the LLM is now capped at `LLM_PASSAGE_MAX_WORDS` (220).
