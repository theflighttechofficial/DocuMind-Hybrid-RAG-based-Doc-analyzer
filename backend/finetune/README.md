# DocuMind local answer model

DocuMind answers questions with a local LLM served by [Ollama](https://ollama.com).
No document text is sent to any external API.

The primary model is **Qwen3-4B-Instruct-2507** (`qwen3:4b-instruct` in Ollama,
Q4_K_M, 2.5 GB). Use the instruct variant. The plain `qwen3:4b` tag is the
*thinking* model: it ignores `think: false` and writes ~250 tokens of reasoning
before every answer (about 60 s on a laptop CPU).

The backend serves `OLLAMA_MODEL` (default `documind-qwen3-4b`). If that model
isn't installed, it falls back to stock `qwen3:4b-instruct`.
`GET /api/system/config` shows which model is active.

## What we want the model to learn

The model should follow DocuMind's answering contract:

- read the numbered passages in the exact format the backend sends;
- answer only from them, with inline `[n]` citations;
- reply with the exact refusal sentence when the passages don't answer;
- start with the answer, with no "Based on the provided documents" opener.

This is a small, format-level change. The training is deliberately light so
it doesn't overwrite what the base model already does well.

There are two levels of adaptation.

### 1. Prompt-level (no GPU): `build_fewshot_model.py`

This script bakes the rules and two worked examples (from SQuAD v2) into an
Ollama model. No weights change.

```powershell
ollama pull qwen3:4b-instruct
python finetune/build_fewshot_model.py --base qwen3:4b-instruct --name documind-qwen3-4b
```

### 2. Weight-level, light (CUDA GPU): `train_lora.py`

This is QLoRA on Qwen3-4B with 60 steps × 4 examples, i.e. 240 SQuAD-v2
examples. A 4B model doesn't fit in CPU training memory, so this step needs
the RTX 5050 machine.

## Training data

`build_sft_data.py` builds training and validation chats from SQuAD v2:
human-written questions over Wikipedia paragraphs, including questions the
paragraph can't answer.

- Each example shows 3–4 passages from one article (the gold paragraph and
  same-article distractors), in random order.
- The target is the answer sentence or short answer with the gold passage
  number, for example `Walvis Bay [2].`
- About 27% of the examples are unanswerable. Their target is the exact
  refusal sentence.
- The DocuMind demo documents and the 63-question eval set are **not** used,
  so the eval stays a held-out test.

## Run on the RTX 5050 machine (8 GB)

Run all commands from `backend/`.

1. Install a CUDA 12.8 PyTorch build. RTX 50-series (Blackwell) cards need it.
   Then install bitsandbytes for 4-bit loading:

   ```powershell
   pip install torch --index-url https://download.pytorch.org/whl/cu128
   pip install bitsandbytes
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

   Keep `numpy==1.26.4` (faiss-cpu and spaCy need it).

2. Install the rest and the base model:

   ```powershell
   pip install -r requirements.txt
   ollama pull qwen3:4b-instruct
   ```

3. Build the data (downloads SQuAD v2 once), then time a few steps:

   ```powershell
   python finetune/build_sft_data.py --unanswerable-share 0.42
   python finetune/train_lora.py --dry-run
   ```

4. Train:

   ```powershell
   python finetune/train_lora.py
   ```

   Defaults: `Qwen/Qwen3-4B-Instruct-2507`, QLoRA (nf4), LoRA r=16, 60 steps ×
   4 examples, validation loss every 50 steps. Output goes to
   `finetune/out/documind-qwen3-4b-lora/`. After training, the adapter is
   merged into a bf16 copy of the base on the CPU, which needs about 10 GB of
   free RAM.

5. Package the model for Ollama:

   ```powershell
   python finetune/build_ollama_model.py --name documind-qwen3-4b-lora --quantize q4_K_M
   ```

6. Compare models on the held-out eval. Restart the backend once per model:

   ```powershell
   $env:OLLAMA_MODEL="qwen3:4b-instruct";      python -m uvicorn app.main:app --port 8000
   python scripts/run_eval.py --label qwen3-4b-stock

   $env:OLLAMA_MODEL="documind-qwen3-4b-lora"; python -m uvicorn app.main:app --port 8000
   python scripts/run_eval.py --label qwen3-4b-lora --skip-upload --compare eval_results/<qwen3-4b-stock>.json

   python scripts/kestrel_probe.py qwen3-4b-lora --skip-upload
   ```

   Make the winner the default: set `OLLAMA_MODEL=<name>` in `backend/.env`.
   Also check the vague question "what is this?" in the KESTREL probe. It is
   where the few-shot version regressed (see below).

## Results on the CPU-only laptop (i7-1165G7, 16 GB, no GPU)

- `qwen3:4b-instruct`: "The maximum payload is 4.5 kg [1]." in 9.5 s for a
  short prompt. Real Ask prompts take about 60–90 s per answer, because CPU
  prompt processing runs at about 30 tokens/s.
- KESTREL probe:

  | Model | "what is this?" | "maximum payload of the KESTREL-7?" |
  |---|---|---|
  | stock `qwen3:4b-instruct` | Correct overview of Project KESTREL, cited [1]–[6] | "4.5 kg [1]" |
  | `documind-qwen3-4b` (few-shot) | **Refused** (the refusal sentence) | "4.5 kg [1]" |

  The few-shot examples made the model refuse vague questions. This happened
  with and without a worked refusal example.
- Eval-subset comparison: see `eval_results/*qwen3-4b-*-subset.json`.
- Earlier smoke test: a 1-step LoRA on Qwen2.5-0.5B went through train →
  merge → `ollama create` → backend `/ask` and answered with a correct
  citation. CPU training speed is about 30 s per example for 0.5B, which is
  why weight training needs the GPU.

## Pitfalls already fixed

- **CRLF Modelfile on Windows.** It put `\r\n` into every ChatML turn, and
  the model echoed its input. Both build scripts now write LF only.
- **Thinking-model tag.** `qwen3:4b` reasons before every answer even with
  `think: false`. Use `qwen3:4b-instruct`.
- **Long parent sections overflow small context windows.** Each passage is
  capped at `LLM_PASSAGE_MAX_WORDS` (220).
