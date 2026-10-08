# Training the DocuMind answer model

These steps train DocuMind's local answer model on a machine with an NVIDIA
GPU (tested target: RTX 5050, 8 GB). The base model is Qwen3-4B-Instruct-2507.
Training is deliberately light: 240 examples that teach only the answer
format (inline `[n]` citations, the exact refusal sentence, short answers).

For background, results and pitfalls, see `backend/finetune/README.md`.

## 0. What you need

- Windows or Linux with an NVIDIA GPU and a recent driver
- Python 3.12, Git, [Ollama](https://ollama.com/download)
- About 15 GB free disk and about 10 GB free RAM (the final merge step runs on the CPU)

## 1. Get the code

```powershell
git clone https://github.com/theflighttechofficial/DocuMind-Hybrid-RAG-based-Doc-analyzer.git documind
cd documind\backend
```

If you already cloned it, run `git pull` instead.

## 2. Install PyTorch for your GPU, then the rest

RTX 50-series (Blackwell) cards need the CUDA 12.8 build. Install it **before**
`requirements.txt` so pip doesn't install the CPU build instead.

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install bitsandbytes
python -m spacy download en_core_web_lg
```

Check that the GPU is visible. This must print `True` and the GPU name:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Keep `numpy==1.26.4`. If any install upgrades it, run `pip install numpy==1.26.4`.

## 3. Pull the base model into Ollama

```powershell
ollama pull qwen3:4b-instruct
```

Use `qwen3:4b-instruct`, not `qwen3:4b`. The plain tag is the "thinking"
variant, which writes long reasoning before every answer.

## 4. Build the training data

This downloads SQuAD v2 from Hugging Face once. It writes
`finetune/data/train.jsonl` (1,600 examples) and `finetune/data/val.jsonl` (120).

```powershell
python finetune/build_sft_data.py --unanswerable-share 0.42
```

## 5. Time a few steps

```powershell
python finetune/train_lora.py --dry-run
```

You should see `device cuda`, `loaded in 4-bit (QLoRA)` and three `step` lines
with seconds per step. If it says `device cpu`, go back to step 2.

## 6. Train

```powershell
python finetune/train_lora.py
```

Defaults:

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen3-4B-Instruct-2507` |
| Method | 4-bit QLoRA, LoRA rank 16 |
| Run length | 60 steps × 4 examples = 240 examples |
| Validation loss | logged every 50 steps |

Results go to `finetune/out/documind-qwen3-4b-lora/`:

- `train_log.json`: loss per step. Validation loss should go down.
- `adapter/`: the LoRA weights.
- `merged/`: the full model, ready for Ollama.

If you run out of GPU memory, add `--max-len 1024`.

## 7. Package the model for Ollama

```powershell
python finetune/build_ollama_model.py --name documind-qwen3-4b-lora --quantize q4_K_M
ollama list          # documind-qwen3-4b-lora should be listed
```

## 8. Check that it beats the stock model

First upload the demo documents and score the stock model:

```powershell
$env:OLLAMA_MODEL="qwen3:4b-instruct"
python -m uvicorn app.main:app --port 8000
# in a second terminal:
python scripts/run_eval.py --label qwen3-4b-stock
```

Then stop the server (Ctrl+C) and score the trained model:

```powershell
$env:OLLAMA_MODEL="documind-qwen3-4b-lora"
python -m uvicorn app.main:app --port 8000
# in a second terminal:
python scripts/run_eval.py --label qwen3-4b-lora --skip-upload --compare eval_results/<the qwen3-4b-stock file>.json
python scripts/kestrel_probe.py qwen3-4b-lora --skip-upload
```

Keep the trained model only if it is at least as good as stock on all three:

- `qa.answer_accuracy`
- `qa.refusal_accuracy`
- the vague KESTREL question "what is this?". The trained model must answer
  it, not refuse.

## 9. Make it the default and share the results

In `backend/.env`:

```
OLLAMA_MODEL=documind-qwen3-4b-lora
```

Commit the eval results and push:

```powershell
git add backend/eval_results
git commit -m "eval: qwen3-4b lora vs stock"
git push origin main
```

`finetune/out/` (the trained weights) is gitignored because it is several
GB. To use the trained model on another machine, copy that folder by hand and
run step 7 there.
