# DocuMind - one-command training on the RTX 5050 (follows TRAINING.md steps 2-7).
# Run from the repo root:   powershell -ExecutionPolicy Bypass -File .\train.ps1
# Re-running is safe: finished steps are skipped. Log: train_run.log

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Backend = Join-Path $Root "backend"
$Log = Join-Path $Root "train_run.log"
Start-Transcript -Path $Log -Append | Out-Null

function Step($n, $msg) { Write-Host ""; Write-Host "=== STEP $n : $msg ===" -ForegroundColor Cyan }
function Check($ok, $msg) { if (-not $ok) { Write-Host "FAILED: $msg" -ForegroundColor Red; Stop-Transcript | Out-Null; exit 1 } }

try {
    Step 0 "Checking prerequisites"
    Check (Get-Command nvidia-smi -ErrorAction SilentlyContinue) "nvidia-smi not found - install/update the NVIDIA driver."
    nvidia-smi
    Check (Get-Command ollama -ErrorAction SilentlyContinue) "Ollama not found - install it from https://ollama.com/download and reopen PowerShell."
    $py = $null
    if (Get-Command py -ErrorAction SilentlyContinue) { try { & py -3.12 --version | Out-Null; $py = @("py", "-3.12") } catch {} }
    if (-not $py) { $py = @("python") }
    $ver = & $py[0] $py[1..9] --version
    Write-Host "Using $ver"
    Check ($ver -match "3\.12") "Python 3.12 is required (found: $ver). Install it from python.org."

    Set-Location $Backend

    Step 1 "Creating the virtual environment (backend\venv)"
    if (-not (Test-Path "venv\Scripts\python.exe")) { & $py[0] $py[1..9] -m venv venv }
    $Py = Join-Path $Backend "venv\Scripts\python.exe"
    & $Py -m pip install --upgrade pip

    Step 2 "Installing PyTorch (CUDA 12.8) and dependencies"
    $hasCuda = "False"
    try { $hasCuda = & $Py -c "import torch,sys; sys.stdout.write(str(torch.cuda.is_available()))" 2>$null } catch {}
    if ($hasCuda -ne "True") { & $Py -m pip install torch --index-url https://download.pytorch.org/whl/cu128 }
    & $Py -m pip install -r requirements.txt
    & $Py -m pip install bitsandbytes
    & $Py -m pip install numpy==1.26.4
    & $Py -m spacy download en_core_web_lg

    Step 3 "Checking the GPU from PyTorch"
    $gpu = & $Py -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
    Write-Host $gpu
    Check ($gpu -match "^True") "PyTorch cannot see the GPU. Update the NVIDIA driver (CUDA 12.8+) and re-run."

    Step 4 "Pulling the base model into Ollama"
    ollama pull qwen3:4b-instruct
    Check ($LASTEXITCODE -eq 0) "ollama pull failed - is the Ollama app running?"

    Step 5 "Building the training data"
    if (-not (Test-Path "finetune\data\train.jsonl")) {
        & $Py finetune/build_sft_data.py --unanswerable-share 0.42
        Check ($LASTEXITCODE -eq 0) "build_sft_data.py failed"
    } else { Write-Host "train.jsonl already exists - skipping" }

    Step 6 "Dry run (timing a few steps)"
    & $Py finetune/train_lora.py --dry-run
    Check ($LASTEXITCODE -eq 0) "Dry run failed - see the error above."

    Step 7 "TRAINING (this is the long part)"
    if (-not (Test-Path "finetune\out\documind-qwen3-4b-lora\merged")) {
        & $Py finetune/train_lora.py
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Training failed - retrying with --max-len 1024 (lower GPU memory)" -ForegroundColor Yellow
            & $Py finetune/train_lora.py --max-len 1024
            Check ($LASTEXITCODE -eq 0) "Training failed twice - send train_run.log to Claude."
        }
    } else { Write-Host "Merged model already exists - skipping training" }

    Step 8 "Packaging the model for Ollama"
    & $Py finetune/build_ollama_model.py --name documind-qwen3-4b-lora --quantize none
    Check ($LASTEXITCODE -eq 0) "build_ollama_model.py failed"
    ollama list

    Write-Host ""
    Write-Host "DONE. Your trained model 'documind-qwen3-4b-lora' is in Ollama." -ForegroundColor Green
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
}
finally {
    Stop-Transcript | Out-Null
}
