"""Packages the merged fine-tune as an Ollama model so the backend can serve
it locally (settings.ollama_model).

    python finetune/build_ollama_model.py --name documind-qwen2.5-1.5b --quantize q8_0
"""
import argparse
import subprocess
from pathlib import Path

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="documind-qwen2.5-1.5b")
    ap.add_argument("--quantize", default="none", help="Ollama quantize type: int4, int8, nvfp4, mxfp4, mxfp8, or 'none' to skip (required on Windows/NVIDIA where MLX is unavailable)")
    args = ap.parse_args()
    merged = (HERE / "out" / args.name / "merged").resolve()
    if not (merged / "config.json").exists():
        raise SystemExit(f"{merged} not found; run finetune/train_lora.py first")
    modelfile = HERE / "out" / args.name / "Modelfile"
    # LF only: on Windows a CRLF template puts "\r\n" into every ChatML turn,
    # which the model never saw in training (it then echoes the passages back)
    template = (HERE / "Modelfile.template").read_text(encoding="utf-8").replace("\r\n", "\n")
    # If GGUF file exists, point FROM to it directly (necessary for Windows/Ollama)
    gguf_path = HERE / "out" / args.name / "model-f16.gguf"
    model_source = gguf_path.resolve().as_posix() if gguf_path.exists() else merged.as_posix()
    modelfile.write_text(template.replace("{MERGED_DIR}", model_source), encoding="utf-8", newline="\n")
    # Map legacy GGUF quant names to the types supported by newer Ollama builds.
    _QUANT_MAP = {
        "q4_0": "int4", "q4_1": "int4", "q4_k_m": "int4", "q4_k_s": "int4",
        "q5_0": "int8", "q5_1": "int8", "q5_k_m": "int8", "q5_k_s": "int8",
        "q6_k": "int8", "q8_0": "int8",
    }
    quant = _QUANT_MAP.get(args.quantize.lower(), args.quantize)
    # 'none' or empty string means skip --quantize (required on Windows/NVIDIA: MLX not available)
    cmd = ["ollama", "create", args.name, "-f", str(modelfile)]
    if quant and quant.lower() != "none":
        cmd += ["--quantize", quant]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
