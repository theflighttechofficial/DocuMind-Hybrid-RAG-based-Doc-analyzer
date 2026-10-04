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
    ap.add_argument("--quantize", default="q8_0", help="q8_0 keeps a 0.5B model close to fp16 quality")
    args = ap.parse_args()
    merged = (HERE / "out" / args.name / "merged").resolve()
    if not (merged / "config.json").exists():
        raise SystemExit(f"{merged} not found; run finetune/train_lora.py first")
    modelfile = HERE / "out" / args.name / "Modelfile"
    # LF only: on Windows a CRLF template puts "\r\n" into every ChatML turn,
    # which the model never saw in training (it then echoes the passages back)
    template = (HERE / "Modelfile.template").read_text(encoding="utf-8").replace("\r\n", "\n")
    modelfile.write_text(template.replace("{MERGED_DIR}", merged.as_posix()), encoding="utf-8", newline="\n")
    cmd = ["ollama", "create", args.name, "-f", str(modelfile)]
    if args.quantize:
        cmd += ["--quantize", args.quantize]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
