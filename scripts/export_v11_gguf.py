#!/usr/bin/env python3
"""Fuse the audited MLX V11 adapter and convert it to a Windows-ready GGUF.

Run this on the Apple-silicon development machine.  The output is intentionally
ignored by Git because even a quantized 7B GGUF is too large for this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-Uncensored-4bit"
ADAPTER = ROOT / "adapters" / "legal_answer_flow_v11_specialist_lora"
EXPECTED_ADAPTER_SHA256 = "18dcd485f52b5747059c03fa0c620ccc027820d0241b04977fc4a0223679e69a"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def find_quantizer(llama_cpp: Path) -> Path:
    names = ("llama-quantize", "llama-quantize.exe", "quantize", "quantize.exe")
    roots = (llama_cpp / "build" / "bin", llama_cpp, llama_cpp / "bin")
    for root in roots:
        for name in names:
            candidate = root / name
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        "llama-quantize was not found. Build llama.cpp first; see WINDOWS.md."
    )


def restore_tokenizer_files(model: str, fused_hf: Path) -> None:
    """Use the base repository's complete HF tokenizer metadata for GGUF conversion."""
    source = Path(model).expanduser()
    if not source.is_dir():
        from huggingface_hub import snapshot_download

        source = Path(snapshot_download(
            repo_id=model,
            allow_patterns=[
                "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                "special_tokens_map.json", "added_tokens.json", "chat_template.jinja",
            ],
        ))
    copied = []
    for name in (
        "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
        "special_tokens_map.json", "added_tokens.json", "chat_template.jinja",
    ):
        candidate = source / name
        if candidate.is_file():
            shutil.copy2(candidate, fused_hf / name)
            copied.append(name)
    if not {"tokenizer.json", "tokenizer_config.json"} <= set(copied):
        raise FileNotFoundError("The base model's complete tokenizer files could not be restored.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-cpp", required=True, type=Path)
    parser.add_argument(
        "--converter-python", type=Path, default=Path(sys.executable),
        help="Python executable containing llama.cpp converter requirements.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "fused" / "v11_gguf")
    parser.add_argument("--quantization", default="Q4_K_M")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--resume-fused", action="store_true",
        help="Reuse an already completed v11_fused_hf directory after a later-stage failure.",
    )
    args = parser.parse_args()

    llama_cpp = args.llama_cpp.expanduser().resolve()
    converter = llama_cpp / "convert_hf_to_gguf.py"
    if not converter.is_file():
        raise FileNotFoundError(f"Official llama.cpp converter not found: {converter}")
    weights = ADAPTER / "adapters.safetensors"
    if not weights.is_file() or sha256(weights) != EXPECTED_ADAPTER_SHA256:
        raise RuntimeError("The audited V11 adapter is missing or failed its integrity check.")

    output = args.output_dir.expanduser().resolve()
    fused_hf = output / "v11_fused_hf"
    f16 = output / "legalchat-v11-f16.gguf"
    quantized = output / f"legalchat-v11-{args.quantization.lower()}.gguf"
    if output.exists() and any(output.iterdir()):
        if not (args.force or args.resume_fused):
            raise FileExistsError(f"Output is not empty: {output} (use --force to replace it)")
        if args.force:
            shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    if args.resume_fused:
        if not (fused_hf / "config.json").is_file() or not list(fused_hf.glob("*.safetensors")):
            raise FileNotFoundError("--resume-fused requires a completed v11_fused_hf directory")
        print(f"Reusing completed fused model: {fused_hf}", flush=True)
    else:
        run([
            sys.executable, "-m", "mlx_lm", "fuse",
            "--model", args.model,
            "--adapter-path", str(ADAPTER),
            "--save-path", str(fused_hf),
            "--dequantize",
        ])
    restore_tokenizer_files(args.model, fused_hf)
    if f16.exists():
        f16.unlink()
    if quantized.exists():
        quantized.unlink()
    run([
        str(args.converter_python.expanduser().absolute()), str(converter), str(fused_hf),
        "--outfile", str(f16), "--outtype", "f16",
    ])
    quantizer = find_quantizer(llama_cpp)
    run([str(quantizer), str(f16), str(quantized), args.quantization])

    manifest = {
        "profile": "v11-fused",
        "base_model": args.model,
        "adapter_sha256": EXPECTED_ADAPTER_SHA256,
        "gguf_sha256": sha256(quantized),
        "quantization": args.quantization,
        "conversion": "MLX fuse (dequantized) -> official llama.cpp HF-to-GGUF -> llama-quantize",
    }
    sidecar = Path(str(quantized) + ".v11.json")
    sidecar.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nWindows model: {quantized}")
    print(f"Integrity sidecar: {sidecar}")
    print("Test the GGUF against the release evaluation suite before distribution.")


if __name__ == "__main__":
    main()
