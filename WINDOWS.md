# Windows support

LegalChatModel can run locally on 64-bit Windows 10/11. The application, legal
RAG, current-law checks, Memory/Private isolation, SQLite storage, supervision,
OSCOLA controls and word-count planner are the same as on macOS. Only the
inference engine changes:

- Apple silicon uses MLX plus the audited V11 LoRA adapter.
- Windows uses a local
  [llama.cpp `llama-server`](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
  plus a GGUF model through its OpenAI-compatible local endpoint.

No question or answer is sent to an external AI API by this Windows path.
The backend rejects non-loopback model URLs by default.

## What “supported” means

There are two explicit model profiles:

1. `base`: any compatible Qwen2.5 7B Instruct GGUF. The site works, but this is
   not the legal V11 fine-tune and should not be advertised as equal quality.
2. `v11-fused`: the audited V11 adapter has been fused into its matching base,
   converted to GGUF, release-tested, and accompanied by the generated SHA-256
   sidecar. This is the intended Windows-equivalent model profile.

The repository contains the Windows code and conversion path. It does not
contain a multi-gigabyte fused GGUF. GitHub is not suitable model hosting, so an
operator must export, test and host that separate artefact before a one-click
V11 download can be offered.

## Requirements

- 64-bit Windows 10 or 11;
- Python 3.10 or newer;
- a current official Windows build of
  [llama.cpp](https://github.com/ggml-org/llama.cpp/releases), including
  `llama-server.exe`;
- a compatible GGUF model;
- about 16 GB system RAM as a practical starting point for a 7B Q4/Q5 model;
  an NVIDIA/AMD/Intel accelerator is optional, subject to the llama.cpp build.

The matching upstream model is
[Orion-zhen/Qwen2.5-7B-Instruct-Uncensored](https://huggingface.co/Orion-zhen/Qwen2.5-7B-Instruct-Uncensored).
Its model card links third-party GGUF quantizations. Check the model and
quantization licences and hashes before distribution.

## Install and run

Open PowerShell in the cloned repository:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1

$env:LEGAL_LLAMA_SERVER = "C:\path\to\llama-server.exe"
$env:LEGAL_GGUF_MODEL = "C:\path\to\model.gguf"
.\scripts\chat_ui_windows.ps1
```

The launcher starts both local processes, waits for the model health check,
opens <http://127.0.0.1:8765/>, and stops `llama-server` when the UI exits.
It generates an ephemeral API key shared only by the two local processes and
restricts llama.cpp CORS to localhost.

To offload layers on a supported GPU build, set the number accepted by that
build before launch:

```powershell
$env:LEGAL_LLAMA_GPU_LAYERS = "99"
.\scripts\chat_ui_windows.ps1
```

For an audited V11-fused file and its adjacent `.v11.json` sidecar:

```powershell
.\scripts\chat_ui_windows.ps1 -ModelProfile v11-fused
```

The launcher recomputes the GGUF SHA-256 and refuses the V11 label if the
sidecar is missing, malformed or does not match.

## Export the V11 Windows model (operator)

This conversion must be run on the Apple-silicon development machine because
the source adapter is in MLX format. Clone and build the official llama.cpp
repository first, then run:

```bash
source .venv/bin/activate
python3 scripts/export_v11_gguf.py --llama-cpp /path/to/llama.cpp
```

If llama.cpp's pinned converter packages conflict with the MLX environment,
install its conversion requirements in a separate virtual environment and pass
that interpreter with `--converter-python /path/to/venv/bin/python`.
If fusion completed but a later conversion step failed, correct the toolchain
and add `--resume-fused` to reuse the completed weight shards.

The exporter:

1. verifies the pinned V11 adapter SHA-256;
2. fuses and dequantizes the adapter with MLX;
3. restores the base repository's complete tokenizer metadata and uses
   llama.cpp's official `convert_hf_to_gguf.py`;
4. quantizes with `llama-quantize` (default `Q4_K_M`);
5. writes the `.v11.json` integrity sidecar.

Output stays under the gitignored `fused/` folder. Before distribution, run
the full legal evaluation bank against the actual GGUF and compare it with the
approved MLX V11 release. Conversion correctness does not itself establish
quality parity.

## Verification

The transport/backend test uses a fake local OpenAI-compatible server and does
not need model weights:

```powershell
.\.venv\Scripts\python.exe scripts\verify_windows_backend.py
```

GitHub Actions runs that test on `windows-latest` on every pull request and
push to `main`. Final hardware validation must still be performed on real
Windows CPU/GPU configurations with the exported GGUF.
