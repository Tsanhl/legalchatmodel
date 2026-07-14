#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source .venv/bin/activate
mlx_lm.server \
  --model "mlx-community/Qwen2.5-7B-Instruct-Uncensored-4bit" \
  --adapter-path adapters/qwen2_5_7b_lora \
  --host 127.0.0.1 \
  --port 8000 \
  "$@"

