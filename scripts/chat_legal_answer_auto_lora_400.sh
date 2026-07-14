#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source .venv/bin/activate
mlx_lm.chat \
  --model mlx-community/Qwen2.5-7B-Instruct-Uncensored-4bit \
  --adapter-path adapters/legal_answer_flow_auto_lora_400 \
  "$@"
