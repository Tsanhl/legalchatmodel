#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source .venv/bin/activate
mlx_lm.lora \
  --config training/legal_answer_flow_auto/config.yaml \
  --resume-adapter-file adapters/legal_answer_flow_auto_lora/adapters.safetensors \
  --adapter-path adapters/legal_answer_flow_auto_lora_400 \
  --iters 200 \
  "$@"
