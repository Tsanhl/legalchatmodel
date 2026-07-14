#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3 scripts/promote_feedback_to_lora_data.py
source .venv/bin/activate
mlx_lm.lora --config training/legal_answer_flow_feedback_v3_clean/config.yaml
