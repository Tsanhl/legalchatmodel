#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
python3 scripts/build_legal_v12_quality_gates_dataset.py
exec "$ROOT/.venv/bin/mlx_lm.lora" --config training/legal_answer_flow_v12_quality_gates/config.yaml
