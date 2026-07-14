#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python3 scripts/watch_feedback_to_lora_data.py "$@"
