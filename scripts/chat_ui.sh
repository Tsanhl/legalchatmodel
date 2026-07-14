#!/usr/bin/env bash
set -euo pipefail

# Launch the local legal chat web UI (uses the fine-tuned LoRA adapter).
# Usage:
#   ./scripts/chat_ui.sh                 # serve on http://127.0.0.1:8765
#   ./scripts/chat_ui.sh --port 9000     # custom port
#   ./scripts/chat_ui.sh --no-adapter    # chat with the base model only

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source .venv/bin/activate
exec python legal_chat_ui/server.py "$@"
