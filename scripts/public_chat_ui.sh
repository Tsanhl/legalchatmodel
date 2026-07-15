#!/usr/bin/env bash
set -euo pipefail
umask 077

# Public pilot launcher. Cloudflare Access performs the login; this origin
# validates its signed JWT and keeps every user's records separate.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${LEGAL_PUBLIC_DATA_DIR:-$HOME/Library/Application Support/LegalAI-public}"

# The macOS service installer writes this operator-only file with mode 0600.
# Keeping deployment values outside the repository prevents tunnel/account
# configuration from being committed accidentally.
CONFIG_FILE="${LEGAL_PUBLIC_CONFIG:-$DATA_DIR/public.env}"
if [[ -f "$CONFIG_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$CONFIG_FILE"
fi
DATA_DIR="${LEGAL_PUBLIC_DATA_DIR:-$DATA_DIR}"

: "${CF_ACCESS_TEAM_DOMAIN:?Set CF_ACCESS_TEAM_DOMAIN, for example https://your-team.cloudflareaccess.com}"
: "${CF_ACCESS_AUD:?Set CF_ACCESS_AUD to the Access application Audience tag}"

mkdir -p "$DATA_DIR/feedback" "$DATA_DIR/uploads"

export LEGAL_PUBLIC_MODE=1
export LEGAL_CHAT_DB="${LEGAL_CHAT_DB:-$DATA_DIR/chat.sqlite3}"
export LEGAL_FEEDBACK_ROOT="${LEGAL_FEEDBACK_ROOT:-$DATA_DIR/feedback}"
export LEGAL_PRIVATE_UPLOAD_ROOT="${LEGAL_PRIVATE_UPLOAD_ROOT:-$DATA_DIR/uploads}"
# Never inherit the repository's private source indexes in public mode. These
# paths may later be replaced with separately reviewed, redistribution-safe
# public indexes; when absent the app uses its bundled anonymised guides.
export LEGAL_RAG_DB="${LEGAL_RAG_DB:-$DATA_DIR/public-rag.sqlite3}"
export LEGAL_GUIDANCE_DB="${LEGAL_GUIDANCE_DB:-$DATA_DIR/public-guidance.sqlite3}"

cd "$ROOT_DIR"
exec ./scripts/chat_ui.sh --host 127.0.0.1 "$@"
