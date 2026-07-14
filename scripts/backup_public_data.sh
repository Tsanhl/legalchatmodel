#!/usr/bin/env bash
set -euo pipefail
umask 077

# Create an encrypted public-service backup. Uses SQLite's online backup API so
# the database snapshot is consistent while the server is running, then encrypts
# the archive to an age public-key recipient. No password or private key is
# written by this script.
command -v age >/dev/null 2>&1 || {
  echo "age is required (for example: brew install age)" >&2
  exit 1
}

: "${LEGAL_BACKUP_AGE_RECIPIENT:?Set LEGAL_BACKUP_AGE_RECIPIENT to your age public recipient}"

DATA_DIR="${LEGAL_PUBLIC_DATA_DIR:-$HOME/Library/Application Support/LegalAI-public}"
DB_PATH="${LEGAL_CHAT_DB:-$DATA_DIR/chat.sqlite3}"
FEEDBACK_ROOT="${LEGAL_FEEDBACK_ROOT:-$DATA_DIR/feedback}"
UPLOAD_ROOT="${LEGAL_PRIVATE_UPLOAD_ROOT:-$DATA_DIR/uploads}"
BACKUP_DIR="${LEGAL_BACKUP_DIR:-$HOME/LegalAI-backups}"

mkdir -p "$BACKUP_DIR"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

python3 - "$DB_PATH" "$tmp/chat.sqlite3" <<'PY'
import sqlite3
import sys

source = sqlite3.connect(sys.argv[1])
target = sqlite3.connect(sys.argv[2])
with target:
    source.backup(target)
target.close()
source.close()
PY

mkdir -p "$tmp/data"
mv "$tmp/chat.sqlite3" "$tmp/data/chat.sqlite3"
if [[ -d "$FEEDBACK_ROOT" ]]; then cp -R "$FEEDBACK_ROOT" "$tmp/data/feedback"; fi
if [[ -d "$UPLOAD_ROOT" ]]; then cp -R "$UPLOAD_ROOT" "$tmp/data/uploads"; fi

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
archive="$BACKUP_DIR/legal-ai-$stamp.tar.gz.age"
tar -C "$tmp" -czf - data | age -r "$LEGAL_BACKUP_AGE_RECIPIENT" -o "$archive"
echo "$archive"
