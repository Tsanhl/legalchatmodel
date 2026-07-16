#!/usr/bin/env bash
# Durable live length sweep for remaining 1k-20k cases.
# Skips ids already present and passing in report.jsonl. Retries failures.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG="${LOG:-/tmp/live_length_all.log}"
REPORT="$ROOT/training/live_private_release_sweep/report.jsonl"
BASE="${BASE:-http://127.0.0.1:8765}"
LOCKFILE="${LOCKFILE:-/tmp/legal_length_sweep.lock}"

if [ -f "$LOCKFILE" ]; then
  old_pid=$(cat "$LOCKFILE" 2>/dev/null || true)
  if [ -n "${old_pid:-}" ] && kill -0 "$old_pid" 2>/dev/null; then
    echo "another length sweep holds $LOCKFILE (pid $old_pid); exiting" | tee -a "$LOG"
    exit 0
  fi
fi
echo $$ > "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT

wait_healthy() {
  for _ in $(seq 1 90); do
    if curl -fsS "$BASE/api/health" 2>/dev/null | grep -q '"ready": true'; then
      return 0
    fi
    sleep 2
  done
  echo "server not healthy" >&2
  return 1
}

wait_lock_free() {
  for _ in $(seq 1 360); do
    busy=$(curl -fsS "$BASE/api/busy" 2>/dev/null | python3 -c 'import sys,json; print(json.load(sys.stdin).get("busy", True))' 2>/dev/null || echo True)
    if [ "$busy" = "False" ] || [ "$busy" = "false" ]; then
      return 0
    fi
    sleep 20
  done
  echo "lock did not free" >&2
  return 1
}

echo "START $(date -Iseconds) pid=$$" | tee -a "$LOG"
wait_healthy || exit 1

for index in $(seq 0 19); do
  words=$(( (index + 1) * 1000 ))
  id_prefix=$(printf 'length_%05d_' "$words")
  if python3 - "$REPORT" "$id_prefix" <<'PY'
import json,sys
path, prefix = sys.argv[1], sys.argv[2]
try:
    rows=[json.loads(x) for x in open(path) if x.strip()]
except FileNotFoundError:
    raise SystemExit(1)
# Only skip a true pass; failed/partial rows must be retried.
raise SystemExit(0 if any(str(r.get("id","")).startswith(prefix) and r.get("passed") for r in rows) else 1)
PY
  then
    echo "SKIP ${id_prefix}* already passed" | tee -a "$LOG"
    continue
  fi
  # Drop prior failed rows for this prefix so the report stays clean.
  python3 - "$REPORT" "$id_prefix" <<'PY'
import json,sys
from pathlib import Path
path, prefix = Path(sys.argv[1]), sys.argv[2]
if not path.exists():
    raise SystemExit(0)
rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
kept=[r for r in rows if not (str(r.get("id","")).startswith(prefix) and not r.get("passed"))]
path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in kept))
PY
  echo "RUN index=$index words=$words $(date -Iseconds)" | tee -a "$LOG"
  wait_lock_free || exit 1
  .venv/bin/python -u scripts/live_private_release_sweep.py --lengths --start "$index" --stop $((index + 1)) 2>&1 | tee -a "$LOG"
done

echo "END $(date -Iseconds)" | tee -a "$LOG"
.venv/bin/python - <<'PY'
import json,pathlib
rows=[json.loads(x) for x in pathlib.Path("training/live_private_release_sweep/report.jsonl").read_text().splitlines() if x.strip()]
lat={r["id"]: r for r in rows}
lens=sorted([r for r in lat.values() if str(r.get("id","")).startswith("length_")], key=lambda r: r["id"])
print("length cases", len(lens), "passed", sum(r.get("passed", False) for r in lens))
for r in lens:
    print(r.get("id"), r.get("passed"), r.get("body_words"), r.get("failures"))
PY
