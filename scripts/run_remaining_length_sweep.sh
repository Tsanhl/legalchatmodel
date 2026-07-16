#!/usr/bin/env bash
# Durable live length sweep for remaining 1k-20k cases.
# Skips ids already present and passing in report.jsonl.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG="${LOG:-/tmp/live_length_all.log}"
REPORT="$ROOT/training/live_private_release_sweep/report.jsonl"
BASE="${BASE:-http://127.0.0.1:8765}"

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
    cid=$(curl -fsS -X POST "$BASE/api/conversations" \
      -H 'Content-Type: application/json' \
      -d '{"mode":"private","title":"lock-wait"}' | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
    code=$(curl -sS -o /tmp/lock_wait_body.json -w '%{http_code}' \
      -X POST "$BASE/api/chat" \
      -H 'Content-Type: application/json' \
      -d "{\"conversation_id\":\"$cid\",\"message\":\"lock probe\",\"jurisdiction\":\"england_wales\"}" \
      --max-time 5 || echo 000)
    curl -fsS -X DELETE "$BASE/api/conversations/$cid" >/dev/null 2>&1 || true
    if [ "$code" != "409" ]; then
      return 0
    fi
    sleep 20
  done
  echo "lock did not free" >&2
  return 1
}

already_passed() {
  local id="$1"
  python3 - "$REPORT" "$id" <<'PY'
import json,sys
path, want = sys.argv[1], sys.argv[2]
try:
    rows=[json.loads(x) for x in open(path) if x.strip()]
except FileNotFoundError:
    raise SystemExit(1)
raise SystemExit(0 if any(r.get("id")==want and r.get("passed") for r in rows) else 1)
PY
}

echo "START $(date -Iseconds)" | tee -a "$LOG"
wait_healthy || exit 1

# Indices in scripts/final_trial_sweep QUESTIONS: 0..19 => 1000..20000
for index in $(seq 0 19); do
  words=$(( (index + 1) * 1000 ))
  # Resolve expected id prefix after a dry import
  id_prefix=$(printf 'length_%05d_' "$words")
  # Skip if any passing length_* for this word count exists
  if python3 - "$REPORT" "$id_prefix" <<'PY'
import json,sys
path, prefix = sys.argv[1], sys.argv[2]
try:
    rows=[json.loads(x) for x in open(path) if x.strip()]
except FileNotFoundError:
    raise SystemExit(1)
raise SystemExit(0 if any(str(r.get("id","")).startswith(prefix) and r.get("passed") for r in rows) else 1)
PY
  then
    echo "SKIP ${id_prefix}* already passed" | tee -a "$LOG"
    continue
  fi
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
