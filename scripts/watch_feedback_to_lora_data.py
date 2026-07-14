#!/usr/bin/env python3
"""Watch app feedback folders and auto-promote new records into LoRA data.

This intentionally defaults to dataset promotion only. Full LoRA training is
left as a deliberate batch action unless --train-command is supplied.
"""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROMOTER = REPO_ROOT / "scripts" / "promote_feedback_to_lora_data.py"
DEFAULT_FEEDBACK_DIR = Path(
    os.environ.get("LEGAL_FEEDBACK_SOURCE", "source-materials")
) / "user's request record for improvements"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuously promote new app feedback into the feedback v2 LoRA dataset."
    )
    parser.add_argument("--feedback-dir", type=Path, default=DEFAULT_FEEDBACK_DIR)
    parser.add_argument("--source-data-dir", type=Path, default=Path("data/legal_answer_flow_auto"))
    parser.add_argument("--output-data-dir", type=Path, default=Path("data/legal_answer_flow_feedback_v2"))
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path("training/legal_answer_flow_feedback_v2/promotion_state.json"),
    )
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument("--min-correction-chars", type=int, default=120)
    parser.add_argument("--include-comments", action="store_true")
    parser.add_argument("--once", action="store_true", help="Run one promotion pass and exit.")
    parser.add_argument(
        "--train-command",
        default="",
        help=(
            "Optional command to run after a pass promotes enough new feedback, "
            "for example './scripts/train_feedback_v2_lora.sh'. Off by default."
        ),
    )
    parser.add_argument(
        "--train-threshold",
        type=int,
        default=10,
        help="Minimum new promoted examples in one pass before --train-command runs.",
    )
    return parser.parse_args()


def feedback_signature(feedback_dir: Path) -> tuple[tuple[str, int, int], ...]:
    if not feedback_dir.exists():
        return tuple()
    rows: list[tuple[str, int, int]] = []
    for path in sorted(feedback_dir.glob("*/corrections/*.json")):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        rows.append((str(path), stat.st_size, stat.st_mtime_ns))
    return tuple(rows)


def parse_new_promoted(output: str) -> int:
    for line in output.splitlines():
        if line.startswith("new_promoted:"):
            raw = line.split(":", 1)[1].strip()
            try:
                return int(raw)
            except ValueError:
                return 0
    return 0


def run_promoter(args: argparse.Namespace) -> int:
    command = [
        sys.executable,
        str(PROMOTER),
        "--feedback-dir",
        str(args.feedback_dir),
        "--source-data-dir",
        str(args.source_data_dir),
        "--output-data-dir",
        str(args.output_data_dir),
        "--state-file",
        str(args.state_file),
        "--min-correction-chars",
        str(args.min_correction_chars),
    ]
    if args.include_comments:
        command.append("--include-comments")

    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = result.stdout.rstrip()
    if output:
        print(output, flush=True)
    if result.returncode != 0:
        print(f"Promotion command exited with code {result.returncode}", flush=True)
        return 0
    return parse_new_promoted(output)


def maybe_train(args: argparse.Namespace, new_promoted: int) -> None:
    if not args.train_command:
        return
    if new_promoted < max(1, args.train_threshold):
        print(
            f"Training skipped: {new_promoted} new promoted example(s), "
            f"threshold is {args.train_threshold}.",
            flush=True,
        )
        return
    print(f"Training trigger reached: {new_promoted} new promoted example(s).", flush=True)
    subprocess.run(
        shlex.split(args.train_command),
        cwd=REPO_ROOT,
        check=False,
    )


def main() -> int:
    args = parse_args()
    stopped = False

    def _stop(_signum: int, _frame: object) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    previous_signature: tuple[tuple[str, int, int], ...] | None = None
    print(f"Watching feedback: {args.feedback_dir}", flush=True)
    print(f"Output dataset: {args.output_data_dir}", flush=True)

    while not stopped:
        current_signature = feedback_signature(args.feedback_dir)
        if previous_signature is None or current_signature != previous_signature:
            new_promoted = run_promoter(args)
            maybe_train(args, new_promoted)
            previous_signature = current_signature
        if args.once:
            return 0
        time.sleep(max(5.0, args.interval_seconds))

    print("Feedback watcher stopped.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
