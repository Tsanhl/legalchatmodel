#!/usr/bin/env python3
"""Build a training set that gives verified full-answer feedback useful weight."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from promote_feedback_to_lora_data import (  # noqa: E402
    DEFAULT_FEEDBACK_DIR,
    feedback_paths,
    looks_trainable,
    read_json,
    training_example,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a feedback-weighted legal LoRA curriculum.")
    parser.add_argument("--base-data-dir", type=Path, default=Path("data/legal_answer_flow_feedback_v2"))
    parser.add_argument("--feedback-dir", type=Path, default=DEFAULT_FEEDBACK_DIR)
    parser.add_argument("--output-data-dir", type=Path, default=Path("data/legal_answer_flow_v4_curriculum"))
    parser.add_argument("--repeat", type=int, default=40)
    return parser.parse_args()


def read_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    curated: list[str] = []
    for path in feedback_paths(args.feedback_dir):
        payload = read_json(path)
        trainable, _reason = looks_trainable(payload, min_chars=120, include_comments=False)
        if trainable:
            curated.append(json.dumps(training_example(payload), ensure_ascii=False, separators=(",", ":")))
    if not curated:
        raise SystemExit("No verified full-answer feedback is available for the curriculum.")

    repeats = max(1, args.repeat)
    base_train = read_lines(args.base_data_dir / "train.jsonl")
    # Place the curated examples first so a short incremental run learns from
    # them even when the trainer does not complete a full dataset epoch.
    train_lines = curated * repeats + base_train
    write_lines(args.output_data_dir / "train.jsonl", train_lines)
    for split in ("valid", "test"):
        write_lines(args.output_data_dir / f"{split}.jsonl", read_lines(args.base_data_dir / f"{split}.jsonl"))

    print("Feedback curriculum built")
    print(f"curated_examples: {len(curated)}")
    print(f"repeat: {repeats}")
    print(f"train_lines: {len(train_lines)}")
    print(f"output_data_dir: {args.output_data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
