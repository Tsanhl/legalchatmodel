#!/usr/bin/env python3
"""Summarise legal-subject and answer-type coverage in an MLX chat dataset."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


TOPICS: dict[str, tuple[str, ...]] = {
    "contract": ("contract", "consideration", "offer", "acceptance", "misrepresentation"),
    "tort": ("tort", "negligence", "occupiers", "nuisance", "defamation"),
    "criminal": ("criminal", "murder", "manslaughter", "mens rea", "actus reus"),
    "equity_trusts": ("trust", "equity", "fiduciary", "beneficial interest"),
    "land": ("land law", "lease", "licence", "property", "easement", "mortgage"),
    "public_jr": ("public law", "judicial review", "legitimate expectation", "wednesbury"),
    "company": ("company law", "director", "companies act", "shareholder"),
    "evidence": ("evidence", "hearsay", "confession", "burden of proof"),
    "family": ("family law", "divorce", "financial remedy", "children act"),
    "employment": ("employment", "unfair dismissal", "redundancy", "discrimination"),
    "intellectual_property": ("intellectual property", "copyright", "patent", "trade mark"),
    "tax": ("tax law", "taxation", "income tax", "vat"),
    "private_international": ("private international", "rome i", "rome ii", "jurisdiction"),
    "public_international": ("public international", "treaty", "state responsibility"),
    "human_rights": ("human rights", "echr", "article 8", "article 10"),
    "competition": ("competition law", "cartel", "article 101", "article 102"),
    "media_privacy": ("privacy", "misuse of private information", "media law"),
    "medical_biolaw": ("medical law", "biolaw", "consent", "clinical negligence"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit legal topic coverage in chat JSONL data.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/legal_answer_flow_feedback_v2"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training/legal_answer_flow_feedback_v2/coverage_report.json"),
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def message_text(row: dict[str, Any], role: str) -> str:
    messages = row.get("messages") or []
    return "\n".join(
        str(message.get("content") or "")
        for message in messages
        if isinstance(message, dict) and message.get("role") == role
    ).lower()


def classify_topic(text: str) -> str:
    for topic, markers in TOPICS.items():
        if any(marker in text for marker in markers):
            return topic
    return "unclassified"


def classify_answer_type(text: str) -> str:
    if re.search(r"\b(problem question|advise\b|facts:)\b", text):
        return "problem"
    if re.search(r"\b(essay|critically discuss|critically evaluate|discuss)\b", text):
        return "essay"
    if re.search(r"\b(explain|outline|summari[sz]e)\b", text):
        return "explanatory"
    return "other"


def main() -> int:
    args = parse_args()
    topic_counts: Counter[str] = Counter()
    topic_word_counts: dict[str, list[int]] = {}
    type_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    output_words: list[int] = []

    for split in ("train", "valid", "test"):
        rows = read_rows(args.data_dir / f"{split}.jsonl")
        split_counts[split] = len(rows)
        for row in rows:
            prompt = message_text(row, "user")
            answer = message_text(row, "assistant")
            topic = classify_topic(prompt)
            topic_counts[topic] += 1
            type_counts[classify_answer_type(prompt)] += 1
            words = len(re.findall(r"\b[\w'-]+\b", answer))
            output_words.append(words)
            topic_word_counts.setdefault(topic, []).append(words)

    report = {
        "data_dir": str(args.data_dir),
        "split_counts": dict(split_counts),
        "topic_counts": dict(sorted(topic_counts.items())),
        "topic_output_word_count": {
            topic: {
                "count": len(counts),
                "median": sorted(counts)[len(counts) // 2],
                "at_least_700": sum(count >= 700 for count in counts),
                "at_least_1000": sum(count >= 1000 for count in counts),
            }
            for topic, counts in sorted(topic_word_counts.items())
        },
        "answer_type_counts": dict(sorted(type_counts.items())),
        "output_word_count": {
            "count": len(output_words),
            "median": sorted(output_words)[len(output_words) // 2] if output_words else 0,
            "at_least_700": sum(count >= 700 for count in output_words),
            "at_least_1000": sum(count >= 1000 for count in output_words),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
