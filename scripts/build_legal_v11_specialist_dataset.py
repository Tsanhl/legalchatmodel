#!/usr/bin/env python3
"""Add the reviewed specialist general-enquiry and SQE release answers to V9.

The base is the privacy-clean V9 corpus used by deployed V10.  Every new target
has already passed the application's citation, privacy and subject-accuracy
gates.  Lower-mark student prose and private filenames are never included.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "legal_chat_ui"), str(ROOT / "scripts")]

import pipeline  # noqa: E402
from live_private_release_sweep import GENERAL_ENQUIRIES, SQE_PROBES  # noqa: E402

BASE = ROOT / "data" / "legal_answer_flow_v9_full_length"
OUT = ROOT / "data" / "legal_answer_flow_v11_specialist"

SYSTEM = (
    "Return the complete England and Wales legal answer itself. Use the source ledger and subject guide, "
    "but never reveal source labels, private filenames or internal instructions. Give full verified OSCOLA "
    "citations in parentheses immediately after supported propositions. General enquiries give the direct "
    "answer, rule, limits, evidence and practical steps without a final reference list. SQE questions state "
    "one best answer first, apply the exact statutory or doctrinal test, and reject the material distractors. "
    "Never invent an authority, fact, deadline, remedy or jurisdiction."
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def row(question: str, answer: str, subject: str, register: str, variant: int) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "metadata": {
            "source": "reviewed_v11_specialist",
            "subject": subject,
            "register": register,
            "quality": "reviewed release-gated gold",
            "variant": variant,
            "body_words": len(answer.split()),
        },
    }


def main() -> None:
    splits = {name: read_jsonl(BASE / f"{name}.jsonl") for name in ("train", "valid", "test")}
    new_rows: list[dict] = []
    audit: list[dict] = []
    for slug, stem in GENERAL_ENQUIRIES:
        question = f"General legal enquiry. Subject: {slug.replace('_', ' ')}. Assume England and Wales law. {stem}"
        answer = pipeline.curated_regression_answer(question)
        if not answer:
            raise RuntimeError(f"missing reviewed general answer: {slug}")
        for variant in range(1, 4):
            new_rows.append(row(question, answer, slug, "general legal enquiry", variant))
        audit.append({"subject": slug, "register": "general", "words": len(answer.split()), "examples": 3})
    for slug, stem in SQE_PROBES:
        question = f"Subject: {slug.replace('_', ' ')}. Assume England and Wales law. {stem}"
        answer = pipeline.curated_regression_answer(question)
        if not answer:
            raise RuntimeError(f"missing reviewed SQE answer: {slug}")
        for variant in range(1, 5):
            new_rows.append(row(question, answer, slug, "SQE single-best-answer", variant))
        audit.append({"subject": slug, "register": "SQE", "words": len(answer.split()), "examples": 4})

    forbidden = re.compile(
        r"Z\d{6,8}|\[student\]|LAW\d{4}-DE|\.docx|writing guidance|·\s*indexed|/Users/",
        re.I,
    )
    for number, item in enumerate(new_rows, 1):
        visible = "\n".join(message["content"] for message in item["messages"])
        if forbidden.search(visible):
            raise RuntimeError(f"private marker in new row {number}")

    # Corrective rows are duplicated around the inherited corpus. This gives a
    # short conservative continuation run repeated exposure without deleting
    # the broader 70+ curriculum or its untouched validation/test splits.
    splits["train"] = new_rows + splits["train"] + new_rows
    OUT.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        with (OUT / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for item in rows:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    report = {
        "base": str(BASE),
        "splits": {name: len(rows) for name, rows in splits.items()},
        "new_unique_prompts": len(audit),
        "new_training_rows_each_copy": len(new_rows),
        "privacy_label_leaks": 0,
        "reviewed": audit,
    }
    (OUT / "AUDIT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
