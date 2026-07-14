#!/usr/bin/env python3
"""Add the reviewed consideration-reform essay to the privacy-clean V11 set."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "legal_answer_flow_v11_specialist"
OUT = ROOT / "data" / "legal_answer_flow_v12_consideration"
GOLD = ROOT / "training" / "gold_answers" / "consideration_reform_essay_1000.md"

QUESTION = (
    "Suggested length: 1,000 words. Assume England and Wales law. "
    "The doctrine of consideration is an outdated technical requirement that English contract law "
    "should abandon. Critically discuss with reference to Williams v Roffey, Foakes v Beer and "
    "promissory estoppel."
)
SYSTEM = (
    "Write the complete first-class legal essay, not a plan. Meet the requested body word count within "
    "+/-1%, use an explicit Introduction and Conclusion, critically resolve the proposition, and give "
    "full verified OSCOLA citations in parentheses immediately after relevant propositions. Never invent "
    "an authority and never reveal private material or internal labels."
)


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    answer = GOLD.read_text(encoding="utf-8").strip()
    if not 990 <= len(answer.split()) <= 1010:
        raise RuntimeError(f"gold body outside +/-1%: {len(answer.split())}")
    forbidden = re.compile(r"Z\d{6,8}|\[student\]|\.docx|/Users/|writing guidance", re.I)
    if forbidden.search(answer):
        raise RuntimeError("private marker in gold answer")
    splits = {name: read(BASE / f"{name}.jsonl") for name in ("train", "valid", "test")}
    corrective = []
    for variant in range(1, 7):
        corrective.append({
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": QUESTION},
                {"role": "assistant", "content": answer},
            ],
            "metadata": {
                "source": "reviewed_v12_consideration",
                "subject": "contract_law",
                "register": "critical essay",
                "quality": "reviewed release-gated gold",
                "variant": variant,
                "body_words": len(answer.split()),
            },
        })
    splits["train"] = corrective + splits["train"] + corrective
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "base": str(BASE),
        "splits": {name: len(rows) for name, rows in splits.items()},
        "corrective_rows_each_copy": len(corrective),
        "gold_body_words": len(answer.split()),
        "privacy_label_leaks": 0,
    }
    (OUT / "AUDIT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
