#!/usr/bin/env python3
"""Build the V8 corrective set for complete, count-safe legal answers.

The base of the set is the clean 70+ V7 corpus.  The additional targets are
reviewed regression answers which demonstrate explicit introductions and
conclusions, proposition-level full OSCOLA citations, and a final References
section.  Private filenames and candidate identifiers are never copied into
the public-facing text or metadata.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V7 = ROOT / "data" / "legal_answer_flow_v7_marked_gold"
GOLD = ROOT / "training" / "gold_answers"
OUT = ROOT / "data" / "legal_answer_flow_v8_complete_answer"

SYSTEM = (
    "Write the complete answer itself for an England and Wales law question. "
    "An essay and a problem question must open with an explicit Introduction and end with an explicit Conclusion. "
    "Use issue-led analysis, exact legal tests, immediate application, counterarguments, remedies and calibrated advice. "
    "After each proposition that depends on authority, give the verified full OSCOLA citation in parentheses; never invent "
    "a case, citation or pinpoint. End with one OSCOLA References section containing only authorities actually used. "
    "Do not output a plan, part label, word-count commentary, source-ledger label or private filename."
)

GOLD_SOURCES = (
    ("meddata_securecloud_contract_problem.md", "contract law", "problem question"),
    ("nature_of_law_multi_theory_essay.md", "jurisprudence", "critical essay"),
    ("cross_subject_legal_values_essay.md", "cross-subject law", "critical essay"),
    ("dana_eli_farah_tort_problem.md", "tort law", "problem question"),
    ("proprietary_estoppel_practical_enquiry.md", "land law", "general legal enquiry"),
)


def words(text: str) -> int:
    return len(re.findall(r"\b[\w’'-]+\b", text))


def split_long_paragraph(paragraph: str, limit: int) -> list[str]:
    if words(paragraph) <= limit:
        return [paragraph]
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z*_(])", paragraph)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        proposed = f"{current} {sentence}".strip()
        if current and words(proposed) > limit:
            chunks.append(current)
            current = sentence
        else:
            current = proposed
    if current:
        chunks.append(current)
    return chunks


def balanced_sections(text: str, target: int = 620, limit: int = 800) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        units.extend(split_long_paragraph(paragraph, limit))

    chunks: list[str] = []
    current: list[str] = []
    for unit in units:
        proposed = "\n\n".join([*current, unit])
        if current and words(proposed) > limit and words("\n\n".join(current)) >= target // 2:
            chunks.append("\n\n".join(current))
            current = [unit]
        else:
            current.append(unit)
    if current:
        tail = "\n\n".join(current)
        if chunks and words(tail) < 260 and words(chunks[-1]) + words(tail) <= limit:
            chunks[-1] = f"{chunks[-1]}\n\n{tail}"
        else:
            chunks.append(tail)
    if len(chunks) > 1 and words(chunks[-1]) < 260:
        # References can otherwise become a tiny standalone training target.
        # Rebalance the last two units at a paragraph boundary while keeping a
        # heading attached to the content that follows it.
        paragraphs = [p for p in re.split(r"\n\s*\n", f"{chunks[-2]}\n\n{chunks[-1]}") if p.strip()]
        total = words("\n\n".join(paragraphs))
        candidates: list[tuple[int, int]] = []
        for boundary in range(1, len(paragraphs)):
            if re.match(r"^#{2,4}\s+", paragraphs[boundary - 1].strip()):
                continue
            left_words = words("\n\n".join(paragraphs[:boundary]))
            right_words = total - left_words
            if left_words >= 260 and right_words >= 260:
                candidates.append((abs(left_words - right_words), boundary))
        if candidates:
            boundary = min(candidates)[1]
            chunks[-2] = "\n\n".join(paragraphs[:boundary])
            chunks[-1] = "\n\n".join(paragraphs[boundary:])
    return chunks


def read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for number, row in enumerate(rows, 1):
        for message in row["messages"]:
            content = message["content"]
            content = re.sub(r"[\[(]*\[student\][\])\]]*", "", content, flags=re.I)
            content = re.sub(r"\bZ?0?192910\b", "", content, flags=re.I)
            message["content"] = re.sub(r"\n{3,}", "\n\n", content).strip()
        # Metadata is not model-visible, but neutralise it as an additional
        # defence against accidental display by a later reporting tool.
        row["metadata"]["source"] = f"legacy_70plus_{path.stem}_{number}"
    return rows


def gold_examples(filename: str, subject: str, register: str, source_number: int) -> list[dict]:
    text = (GOLD / filename).read_text(encoding="utf-8").strip()
    chunks = balanced_sections(text)
    rows: list[dict] = []
    for index, target in enumerate(chunks):
        if index == 0:
            role = "opening unit: include the Introduction and begin substantive analysis"
        elif index == len(chunks) - 1:
            role = "final unit: complete the analysis, include the Conclusion, then the used-authority-only References"
        else:
            role = "middle analytical unit: continue without repetition or meta-commentary"
        prior = "" if index == 0 else " ".join(chunks[index - 1].split()[-120:])
        for variant in range(2):
            prompt = (
                f"Write the {role} of this {register} in {subject}. Return submission-ready answer prose only."
                + (f"\n\nPRIOR ENDING:\n{prior}" if prior else "")
            )
            rows.append({
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": target},
                ],
                "metadata": {
                    "source": f"reviewed_regression_{source_number}",
                    "subject": subject,
                    "register": register,
                    "quality": "reviewed complete-answer gold",
                    "part": index + 1,
                    "parts": len(chunks),
                    "variant": variant + 1,
                },
            })
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    splits = {name: read_jsonl(V7 / f"{name}.jsonl") for name in ("train", "valid", "test")}
    added: list[dict] = []
    for number, (filename, subject, register) in enumerate(GOLD_SOURCES, 1):
        rows = gold_examples(filename, subject, register, number)
        splits["train"].extend(rows)
        added.append({
            "source": f"reviewed_regression_{number}",
            "subject": subject,
            "register": register,
            "examples": len(rows),
            "parts": rows[0]["metadata"]["parts"],
            "target_words": [words(rows[i]["messages"][-1]["content"]) for i in range(0, len(rows), 2)],
        })

    for split, rows in splits.items():
        with (OUT / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    public_forbidden = re.compile(
        r"Z\d{6,8}|\[student\]|LAW\d{4}-DE|\.docx|writing guidance|\u00b7 indexed",
        re.I,
    )
    leaks = []
    for split, rows in splits.items():
        for row_number, row in enumerate(rows, 1):
            public_text = "\n".join(message["content"] for message in row["messages"])
            if public_forbidden.search(public_text):
                leaks.append({"split": split, "row": row_number})
    if leaks:
        raise RuntimeError(f"Private/public label leakage in corrective dataset: {leaks[:5]}")

    report = {
        "policy": "70+ V7 prose plus reviewed complete-answer regressions; no lower-mark prose, chat data or private filenames.",
        "splits": {split: len(rows) for split, rows in splits.items()},
        "added": added,
        "privacy_label_leaks": len(leaks),
    }
    (OUT / "AUDIT.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
