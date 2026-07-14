#!/usr/bin/env python3
"""Parse the user's 100-question law bank and run deterministic routing QA."""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "legal_chat_ui"))

import guides  # noqa: E402
import pipeline  # noqa: E402


SOURCE = Path(
    os.environ.get("LEGAL_QUESTION_BANK", "legal_question_bank.txt")
)
OUT = ROOT / "data" / "legal_eval_100_questions.json"
REPORT_JSON = ROOT / "training" / "legal_eval_100_routing_report.json"
REPORT_MD = ROOT / "training" / "LEGAL_EVAL_100_ROUTING_REPORT.md"


def parse(text: str) -> list[dict]:
    headings = list(re.finditer(r"^##\s+(\d{1,3})\.\s+(.+?)\s*$", text, re.M))
    questions: list[dict] = []
    for index, match in enumerate(headings):
        number = int(match.group(1))
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        body = text[match.end():end].strip()
        # The bank's horizontal rule ends the question.  Section headings such
        # as "Around 2,000-3,500 words each" belong to the next group and must
        # not override the question's own Suggested length.
        body = re.split(r"(?m)^---\s*$", body, maxsplit=1)[0].strip()
        title = match.group(2).strip()
        prompt = f"{title}\n\n{body}".strip()
        count = pipeline.requested_word_count(prompt)
        qtype = "essay" if pipeline.is_essay(prompt) or "Essay" in title else "problem"
        questions.append(
            {
                "id": number,
                "title": title,
                "type": qtype,
                "word_count": count,
                "subjects": guides.detect_subjects(prompt),
                "prompt": prompt,
            }
        )
    return questions


def main() -> None:
    questions = parse(SOURCE.read_text(encoding="utf-8"))
    ids = [q["id"] for q in questions]
    failures: list[str] = []
    if ids != list(range(1, 101)):
        failures.append(f"Expected IDs 1-100, got {ids}")
    for q in questions:
        if q["word_count"] is None:
            failures.append(f"Q{q['id']}: missing word count")
            continue
        sections = pipeline.plan_sections(q["prompt"], q["word_count"])
        if sum(words for _, words in sections) != q["word_count"]:
            failures.append(f"Q{q['id']}: part budgets do not total requested length")
        if q["word_count"] > 2500 and max(words for _, words in sections) > 800:
            failures.append(f"Q{q['id']}: a long-answer part exceeds 800 words")
        if not q["subjects"]:
            failures.append(f"Q{q['id']}: no subject route")
        q["part_plan"] = [{"focus": title, "words": words} for title, words in sections]
    counts = Counter(subject for q in questions for subject in q["subjects"])
    report = {
        "question_count": len(questions),
        "essay_count": sum(q["type"] == "essay" for q in questions),
        "problem_count": sum(q["type"] == "problem" for q in questions),
        "min_words": min((q["word_count"] or 0) for q in questions),
        "max_words": max((q["word_count"] or 0) for q in questions),
        "subject_routes": dict(counts.most_common()),
        "failures": failures,
        "passed": not failures,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(questions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Legal Evaluation Bank — Routing QA",
        "",
        f"- Questions parsed: {report['question_count']}",
        f"- Essays/problems: {report['essay_count']}/{report['problem_count']}",
        f"- Requested range: {report['min_words']:,}–{report['max_words']:,} words",
        f"- Result: {'PASS' if report['passed'] else 'FAIL'}",
        "",
        "## Subject routes",
        "",
    ]
    lines += [f"- {subject}: {count}" for subject, count in counts.most_common()]
    if failures:
        lines += ["", "## Failures", ""] + [f"- {failure}" for failure in failures]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
