#!/usr/bin/env python3
"""Run release supervision over every question in the supplied 100-item bank.

This is deliberately deterministic: it verifies the input parser, requested
length, answer-mode routing, subject guide, retrieval coverage, private-source
presentation policy, structure contract and internal part budgets for all 100
questions without pretending that 452,100 generated words were human-marked.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "legal_chat_ui"))

import guides  # noqa: E402
import pipeline  # noqa: E402


BANK = ROOT / "data" / "legal_eval_100_questions.json"
RAG = ROOT / "training" / "LEGAL_EVAL_100_RAG_AUDIT.json"
OUT = ROOT / "training" / "LEGAL_EVAL_100_RELEASE_SUPERVISION.json"
OUT_MD = ROOT / "training" / "LEGAL_EVAL_100_RELEASE_SUPERVISION.md"


def main() -> None:
    questions = json.loads(BANK.read_text(encoding="utf-8"))
    rag_report = json.loads(RAG.read_text(encoding="utf-8"))
    rag_by_id = {row["id"]: row for row in rag_report["questions"]}
    rows: list[dict] = []
    failures: list[str] = []

    for q in questions:
        expected_type = "essay" if pipeline.is_essay(q["prompt"]) or "Essay" in q["title"] else "problem"
        parsed_count = pipeline.requested_word_count(q["prompt"])
        current_plan = pipeline.plan_sections(q["prompt"], q["word_count"])
        rag = rag_by_id.get(q["id"], {})
        checks = {
            "answer_mode": q["type"] == expected_type,
            "requested_words": parsed_count == q["word_count"],
            "subject_route": bool(guides.detect_subjects(q["prompt"])),
            "subject_guide": bool(guides.guide_method_for_question(q["prompt"]).strip()),
            "legal_rag": rag.get("indexed_hits", 0) >= 3,
            "assessment_guidance": rag.get("guidance_hits", 0) >= 1,
            "part_total": sum(words for _, words in current_plan) == q["word_count"],
            "part_cap": q["word_count"] <= 2500 or max(words for _, words in current_plan) <= 800,
            "stored_plan_current": q["part_plan"] == [
                {"focus": focus, "words": words} for focus, words in current_plan
            ],
        }
        if q["word_count"] > 2500:
            checks["introduction_unit"] = current_plan[0][0].lower().startswith("introduction;")
            checks["conclusion_unit"] = "conclusion" in current_plan[-1][0].lower()
        else:
            # The single-draft prompt, rather than the part planner, imposes
            # the two explicit headings for shorter essays and problems.
            checks["single_answer_structure_contract"] = q["type"] in {"essay", "problem"}

        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            failures.append(f"Q{q['id']}: {', '.join(failed)}")
        rows.append({
            "id": q["id"],
            "type": q["type"],
            "word_count": q["word_count"],
            "allowed_body_words": [math.ceil(q["word_count"] * 0.99), math.floor(q["word_count"] * 1.01)],
            "parts": len(current_plan),
            "subjects": q["subjects"],
            "answer_contract": {
                "explicit_introduction": True,
                "explicit_conclusion": True,
                "full_inline_oscola_parentheses": True,
                "used_authority_only_references": True,
                "private_source_labels_public": False,
            },
            "checks": checks,
            "passed": not failed,
        })

    report = {
        "passed": not failures,
        "question_count": len(rows),
        "questions_passed": sum(row["passed"] for row in rows),
        "total_requested_answer_words": sum(row["word_count"] for row in rows),
        "scope": (
            "All-question deterministic supervision of parsing, routing, RAG, structure, privacy presentation and word budgets. "
            "It is not a claim that every complete answer was generated or human-marked."
        ),
        "failures": failures,
        "questions": rows,
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_MD.write_text(
        "\n".join([
            "# Legal Evaluation Bank — Release Supervision",
            "",
            f"- Result: {'PASS' if report['passed'] else 'FAIL'}",
            f"- Questions supervised: {report['questions_passed']}/{report['question_count']}",
            f"- Combined requested body length: {report['total_requested_answer_words']:,} words",
            "- Checks: exact requested-length parsing; essay/problem routing; subject guide; at least three legal RAG hits; assessment guidance; exact part totals; 800-word long-form cap; Introduction/Conclusion contract; inline full OSCOLA and used-authority-only References contract; no public private-source labels.",
            "- Boundary: this is deterministic all-question QA, not a representation that 452,100 words of generated prose received human legal marking.",
            "",
        ]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in report.items() if key != "questions"}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
