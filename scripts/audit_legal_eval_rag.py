#!/usr/bin/env python3
"""Run the complete 100-question bank through offline RAG/guide assembly."""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "legal_chat_ui"))

import guides  # noqa: E402
import retrieval  # noqa: E402


def main() -> None:
    questions = json.loads((ROOT / "data" / "legal_eval_100_questions.json").read_text())
    rows: list[dict] = []
    failures: list[str] = []
    for question in questions:
        hits = retrieval.search(question["prompt"], k=6, subjects=question["subjects"])
        guidance = retrieval.search_feedback_guidance(
            question["prompt"], k=5, subjects=question["subjects"]
        )
        method = guides.guide_method_for_question(
            question["prompt"], guides.detect_subject(question["prompt"])
        )
        missing_guides = [
            subject for subject in question["subjects"]
            if not (guides.GUIDES_DIR / f"{subject}.md").exists()
        ]
        if len(hits) < 3:
            failures.append(f"Q{question['id']}: only {len(hits)} indexed legal hits")
        if not method:
            failures.append(f"Q{question['id']}: no assembled answer guide")
        if missing_guides:
            failures.append(f"Q{question['id']}: missing guides {missing_guides}")
        rows.append({
            "id": question["id"],
            "title": question["title"],
            "subjects": question["subjects"],
            "indexed_hits": len(hits),
            "guidance_hits": len(guidance),
            "top_sources": [hit["document_name"] for hit in hits[:3]],
            "top_categories": [hit["category"] for hit in hits[:3]],
            "guide_chars": len(method),
        })
    category_counts = Counter(category for row in rows for category in row["top_categories"] if category)
    report = {
        "question_count": len(rows),
        "passed": not failures,
        "questions_with_at_least_3_legal_hits": sum(row["indexed_hits"] >= 3 for row in rows),
        "questions_with_assessment_guidance": sum(row["guidance_hits"] > 0 for row in rows),
        "median_guide_chars": int(statistics.median(row["guide_chars"] for row in rows)),
        "top_retrieved_categories": dict(category_counts.most_common(20)),
        "failures": failures,
        "questions": rows,
    }
    out = ROOT / "training" / "LEGAL_EVAL_100_RAG_AUDIT.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {key: value for key, value in report.items() if key != "questions"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
