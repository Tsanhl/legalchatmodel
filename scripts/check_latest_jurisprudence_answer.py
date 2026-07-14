#!/usr/bin/env python3
"""Quality gate for the latest full multi-theory jurisprudence site answer."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "legal_chat_ui" / "chat.sqlite3"


def main() -> None:
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    row = con.execute(
        """SELECT m.content, m.created_at, m.conversation_id
             FROM messages m
             JOIN conversations c ON c.id = m.conversation_id
            WHERE m.role='assistant' AND c.deleted_at IS NULL
              AND EXISTS (
                    SELECT 1 FROM messages q
                     WHERE q.conversation_id=m.conversation_id AND q.role='user'
                       AND lower(q.content) LIKE '%nature of law cannot be explained by one theory alone%'
                       AND lower(q.content) LIKE '%postcolonial theory%'
              )
            ORDER BY m.id DESC LIMIT 1"""
    ).fetchone()
    con.close()
    if row is None:
        raise SystemExit("No completed multi-theory jurisprudence answer found")

    answer = row["content"]
    body = re.split(r"(?im)^#{1,3}\s*References\s*$", answer, maxsplit=1)[0]
    body_words = len(body.split())
    theories = (
        "positivism", "natural law", "interpretivism", "realism", "feminis",
        "critical race", "marxis", "postcolonial",
    )
    thinkers = (
        "Hart", "Fuller", "Dworkin", "Holmes", "MacKinnon", "Crenshaw", "Pashukanis", "Anghie",
    )
    sections = re.findall(
        r"(?ms)^###\s+([^\n]+)\n+(.*?)(?=^###\s+|^---\s*$|\Z)",
        body,
    )
    theory_section_words = {
        title: len(text.split())
        for title, text in sections
        if any(term in title.lower() for term in theories)
    }
    checks = {
        "body_1980_to_2020_words": 1980 <= body_words <= 2020,
        "all_eight_theories_present": all(term in body.lower() for term in theories),
        "named_theorists_across_traditions": all(name.lower() in body.lower() for name in thinkers),
        "critical_ranked_thesis": bool(re.search(r"disciplined pluralism|ranked pluralism", body, re.I)),
        "theory_sections_developed": len(theory_section_words) >= 8
        and all(words >= 100 for words in theory_section_words.values()),
        "single_references_section": len(
            re.findall(r"(?im)^#{1,3}\s*References\s*$", answer)
        ) == 1,
        "oscola_bibliography_breadth": len(re.findall(r"(?m)^- ", answer)) >= 15,
        "no_pipeline_or_part_placeholder": not re.search(
            r"source ledger|draft answer|part \d+[/ ](?:of )?\d+|\(end of answer\)",
            answer,
            re.I,
        ),
        "no_caricature_shortcuts": not re.search(
            r"positivism (?:simply )?says law is whatever the state commands|"
            r"natural law (?:simply )?says unjust law is not law|"
            r"realism says rules never matter",
            body,
            re.I,
        ),
        "conclusion_present": bool(re.search(r"(?im)^###\s+Conclusion", body)),
    }
    report = {
        "passed": all(checks.values()),
        "conversation_id": row["conversation_id"],
        "created_at": row["created_at"],
        "body_words": body_words,
        "total_words_with_references": len(answer.split()),
        "theory_section_words": theory_section_words,
        "checks": checks,
    }
    out = ROOT / "training" / "JURISPRUDENCE_SITE_FULL_ANSWER_GATE.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
