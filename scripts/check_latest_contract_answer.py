#!/usr/bin/env python3
"""Quality gate for the latest full MedData/SecureCloud site answer."""

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
                       AND q.content LIKE '%MedData Ltd%SecureCloud Ltd%'
              )
            ORDER BY m.id DESC LIMIT 1"""
    ).fetchone()
    con.close()
    if row is None:
        raise SystemExit("No completed MedData/SecureCloud answer found")
    answer = row["content"]
    body = re.split(r"(?im)^#{1,3}\s*References\s*$", answer, maxsplit=1)[0]
    body_words = len(body.split())
    required_issue_patterns = [
        r"misrepresentation", r"\bterms?\b", r"non[- ]reliance", r"exclusion",
        r"limitation", r"breach", r"causation", r"remoteness", r"penalt", r"remed",
    ]
    authority_patterns = [
        r"Misrepresentation Act 1967", r"(?:Unfair Contract Terms Act|UCTA)\s*1977",
        r"Cavendish", r"Hadley\s+v\s+Baxendale", r"First Tower",
    ]
    checks = {
        "body_1980_to_2020_words": 1980 <= body_words <= 2020,
        "all_requested_issues_present": all(
            re.search(pattern, body, re.I) for pattern in required_issue_patterns
        ),
        "both_parties_advised": "MedData" in body and "SecureCloud" in body,
        "core_authorities_present": sum(bool(re.search(pattern, body, re.I)) for pattern in authority_patterns) >= 4,
        "structured_headings": len(re.findall(r"(?m)^###\s+", body)) >= 5,
        "single_references_section": len(re.findall(r"(?im)^#{1,3}\s*References\s*$", answer)) == 1,
        "no_pipeline_leak": not re.search(
            r"apply these quality gates|source ledger|draft answer|end of answer|part \d+ of \d+",
            answer, re.I,
        ),
        "conclusion_present": bool(re.search(r"(?im)^###\s+.*(?:conclusion|overall advice|overall outcome)", body)),
        "no_invented_clause_or_entire_agreement": not re.search(
            r"\bclause\s+\d+\b|entire agreement clause", body, re.I
        ),
        "actual_reliance_not_erased": not re.search(
            r"(?:even|although|whether)\s+(?:MedData\s+)?(?:it\s+)?did not rely|"
            r"did not rely.{0,100}(?:actionable|claim|damages|induced)|"
            r"without (?:actual )?reliance.{0,90}(?:actionable|claim|damages)", body, re.I | re.S
        ),
        "section_2_1_not_called_innocent": not re.search(
            r"section\s+2\s*\(1\).{0,100}innocent misrepresentation|"
            r"innocent misrepresentation.{0,100}section\s+2\s*\(1\)", body, re.I | re.S
        ),
        "no_known_wrong_court_claims": not re.search(
            r"(?:First Tower|Watford Electronics|Transocean Drilling).{0,90}Supreme Court|"
            r"Supreme Court.{0,90}(?:First Tower|Watford Electronics|Transocean Drilling)",
            body, re.I | re.S,
        ),
        "cap_assigned_to_securecloud": not re.search(
            r"(?:MedData(?:'s|’s)? liability|limits? MedData(?:'s|’s)? liability).{0,80}£?5,?000|"
            r"£?5,?000.{0,80}(?:MedData(?:'s|’s)? liability|limits? MedData)", body, re.I | re.S
        ),
        "modern_penalty_test_present": bool(
            re.search(r"secondary obligation", body, re.I)
            and re.search(r"legitimate interest", body, re.I)
            and re.search(r"out of all proportion|proportionate|proportionality", body, re.I)
        ),
        "invoice_and_setoff_addressed": bool(
            re.search(r"principal invoice|invoice debt|invoice remains|claim the invoice", body, re.I)
            and re.search(r"set[- ]?off|counterclaim", body, re.I)
        ),
    }
    report = {
        "passed": all(checks.values()),
        "conversation_id": row["conversation_id"],
        "created_at": row["created_at"],
        "body_words": body_words,
        "total_words_with_references": len(answer.split()),
        "checks": checks,
    }
    (ROOT / "training" / "CONTRACT_SITE_FULL_ANSWER_GATE.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
