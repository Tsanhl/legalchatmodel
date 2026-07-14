#!/usr/bin/env python3
"""Build a separate, privacy-safe FTS index for marked work and feedback.

This index is deliberately separate from the legal-authority database. At
runtime it supplies assessment/writing guidance only; it must never be cited
as authority for a legal proposition. Scripts below 70 are indexed for audit
and diagnostic use, but their answer text is excluded from runtime retrieval.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "model_database" / "imported_feedback"
DB_PATH = ROOT / "model_database" / "feedback_index.sqlite3"

SUBJECT_KEYWORDS = {
    "biolaw_ai_data": ("artificial intelligence", "data protection", "privacy", "algorithm"),
    "business_law": ("director", "company", "shareholder", "insolvency"),
    "commercial_law": ("sale of goods", "nemo dat", "retention of title", "agency"),
    "competition_law": ("competition law", "article 101", "article 102", "dominance"),
    "criminal_law": ("criminal law", "insanity", "actus reus", "mens rea"),
    "environmental_law": ("environmental law", "pollution", "climate", "planning"),
    "eu_law": ("european union", "free movement", "tfeu", "directive 2004/38"),
    "land_law": ("land law", "co-ownership", "severance", "registered land"),
    "law_medicine": ("medical law", "mental capacity", "human tissue", "abortion"),
    "pensions_law": ("pensions law", "pension scheme", "normal retirement age", "trustee investment"),
    "private_international_law": ("private international law", "jurisdiction", "foreign judgment", "forum conveniens"),
    "trusts_law": ("trusts law", "three certainties", "beneficiary principle", "fiduciary"),
}


def subject_for(output: str, text: str) -> str:
    """Use an ignored local map when present; otherwise classify by legal content."""
    local_map_path = CORPUS / "subject_map.local.json"
    if local_map_path.is_file():
        local_map = json.loads(local_map_path.read_text(encoding="utf-8"))
        if output in local_map:
            return str(local_map[output])
    lowered = text.lower()
    scored = [
        (sum(lowered.count(term) for term in terms), subject)
        for subject, terms in SUBJECT_KEYWORDS.items()
    ]
    score, subject = max(scored)
    return subject if score else "general"


def final_mark(text: str) -> int | None:
    match = re.search(r"FINAL GRADE\s+(\d{1,3})\s*/\s*100", text, re.I)
    return int(match.group(1)) if match else None


def chunk_text(text: str, target: int = 1500, overlap: int = 180) -> list[str]:
    """Paragraph-aware chunks with small overlap; preserves the whole corpus."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > target * 2:
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        else:
            sentences = [paragraph]
        for piece in sentences:
            candidate = f"{current}\n\n{piece}".strip() if current else piece
            if current and len(candidate) > target:
                chunks.append(current)
                current = (current[-overlap:] + " " + piece).strip()
            else:
                current = candidate
    if current:
        chunks.append(current)
    return chunks


def split_sections(text: str) -> list[tuple[str, str]]:
    """Label marker feedback separately from the student's submitted answer."""
    match = re.search(r"\nFINAL GRADE\s*\n", text, re.I)
    if not match:
        return [("marker_guidance", text)]
    return [
        ("submitted_answer", text[: match.start()].strip()),
        ("marker_feedback", text[match.start():].strip()),
    ]


def main() -> None:
    manifest = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    if DB_PATH.exists():
        DB_PATH.unlink()
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """CREATE VIRTUAL TABLE feedback_chunks USING fts5(
               text,
               source UNINDEXED,
               source_kind UNINDEXED,
               quality_tier UNINDEXED,
               subject UNINDEXED,
               final_mark UNINDEXED,
               section_kind UNINDEXED,
               chunk_index UNINDEXED,
               total_chunks UNINDEXED,
               tokenize='unicode61 remove_diacritics 2'
           )"""
    )
    inserted = 0
    files = 0
    for item in manifest:
        output = item.get("output")
        if not output:
            continue
        path = CORPUS / output
        text = path.read_text(encoding="utf-8", errors="ignore")
        subject = subject_for(output, text)
        mark = final_mark(text)
        if item["kind"] == "marker_guidance":
            tier = "marker_guidance"
        elif mark is not None and mark >= 70:
            tier = "gold"
        else:
            tier = "diagnostic"
        sections = split_sections(text)
        section_chunks = [(kind, chunk) for kind, body in sections for chunk in chunk_text(body)]
        total = len(section_chunks)
        for index, (section_kind, chunk) in enumerate(section_chunks, 1):
            con.execute(
                "INSERT INTO feedback_chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    chunk,
                    item["source"],
                    item["kind"],
                    tier,
                    subject,
                    "" if mark is None else str(mark),
                    section_kind,
                    str(index),
                    str(total),
                ),
            )
            inserted += 1
        files += 1
    con.commit()
    con.execute("INSERT INTO feedback_chunks(feedback_chunks) VALUES('optimize')")
    con.commit()
    con.close()
    print(json.dumps({"database": str(DB_PATH), "files": files, "chunks": inserted}, indent=2))


if __name__ == "__main__":
    main()
