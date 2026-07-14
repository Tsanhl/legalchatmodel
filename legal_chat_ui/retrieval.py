"""RAG retrieval over the indexed legal database (no model, no extra deps).

The 5.8 GB Chroma store was built with a 384-dim hash embedding, but it also
keeps a SQLite FTS5 full-text table. For legal text, BM25 keyword search over
that table is fast, accurate, and needs nothing but the standard library — so
this module queries it directly, read-only, and formats a source ledger.
"""

from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
CHROMA_DB = Path(os.environ.get(
    "LEGAL_RAG_DB",
    APP_DIR.parent / "model_database" / "snapshot" / "chroma_db" / "chroma.sqlite3",
)).expanduser()
FEEDBACK_DB = Path(os.environ.get(
    "LEGAL_GUIDANCE_DB", APP_DIR.parent / "model_database" / "feedback_index.sqlite3"
)).expanduser()
PUBLIC_GUIDES_DIR = APP_DIR / "law_guides"
PUBLIC_GOLD_DIR = APP_DIR.parent / "training" / "gold_answers"

_STOP = {
    "the", "and", "for", "with", "from", "that", "this", "what", "which", "does",
    "under", "about", "explain", "essay", "problem", "question", "law", "legal",
    "answer", "advise", "critically", "evaluate", "how", "why", "when", "who",
    "are", "was", "were", "can", "could", "would", "should", "please", "give",
    "your", "you", "into", "onto", "a", "an", "of", "in", "on", "to", "is", "it",
}

_SUBJECT_SOURCE_HINTS: dict[str, tuple[str, ...]] = {
    "contract_law": ("contract", "commercial law"),
    "tort_law": ("tort", "negligence"),
    "law_medicine": ("law and medicine", "medical law", "clinical"),
    "biolaw_ai_data": ("ai and data", "data protection", "biolaw", "robotics and ai"),
    "privacy_media_law": ("privacy", "media", "freedom of expression", "data protection", "constitutional law"),
    "consumer_law": ("consumer", "commercial law", "competition law", "contract law"),
    "trusts_law": ("trust", "equity"),
    "land_law": ("land law", "real property", "property law"),
    "criminal_law": ("criminal law",),
    "criminal_procedure_law": ("criminal law", "criminal procedure"),
    "evidence_law": ("evidence law", "criminal law"),
    "competition_law": ("competition law",),
    "commercial_law": ("commercial law", "sale of goods", "contract law"),
    "business_law": ("business law", "company law", "corporate", "insolvency"),
    "insolvency_law": ("insolvency", "corporate"),
    "employment_law": ("employment law",),
    "family_law": ("family law",),
    "tax_law": ("tax law", "taxation", "revenue and customs"),
    "intellectual_property_law": ("intellectual property", "copyright", "patent", "trade mark"),
    "pensions_law": ("pensions law",),
    "environmental_law": ("environmental law", "planning law"),
    "succession_wills": ("wills", "succession", "probate"),
    "public_law": ("constitutional law", "public law", "administrative law", "judicial review"),
    "human_rights_law": ("human rights", "constitutional law", "freedom of expression"),
    "eu_law": ("eu law", "european union", "brexit"),
    "private_international_law": ("private international law", "conflict of laws"),
    "public_international_law": ("public international law",),
    "mediation_law": ("mediation", "arbitration", "civil procedure"),
    "restitution_law": ("restitution", "unjust enrichment", "trust"),
    "remedies_law": ("remedies", "restitution", "civil procedure", "contract law", "tort law", "trust"),
    "equality_law": ("equality law", "discrimination"),
    "immigration_refugee_law": ("immigration", "refugee", "asylum"),
    "housing_law": ("housing law", "land law"),
    "jurisprudence_law": ("jurisprudence", "legal theory"),
    "civil_procedure_law": ("civil procedure",),
    "sentencing_law": ("sentencing", "criminal law"),
    "financial_regulation_law": ("financial regulation", "banking law"),
}


def _fts_query(text: str, max_terms: int = 12) -> str | None:
    raw = (text or "").lower()
    # Exam problems commonly put the decisive doctrines in a final "Consider:"
    # list. Search those first instead of wasting the term cap on party names and
    # scenario nouns from the opening facts.
    focus_match = re.search(
        r"\bconsider(?:\s*,?\s*where relevant)?\s*:?\s*(.{20,})$", raw, re.I | re.S
    )
    focus = focus_match.group(1) if focus_match else ""
    toks = re.findall(r"[A-Za-z][A-Za-z0-9'\-]{1,}", f"{focus} {raw}")
    toks = [t for t in toks if t not in _STOP]
    # de-dup, keep order, cap
    seen, keep = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t); keep.append(t)
        if len(keep) >= max_terms:
            break
    if not keep:
        return None
    # quote each term so FTS treats it as a literal (avoids operator parsing)
    return " OR ".join(f'"{t}"' for t in keep)


def _meta(cur: sqlite3.Cursor, rowid: int) -> dict:
    out = {}
    for key, val in cur.execute(
        "SELECT key, string_value FROM embedding_metadata WHERE id = ?", (rowid,)
    ):
        if key in ("document_name", "category", "subcategory", "chunk_index",
                   "total_chunks", "document_type", "file_path"):
            out[key] = val
    return out


def _query_terms(text: str, max_terms: int = 18) -> list[str]:
    """Return stable lexical terms for the bundled database-free fallback."""
    terms = re.findall(r"[a-z][a-z0-9'\-]{2,}", (text or "").lower())
    out: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if term in _STOP or term in seen:
            continue
        seen.add(term)
        out.append(term)
        if len(out) >= max_terms:
            break
    return out


def _public_sections(path: Path) -> list[tuple[str, str]]:
    """Split a public Markdown guide/answer into bounded retrieval sections."""
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    title_match = re.search(r"(?m)^#\s+(.+?)\s*$", raw)
    title = title_match.group(1).strip() if title_match else path.stem.replace("_", " ").title()
    pieces = re.split(r"(?m)(?=^#{2,4}\s+)", raw)
    sections: list[tuple[str, str]] = []
    for piece in pieces:
        text = re.sub(r"\n{3,}", "\n\n", piece).strip()
        if len(text.split()) < 12:
            continue
        # Keep prompt ledgers compact while preserving complete propositions.
        sections.append((title, text[:3600]))
    return sections


def _search_public_knowledge(query: str, k: int,
                             subjects: list[str] | None = None) -> list[dict]:
    """Search bundled anonymized guides/gold answers when no private DB exists."""
    terms = _query_terms(query)
    if not terms:
        return []
    subject_set = {subject for subject in (subjects or []) if subject}
    candidates: list[dict] = []
    paths = sorted(PUBLIC_GUIDES_DIR.glob("*.md")) if PUBLIC_GUIDES_DIR.is_dir() else []
    if PUBLIC_GOLD_DIR.is_dir():
        paths += sorted(PUBLIC_GOLD_DIR.glob("*.md"))
    for path in paths:
        slug = path.stem
        for section_index, (title, text) in enumerate(_public_sections(path), 1):
            blob = f"{title} {text}".lower()
            lexical = sum(min(blob.count(term), 4) for term in terms)
            subject_bonus = 10 if slug in subject_set else 0
            if not lexical and not subject_bonus:
                continue
            candidates.append({
                "text": text,
                "score": -(lexical + subject_bonus),
                "document_name": f"Bundled legal guide — {title}",
                "category": slug,
                "subcategory": "anonymized public fallback",
                "chunk_index": str(section_index),
                "total_chunks": "",
                "_rank": lexical + subject_bonus,
            })
    candidates.sort(key=lambda item: (-item["_rank"], item["document_name"], item["chunk_index"]))
    results: list[dict] = []
    per_document: dict[str, int] = {}
    for candidate in candidates:
        name = candidate["document_name"]
        if per_document.get(name, 0) >= 2:
            continue
        candidate.pop("_rank", None)
        results.append(candidate)
        per_document[name] = per_document.get(name, 0) + 1
        if len(results) >= k:
            break
    return results


def search(query: str, k: int = 8, subjects: list[str] | None = None) -> list[dict]:
    """Return up to k indexed chunks most relevant to the query (BM25)."""
    q = _fts_query(query)
    if not q:
        return []
    if not CHROMA_DB.exists():
        return _search_public_knowledge(query, k, subjects)
    con = sqlite3.connect(f"file:{CHROMA_DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    results: list[dict] = []
    try:
        rows = cur.execute(
            "SELECT rowid, bm25(embedding_fulltext_search) AS score, string_value "
            "FROM embedding_fulltext_search WHERE string_value MATCH ? "
            "ORDER BY score LIMIT ?",
            (q, max(k * 12, k)),
        ).fetchall()
        candidates: list[dict] = []
        for r in rows:
            meta = _meta(cur, r["rowid"])
            document_name = meta.get("document_name", "unknown source")
            candidates.append({
                "text": (r["string_value"] or "").strip(),
                "score": r["score"],
                "document_name": document_name,
                "category": meta.get("category", ""),
                "subcategory": meta.get("subcategory", ""),
                "chunk_index": meta.get("chunk_index", ""),
                "total_chunks": meta.get("total_chunks", ""),
            })
        hints = tuple(
            hint
            for subject in (subjects or [])
            for hint in _SUBJECT_SOURCE_HINTS.get(subject, ())
        )
        if hints:
            strong, weak, fallback = [], [], []
            for candidate in candidates:
                category_blob = (candidate["category"] + " " + candidate["subcategory"]).lower()
                name_blob = candidate["document_name"].lower()
                if any(hint in category_blob for hint in hints):
                    strong.append(candidate)
                elif any(hint in name_blob for hint in hints):
                    weak.append(candidate)
                else:
                    fallback.append(candidate)
            candidates = strong + weak + fallback
        per_document: dict[str, int] = {}
        for candidate in candidates:
            document_name = candidate["document_name"]
            if per_document.get(document_name, 0) >= 2:
                continue
            results.append(candidate)
            per_document[document_name] = per_document.get(document_name, 0) + 1
            if len(results) >= k:
                break
    except sqlite3.OperationalError:
        return []
    finally:
        con.close()
    return results or _search_public_knowledge(query, k, subjects)


def build_source_ledger(results: list[dict], max_chars: int = 900) -> str:
    """Format retrieved chunks into a numbered source ledger for the prompt."""
    if not results:
        return "SOURCE LEDGER: (no indexed sources matched — say so and rely on general knowledge with caution)"
    lines = ["SOURCE LEDGER (legal knowledge base — cite only what these support):"]
    for i, r in enumerate(results, 1):
        loc = (f", chunk {r['chunk_index']}/{r['total_chunks']}"
               if r.get("chunk_index") not in ("", None) else "")
        text = re.sub(r"\s+", " ", r["text"])[:max_chars]
        lines.append(f"[{i}] {r['document_name']}{loc}\n    {text}")
    return "\n".join(lines)


def search_feedback_guidance(query: str, k: int = 5,
                             subjects: list[str] | None = None) -> list[dict]:
    """Retrieve assessment technique without mixing it into legal authorities.

    All marked work is indexed. Runtime retrieval permits: official marker
    guidance, marker feedback from every script, and submitted answers only
    where the overall script achieved 70+. Lower-mark answer prose remains
    indexed for audit/training diagnostics but is never a runtime exemplar.
    """
    q = _fts_query(query, max_terms=16)
    if not q:
        return []
    if not FEEDBACK_DB.exists():
        terms = _query_terms(query)
        subject_set = {subject for subject in (subjects or []) if subject}
        candidates: list[tuple[int, dict]] = []
        paths: list[Path] = []
        standard = PUBLIC_GUIDES_DIR / "first_class_writing_standards.md"
        if standard.is_file():
            paths.append(standard)
        paths += [PUBLIC_GUIDES_DIR / f"{subject}.md" for subject in subject_set]
        for path in paths:
            if not path.is_file():
                continue
            for section_index, (title, text) in enumerate(_public_sections(path), 1):
                heading = text.splitlines()[0].lower() if text else ""
                if path != standard and not any(label in heading for label in (
                    "answer method", "feedback rules", "strong first-class", "avoid"
                )):
                    continue
                blob = text.lower()
                score = sum(min(blob.count(term), 3) for term in terms)
                if path == standard:
                    score += 4
                candidates.append((score, {
                    "text": text,
                    "source": f"Bundled anonymized writing standard — {title}",
                    "source_kind": "public_guide",
                    "quality_tier": "marker_guidance",
                    "subject": path.stem if path != standard else "general",
                    "final_mark": None,
                    "section_kind": "marker_feedback",
                    "chunk_index": section_index,
                    "total_chunks": "",
                }))
        candidates.sort(key=lambda item: -item[0])
        return [item for _score, item in candidates[:k]]
    con = sqlite3.connect(f"file:{FEEDBACK_DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        filters = [subject for subject in (subjects or []) if subject]
        subject_sql = ""
        params: list[object] = [q]
        if filters:
            placeholders = ",".join("?" for _ in filters)
            subject_sql = f" AND (subject = 'general' OR subject IN ({placeholders}))"
            params.extend(filters)
        params.append(k)
        rows = con.execute(
            f"""SELECT rowid, bm25(feedback_chunks) AS score, text, source,
                      source_kind, quality_tier, subject, final_mark, section_kind,
                      chunk_index, total_chunks
                 FROM feedback_chunks
                WHERE feedback_chunks MATCH ?
                  AND (quality_tier IN ('marker_guidance', 'gold')
                       OR section_kind = 'marker_feedback')
                  {subject_sql}
                ORDER BY score LIMIT ?""",
            params,
        ).fetchall()
        return [dict(row) for row in rows]
    except sqlite3.OperationalError:
        return []
    finally:
        con.close()


def build_feedback_guidance(results: list[dict], max_chars: int = 650) -> str:
    """Format a technique-only block which the model is forbidden to cite."""
    if not results:
        return ""
    lines = [
        "ASSESSMENT & WRITING GUIDANCE (NOT legal authority; never cite this block):",
        "Use only for structure, issue coverage, analysis depth and avoiding prior mark-loss patterns. "
        "Verify every legal proposition against the SOURCE LEDGER or official law.",
    ]
    for index, result in enumerate(results, 1):
        mark = f", mark {result['final_mark']}" if result.get("final_mark") else ""
        text = re.sub(r"\s+", " ", result["text"])[:max_chars]
        lines.append(
            f"[G{index}] {result['source']} ({result['quality_tier']}{mark}; "
            f"{result.get('subject', 'general')}; "
            f"{result['section_kind']})\n    {text}"
        )
    return "\n".join(lines)


if __name__ == "__main__":  # quick manual test
    import sys, json
    q = " ".join(sys.argv[1:]) or "consideration and practical benefit in contract"
    res = search(q, k=5)
    print(f"query: {q!r} -> {len(res)} hits\n")
    for r in res:
        print(f"- [{r['score']:.2f}] {r['document_name']} (chunk {r['chunk_index']})")
    print("\n--- ledger preview ---")
    print(build_source_ledger(res[:3], max_chars=200))
