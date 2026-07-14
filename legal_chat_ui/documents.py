"""Read user-uploaded documents (PDF / DOCX / TXT / MD / CSV) into text,
chunk them, and select the chunks most relevant to a question.

Uploaded documents become *retrievable context* (knowledge for this chat), not
training data. Term-overlap ranking keeps it dependency-light and fast.
"""

from __future__ import annotations

import re
from pathlib import Path

_STOP = {
    "the", "and", "for", "with", "from", "that", "this", "what", "which", "does",
    "under", "about", "law", "legal", "answer", "how", "why", "are", "was", "were",
    "a", "an", "of", "in", "on", "to", "is", "it", "as", "by", "or", "be",
}


def extract_text(path: str | Path) -> str:
    """Extract plain text from a supported document; '' if unsupported/failed."""
    p = Path(path)
    suffix = p.suffix.lower()
    try:
        if suffix == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(str(p))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        if suffix == ".docx":
            import docx
            d = docx.Document(str(p))
            return "\n".join(par.text for par in d.paragraphs)
        if suffix in (".txt", ".md", ".csv"):
            return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    return ""


def chunk_text(text: str, size: int = 1200, overlap: int = 150) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text or "").strip()
    if not text:
        return []
    # split on paragraph boundaries, then pack into ~size windows
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, buf = [], ""
    for para in paras:
        if len(buf) + len(para) + 1 <= size:
            buf = f"{buf}\n{para}".strip()
        else:
            if buf:
                chunks.append(buf)
            if len(para) <= size:
                buf = para
            else:  # hard-split an over-long paragraph
                for i in range(0, len(para), size - overlap):
                    chunks.append(para[i:i + size])
                buf = ""
    if buf:
        chunks.append(buf)
    return chunks


def _terms(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z][a-z0-9'\-]{1,}", (text or "").lower()) if t not in _STOP}


def relevant_chunks(query: str, chunks: list[str], k: int = 4) -> list[str]:
    """Rank chunks by term overlap with the query; return the top k."""
    q = _terms(query)
    if not q or not chunks:
        return chunks[:k]
    scored = []
    for ch in chunks:
        overlap = len(q & _terms(ch))
        if overlap:
            scored.append((overlap, ch))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ch for _, ch in scored[:k]] or chunks[:k]


def build_upload_ledger(doc_chunks: list[tuple[str, str]], start_index: int = 1,
                        max_chars: int = 900) -> str:
    """doc_chunks: list of (document_name, chunk_text). Formats a ledger block."""
    if not doc_chunks:
        return ""
    lines = ["USER-UPLOADED DOCUMENTS (highest priority — the user chose these):"]
    for i, (name, ch) in enumerate(doc_chunks, start_index):
        text = re.sub(r"\s+", " ", ch)[:max_chars]
        lines.append(f"[U{i}] {name}\n    {text}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else ""
    if p:
        txt = extract_text(p)
        chunks = chunk_text(txt)
        print(f"{Path(p).name}: {len(txt)} chars, {len(chunks)} chunks")
        q = sys.argv[2] if len(sys.argv) > 2 else "duty of care negligence"
        top = relevant_chunks(q, chunks, k=2)
        print(f"\ntop chunks for {q!r}:\n")
        for c in top:
            print("-", c[:220].replace("\n", " "), "\n")
