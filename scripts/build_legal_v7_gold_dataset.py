#!/usr/bin/env python3
"""Build an answering-only V7 LoRA set from locally approved work.

Unlike earlier corpora, every assistant target is answer prose. Work below 70
is never copied into targets. A small number of repaired/high-style sections
from an overall 72 script are sourced from the already curated gold transcript.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMPORTED = ROOT / "model_database" / "imported_feedback"
GOLD = ROOT / "model_database" / "snapshot" / "gold_standard_shapes"
OUT = ROOT / "data" / "legal_answer_flow_v7_marked_gold"

SYSTEM = (
    "You are writing the answer itself for an England and Wales law question. "
    "Use a direct thesis or roadmap, issue-led headings, exact legal tests, immediate fact application, "
    "the strongest counterargument, calibrated conclusions and practical remedies. Default to OSCOLA: "
    "put authority immediately after the proposition it supports and never invent a citation or pinpoint. "
    "Write complete continuous legal prose, not a plan, rubric, commentary or explanation of method."
)


def between(text: str, start: str, end: str | None = None) -> str:
    start_match = re.search(start, text, re.M | re.I)
    if not start_match:
        raise ValueError(f"Missing start marker: {start}")
    tail = text[start_match.start():]
    if end:
        end_match = re.search(end, tail, re.M | re.I)
        if not end_match:
            raise ValueError(f"Missing end marker: {end}")
        tail = tail[:end_match.start()]
    return tail.strip()


def clean(text: str) -> str:
    text = re.sub(r"^===== PAGE \d+ =====$", "", text, flags=re.M)
    drop = (
        r"^(?:Page \d+ of \d+|\[student\]|\(\[student\]\)|LAW\d+|Private International Law|"
        r"Pensions Law|[\[(]?\d{3,10}[\])]?|rage de .*|SOURCE:.*|SOURCE_KIND:.*|PAGES?:.*|"
        r"TRAINING_POLICY:.*)$"
    )
    text = re.sub(drop, "", text, flags=re.M | re.I)
    # Remove raw OCR footnote-only lines; inline authority in the gold transcript
    # and main prose remains. Numbered analytical paragraphs ("1.") are retained.
    text = re.sub(
        r"^\d{1,3}\s+(?=(?:ibid\b|[A-Z].*\b(?:v|Act|Convention|Regulation|Directive|Code|Report)\b)).*$",
        "", text, flags=re.M,
    )
    text = text.replace("Part Il:", "Part II:").replace("Part Ill:", "Part III:")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def encode_layout(text: str) -> str:
    """Protect headings/paragraphs while balancing chunks by word count."""
    out: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = " ".join(line.strip() for line in paragraph.splitlines() if line.strip())
        if not paragraph:
            continue
        match = re.match(r"^(#{2,4})\s+(.+)$", paragraph)
        if match:
            out.append(f"[[{match.group(1)}]]{match.group(2)}[[/H]]")
        elif re.match(r"^(?:Q\d+|Part [IVX]+:|[IVX]+\. |[A-Z]\. |Conclusion$)", paragraph):
            out.append(f"[[###]]{paragraph}[[/H]]")
        else:
            out.append(paragraph)
    return " [[P]] ".join(out)


def decode_layout(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = text.replace(" [[P]] ", "\n\n").replace("[[P]]", "\n\n")
    text = re.sub(r"\[\[(#{2,4})\]\](.*?)\[\[/H\]\]", r"\1 \2", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def balanced_chunks(text: str) -> list[str]:
    tokens = encode_layout(clean(text)).split()
    total = len(tokens)
    count = max(1, round(total / 620))
    while total / count > 800:
        count += 1
    while count > 1 and total / count < 420:
        count -= 1
    sizes = [total // count + (1 if i < total % count else 0) for i in range(count)]
    chunks: list[str] = []
    cursor = 0
    for size in sizes:
        chunks.append(decode_layout(tokens[cursor:cursor + size]))
        cursor += size
    return chunks


def source_texts() -> list[dict]:
    manifest_path = GOLD / "private_seed_map.local.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Private seed manifest is intentionally untracked; create it locally before rebuilding V7."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    roots = {"gold": GOLD, "imported": IMPORTED}
    for item in manifest:
        text = (roots[item["root"]] / item["file"]).read_text(encoding="utf-8")
        rows.append({
            "name": item["name"],
            "subject": item["subject"],
            "register": item["register"],
            "mark": item["mark"],
            "split": item["split"],
            "text": between(text, item["start"], item.get("end")),
        })
    return rows


def examples(source: dict) -> list[dict]:
    chunks = balanced_chunks(source["text"])
    rows: list[dict] = []
    variants = (
        "Write the answer text only at first-class standard.",
        "Produce polished submission-ready legal prose; do not narrate the method.",
        "Continue the developed legal answer without repetition or meta-commentary.",
    )
    for index, target in enumerate(chunks):
        previous = "" if index == 0 else " ".join(chunks[index - 1].split()[-150:])
        for variant_number, variant in enumerate(variants):
            if index == 0:
                instruction = (
                    f"Begin a {source['register']} in {source['subject']}. {variant} "
                    "Open with the thesis/roadmap and move directly into the first issue."
                )
            else:
                instruction = (
                    f"Continue the same {source['register']} in {source['subject']} from the prior ending below. "
                    f"{variant}\n\nPRIOR ENDING:\n{previous}"
                )
            rows.append({
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": target},
                ],
                "metadata": {
                    "source": source["name"], "subject": source["subject"],
                    "register": source["register"], "mark": source["mark"],
                    "part": index + 1, "parts": len(chunks), "variant": variant_number + 1,
                },
            })
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows: dict[str, list[dict]] = {"train": [], "valid": [], "test": []}
    audit: list[dict] = []
    for source in source_texts():
        chunks = balanced_chunks(source["text"])
        rows = examples(source)
        all_rows[source["split"]].extend(rows)
        audit.append({
            "source": source["name"], "split": source["split"], "mark": source["mark"],
            "chunks": len(chunks), "chunk_words": [len(chunk.split()) for chunk in chunks],
            "examples": len(rows),
        })
    for split, rows in all_rows.items():
        with (OUT / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "policy": "Assistant targets contain only marker-approved 70+ or curated-gold answer prose.",
        "splits": {split: len(rows) for split, rows in all_rows.items()},
        "sources": audit,
        "forbidden_terminal_markers": sum(
            "(End of Answer)" in row["messages"][-1]["content"]
            for rows in all_rows.values() for row in rows
        ),
    }
    (OUT / "AUDIT.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
