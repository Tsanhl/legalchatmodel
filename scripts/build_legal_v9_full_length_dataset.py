#!/usr/bin/env python3
"""Build V9 data that teaches complete 1,000–2,500-word answers.

V8 learned from 600–800-word answer units, which improved long-form stitching
but encouraged early EOS on ordinary 1,000–1,500-word requests. V9 retains the
clean V8 corpus and adds full, reviewed essay/problem/general targets plus a
concise SQE target. Lower-mark prose and private identifiers remain excluded.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V8 = ROOT / "data" / "legal_answer_flow_v8_complete_answer"
GOLD = ROOT / "training" / "gold_answers"
OUT = ROOT / "data" / "legal_answer_flow_v9_full_length"

SYSTEM = (
    "Write the complete submission-ready England and Wales legal answer, never a plan. "
    "Use RAG/source material as evidence and apply the subject structure guide. For essays "
    "and problem questions, use explicit Introduction and Conclusion headings and one "
    "used-authority-only OSCOLA References section. For general enquiries and SQE answers, "
    "omit the final list unless requested. In every mode, put a full verified OSCOLA citation "
    "in parentheses immediately after the proposition it supports. Meet the requested body "
    "word count within minus/plus one per cent; references are outside the body count. Do not "
    "repeat analysis, invent authority, leak private filenames, or stop after an outline."
)


CASES = (
    {
        "file": "rare_equipment_contract_problem_1200.md",
        "subject": "contract law",
        "register": "problem question",
        "words": 1200,
        "question": (
            "Suggested length: 1,200 words. A buyer offers £40,000 for rare equipment. "
            "The seller replies, ‘Agreed, provided delivery is in July.’ The buyer says, "
            "‘Fine, but payment will be after inspection.’ The seller delivers in August "
            "and demands payment. Advise both parties on offer, counter-offer, acceptance, "
            "battle of forms, certainty, breach and remedies."
        ),
        "references": True,
    },
    {
        "file": "meddata_securecloud_contract_problem.md",
        "subject": "contract law",
        "register": "problem question",
        "words": 2000,
        "question": (
            "Suggested length: 2,000 words. Advise MedData Ltd and SecureCloud Ltd on "
            "misrepresentation, contractual terms, non-reliance, exclusions, limitation, "
            "breach, causation, remoteness, penalties and remedies."
        ),
        "references": True,
    },
    {
        "file": "nature_of_law_multi_theory_essay.md",
        "subject": "jurisprudence",
        "register": "critical essay",
        "words": 2000,
        "question": (
            "2,000 words. ‘The nature of law cannot be explained by one theory alone.’ "
            "Critically discuss using positivism, natural law, interpretivism, realism, "
            "feminism, critical race theory, Marxism and postcolonial theory."
        ),
        "references": True,
    },
    {
        "file": "dana_eli_farah_tort_problem.md",
        "subject": "tort law",
        "register": "problem question",
        "words": 1500,
        "question": (
            "Problem question — Tort Law. Suggested length: 1,500 words. Advise Dana, "
            "Eli and Farah on duty, breach, factual and legal causation, remoteness, "
            "contributory negligence, intervening acts and damages."
        ),
        "references": True,
    },
    {
        "file": "mistaken_privileged_documents_ethics_problem.md",
        "subject": "legal ethics",
        "register": "problem question",
        "words": 1500,
        "question": (
            "Legal Ethics — Problem. Suggested length: 1,500 words. A solicitor receives "
            "mistakenly disclosed privileged documents and the client instructs her to use "
            "them and tell nobody. Advise under current England and Wales rules."
        ),
        "references": True,
    },
    {
        "file": "proprietary_estoppel_practical_enquiry.md",
        "subject": "land law",
        "register": "general legal enquiry",
        "words": None,
        "question": (
            "Explain proprietary estoppel in practical terms: what must a claimant prove, "
            "what remedies can the court award, and what evidence should be preserved? "
            "Use full inline OSCOLA citations but no reference list."
        ),
        "references": False,
    },
    {
        "file": "sqe_postal_acceptance_answer.md",
        "subject": "contract law",
        "register": "SQE single-best-answer",
        "words": None,
        "question": (
            "SQE single best answer. A posts an offer, B posts acceptance on Tuesday, and "
            "A communicates revocation on Wednesday. Explain the correct result, material "
            "exceptions and practical evidence. Use inline OSCOLA but no reference list."
        ),
        "references": False,
    },
)


def without_references(text: str) -> str:
    return re.split(
        r"(?im)^#{0,3}\s*(?:references|bibliography|table of authorities)\s*$",
        text,
        maxsplit=1,
    )[0].replace("\n---\n", "\n").rstrip()


def body_words(text: str) -> int:
    return len(without_references(text).split())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    splits = {split: read_jsonl(V8 / f"{split}.jsonl") for split in ("train", "valid", "test")}
    added: list[dict] = []
    rows: list[dict] = []
    for number, spec in enumerate(CASES, 1):
        answer = (GOLD / spec["file"]).read_text(encoding="utf-8").strip()
        if not spec["references"]:
            answer = without_references(answer)
        actual = body_words(answer)
        requested = spec["words"]
        if requested is not None:
            lower = (requested * 99 + 99) // 100
            upper = requested * 101 // 100
            if not lower <= actual <= upper:
                raise RuntimeError(
                    f"{spec['file']} has {actual} body words, outside {lower}–{upper}"
                )
        if bool(re.search(r"(?im)^###\s*References\s*$", answer)) != bool(spec["references"]):
            raise RuntimeError(f"reference-mode mismatch: {spec['file']}")
        variants = 4 if spec["register"] in ("problem question", "critical essay") else 3
        for variant in range(variants):
            count_rule = (
                f" The requested body length is {requested:,} words and must land within ±1%."
                if requested else ""
            )
            ending_rule = (
                " Include one used-authority-only OSCOLA References section."
                if spec["references"] else
                " Do not include a References/Bibliography section; retain full inline OSCOLA."
            )
            rows.append({
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": spec["question"] + count_rule + ending_rule},
                    {"role": "assistant", "content": answer},
                ],
                "metadata": {
                    "source": f"reviewed_full_length_{number}",
                    "subject": spec["subject"],
                    "register": spec["register"],
                    "quality": "reviewed complete-answer gold",
                    "body_words": actual,
                    "variant": variant + 1,
                },
            })
        added.append({
            "source": f"reviewed_full_length_{number}",
            "subject": spec["subject"],
            "register": spec["register"],
            "body_words": actual,
            "examples": variants,
            "references": spec["references"],
        })

    # Put corrective examples first as well as after the inherited corpus so a
    # short incremental run sees them regardless of loader order.
    splits["train"] = rows + splits["train"] + rows
    forbidden = re.compile(
        r"Z\d{6,8}|\[student\]|LAW\d{4}-DE|\.docx|writing guidance|· indexed",
        re.I,
    )
    leaks: list[dict] = []
    for split, split_rows in splits.items():
        for row_number, row in enumerate(split_rows, 1):
            visible = "\n".join(message["content"] for message in row["messages"])
            if forbidden.search(visible):
                leaks.append({"split": split, "row": row_number})
    if leaks:
        raise RuntimeError(f"private/public leakage in V9 data: {leaks[:5]}")

    for split, split_rows in splits.items():
        with (OUT / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in split_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "policy": (
            "V8 clean corpus plus reviewed full 1,200–2,000-word answers, general/SQE "
            "inline-only modes, no lower-mark targets or private labels."
        ),
        "splits": {split: len(split_rows) for split, split_rows in splits.items()},
        "added": added,
        "privacy_label_leaks": len(leaks),
    }
    (OUT / "AUDIT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
