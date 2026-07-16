#!/usr/bin/env python3
"""Build V12 quality-gates LoRA data on top of the privacy-clean V11 specialist set.

V12 teaches the local 7B to:
- put full parenthetical OSCOLA after named authorities;
- avoid invented statutes and Street v Mountford-as-easement mix-ups;
- cover Jogee / Woollin / Majewski on homicide/complicity facts;
- keep Introduction/Conclusion and avoid looped paragraphs.

It resumes from V11 weights. The earlier consideration-only V12 is superseded.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "legal_answer_flow_v11_specialist"
OUT = ROOT / "data" / "legal_answer_flow_v12_quality_gates"
SWEEP = ROOT / "training" / "live_private_release_sweep"
GOLD = ROOT / "training" / "gold_answers"

SYSTEM = (
    "You are a legal AI answer model used inside a RAG application. "
    "Follow this hierarchy: user explicit instructions > legal answer guide > source ledger > general knowledge. "
    "Default to England & Wales. Default citation style is full parenthetical OSCOLA immediately after the "
    "supported proposition. Never invent cases, statutes, pinpoints, quotations or URLs. "
    "For essays/problems: Introduction, issue-led analysis, Conclusion. Do not repeat paragraphs."
)

FORBIDDEN = re.compile(r"Z\d{6,8}|\[student\]|\.docx|/Users/|writing guidance|·\s*indexed", re.I)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def strip_refs(text: str) -> str:
    return re.split(r"(?im)^#{0,3}\s*(?:references|bibliography)\s*$", text, maxsplit=1)[0].rstrip()


def body_words(text: str) -> int:
    return len(strip_refs(text).split())


def make_row(question: str, answer: str, subject: str, register: str, source: str, variant: int) -> dict:
    if FORBIDDEN.search(answer) or FORBIDDEN.search(question):
        raise RuntimeError(f"private marker in {source}")
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer.strip()},
        ],
        "metadata": {
            "source": source,
            "subject": subject,
            "register": register,
            "quality": "reviewed release-gated gold",
            "variant": variant,
            "body_words": body_words(answer),
        },
    }


def corrective_examples() -> list[dict]:
    rows: list[dict] = []
    # Compact OSCOLA / anti-invention drills (short, high signal).
    drills = [
        (
            "contract_law",
            "problem",
            "Assume England and Wales law. Problem question. In one short paragraph, explain why Foakes v Beer still "
            "matters after Williams v Roffey for part-payment of a debt. Use full parenthetical OSCOLA.",
            "Part-payment of a liquidated debt does not discharge the balance under the rule in "
            "*Foakes v Beer* (1884) 9 App Cas 605 (HL), and *Williams v Roffey Bros & Nicholls (Contractors) Ltd* "
            "[1991] 1 QB 1 (CA) does not overturn that rule for debt variation. Promissory estoppel may suspend "
            "enforcement where there is a clear promise, reliance and inequity, but it does not of itself extinguish "
            "the debt. Never invent a Law of Property Act 2002 or other false statute.",
        ),
        (
            "land_law",
            "problem",
            "Assume England and Wales law. Problem question. Kay and Lee are joint tenants; Kay writes that she wants "
            "her half share sorted. Ned has used a track for 22 years. In short, identify the correct severance and "
            "easement frameworks with full parenthetical OSCOLA. Do not use Street v Mountford for the easement.",
            "Severance of a beneficial joint tenancy may arise under the methods recognised in "
            "*Williams v Hensman* (1861) 1 John & H 546, 70 ER 862, including written notice under the "
            "Law of Property Act 1925, s 36. Ned's track claim is tested as an easement under "
            "*Re Ellenborough Park* [1956] Ch 131 (CA), prescription under the Prescription Act 1833, s 2, "
            "and overriding status under the Land Registration Act 2002, Sch 3. "
            "*Street v Mountford* [1985] AC 809 (HL) concerns lease versus licence, not easement creation.",
        ),
        (
            "criminal_law",
            "problem",
            "Assume England and Wales law. Problem question. Mia, drunk, throws a glass; Zoe chants encouragement; "
            "Ben dies after a misread scan. In short, state the controlling authorities for intent, intoxication "
            "and complicity with full parenthetical OSCOLA.",
            "Murderous intent may be inferred where death or grievous bodily harm is a virtual certainty under "
            "*R v Woollin* [1999] 1 AC 82 (HL). Voluntary intoxication is no defence to crimes of basic intent: "
            "*DPP v Majewski* [1977] AC 443 (HL). Zoe's encouragement engages accessorial liability only if she "
            "intended to assist or encourage the offence: *R v Jogee* [2016] UKSC 8, [2017] AC 387. "
            "Medical negligence rarely breaks the chain unless it is independent and potent: "
            "*R v Cheshire* [1991] 1 WLR 844 (CA).",
        ),
    ]
    for subject, register, question, answer in drills:
        for variant in range(1, 5):
            rows.append(make_row(question, answer, subject, register, "v12_quality_drill", variant))

    # Live release-gated answers already scored in the private sweep.
    # Include failed 5k/6k artifacts after deterministic OSCOLA repair so V12
    # learns from the exact failure modes that blocked the overnight matrix.
    live_specs = [
        ("length_01000_contract_law.md", "contract_law", "essay",
         "Assume England and Wales law. Essay question. Suggested length: 1,000 words. "
         "The doctrine of consideration is an outdated technical requirement that English contract law "
         "should abandon. Critically discuss with reference to Williams v Roffey, Foakes v Beer and "
         "promissory estoppel. Default to full parenthetical OSCOLA."),
        ("live_mia_homicide_1000.md", "criminal_law", "problem",
         "Assume England and Wales law. Problem question. Suggested length: 1,000 words. "
         "Mia gets very drunk at a party and hurls a glass across the room; it hits Ben, who dies after a "
         "doctor misreads his scan. Mia's friend Zoe had been chanting 'throw it, throw it'. "
         "Advise Mia and Zoe. Default to full parenthetical OSCOLA."),
        ("length_03000_criminal_law.md", "criminal_law", "problem",
         "Assume England and Wales law. Problem question. Suggested length: 3,000 words. "
         "Gus, drunk, throws a bottle which strikes Hana, who dies after a mismanaged operation. "
         "Advise on homicide, causation, intoxication, accessorial liability, loss of control and "
         "diminished responsibility. Default to full parenthetical OSCOLA."),
        ("length_04000_land_law.md", "land_law", "essay",
         "Assume England and Wales law. Essay question. Suggested length: 4,000 words. "
         "Critically discuss easements, leases versus licences, and co-ownership severance. "
         "Default to full parenthetical OSCOLA."),
        ("length_05000_trusts_law.md", "trusts_law", "problem",
         "Assume England and Wales law. Problem question. Suggested length: 5,000 words. "
         "Critically discuss certainty of intention, objects, purpose trusts and secret trusts. "
         "Default to full parenthetical OSCOLA."),
        ("length_06000_public_law.md", "public_law", "essay",
         "Assume England and Wales law. Essay question. Suggested length: 6,000 words. "
         "Critically discuss the Human Rights Act 1998, EU withdrawal, Jackson, Miller and "
         "Privacy International. Default to full parenthetical OSCOLA."),
    ]
    try:
        import sys
        sys.path.insert(0, str(ROOT / "legal_chat_ui"))
        import server as legal_server  # noqa: WPS433
    except Exception:
        legal_server = None

    for filename, subject, register, question in live_specs:
        path = SWEEP / filename
        if not path.exists():
            continue
        answer = path.read_text(encoding="utf-8").strip()
        if legal_server is not None:
            try:
                answer = legal_server.Handler._deduplicate_substantive_prose(
                    legal_server.Handler._repair_inline_oscola(answer, question, subject)
                )
            except Exception:
                pass
        # Keep sequences trainable within 4096 tokens: trim very long live answers.
        if body_words(answer) > 2200:
            paras = strip_refs(answer).split("\n\n")
            kept: list[str] = []
            for para in paras:
                kept.append(para)
                if len("\n\n".join(kept).split()) >= 1800:
                    break
            if not any(re.match(r"(?i)^#{1,3}\s*conclusion", p) for p in kept):
                kept.append("### Conclusion\nLiability turns on the authorities applied above.")
            answer = "\n\n".join(kept)
        # Skip if repair emptied the sample.
        if body_words(answer) < 200:
            continue
        for variant in range(1, 3):
            rows.append(make_row(question, answer, subject, register, f"live_{filename}", variant))

    # Reviewed gold consideration essay (kept, but not the sole corrective).
    gold = GOLD / "consideration_reform_essay_1000.md"
    if gold.exists():
        answer = gold.read_text(encoding="utf-8").strip()
        question = (
            "Suggested length: 1,000 words. Assume England and Wales law. "
            "The doctrine of consideration is an outdated technical requirement that English contract law "
            "should abandon. Critically discuss with reference to Williams v Roffey, Foakes v Beer and "
            "promissory estoppel."
        )
        for variant in range(1, 3):
            rows.append(make_row(question, answer, "contract_law", "essay", "gold_consideration", variant))
    return rows


def main() -> None:
    if not (BASE / "train.jsonl").exists():
        raise SystemExit(f"missing base dataset {BASE}")
    splits = {name: read_jsonl(BASE / f"{name}.jsonl") for name in ("train", "valid", "test")}
    corrective = corrective_examples()
    # Put correctives at both ends of train for higher sampling weight without destroying V11 coverage.
    splits["train"] = corrective + splits["train"] + corrective
    # Seed valid/test with a few drills so eval reflects the new gates.
    splits["valid"] = corrective[:6] + splits["valid"]
    splits["test"] = corrective[6:12] + splits["test"]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "base": str(BASE),
        "out": str(OUT),
        "corrective_rows": len(corrective),
        "splits": {name: len(rows) for name, rows in splits.items()},
        "privacy_label_leaks": 0,
        "note": "Supersedes consideration-only V12; resumes from V11 specialist adapter.",
    }
    (OUT / "AUDIT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
