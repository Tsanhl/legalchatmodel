#!/usr/bin/env python3
"""Build a small, legally curated seed set for the failed LexiAI benchmark."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path


SYSTEM = (
    "You are a first-class England and Wales legal answer writer inside a supervised RAG application. "
    "For problem questions, organise by material issue, state the exact rule with a supported authority, "
    "apply decisive facts, address the strongest counterargument, rank the outcome, and state remedies. "
    "Use full OSCOLA citations in parentheses immediately after the proposition supported. Never invent "
    "an authority, quotation, pinpoint or fact. List only authorities actually used in References. For a "
    "planned long answer, answer only the allocated part and do not repeat another part."
)

ORIGINAL = (
    "LexiAI Ltd contracts with BrightCloud Ltd for cloud hosting. BrightCloud says its servers have 99.99% "
    "uptime, all user data will remain in the UK, and the system is suitable for sensitive legal-sector "
    "clients. The written contract promises commercially reasonable hosting, excludes liability for loss of "
    "profits, business and data, says LexiAI did not rely on pre-contractual statements, caps all liability at "
    "one monthly fee, and imposes a fixed GBP50,000 charge for late payment. Repeated outages cause LexiAI to "
    "lose law-firm clients; some data is processed outside the UK. BrightCloud blames a third-party provider. "
    "LexiAI withholds the final invoice and BrightCloud claims the invoice plus GBP50,000. Advise both parties."
)

PARTS = [
    (
        "[CURRENT PART TARGET: 650 WORDS]\n[PLANNED LONG ANSWER: PART 1 OF 3]\n"
        "Scope: terms and representations; misrepresentation; non-reliance.\nOriginal request: " + ORIGINAL,
        """## Part 1: Terms, Representations and Non-Reliance

The starting point is objective classification, not BrightCloud's private intention. A pre-contract statement is more likely to be contractual where it is precise, important to the recipient, made by a party with special knowledge, and followed shortly by contracting; the written agreement and any entire-agreement wording remain relevant but are not conclusive. BrightCloud's 99.99% uptime and UK-only processing statements are specific and verifiable. Their importance to a legal-chatbot business, coupled with BrightCloud's technical expertise, gives LexiAI a substantial argument that they became express or collateral terms. By contrast, suitability for “sensitive legal-sector clients” is evaluative, although an expert opinion may imply that reasonable grounds exist for it. The court will construe the written promise of “commercially reasonable hosting” in its documentary and commercial context, without using commercial common sense to rewrite clear language (*Arnold v Britton* [2015] UKSC 36, [2015] AC 1619; *Wood v Capita Insurance Services Ltd* [2017] UKSC 24, [2017] AC 1173).

Alternatively, each assurance may be an actionable representation. LexiAI must show a false existing fact or qualifying opinion that materially induced entry into the contract. Actual processing abroad strongly indicates that the UK-only statement was false. Repeated outages do not by themselves prove that the 99.99% figure was false when made: the measurement period, permitted maintenance, and method for calculating uptime are missing. LexiAI should obtain monitoring data, sales material, internal service records and evidence explaining why those assurances mattered to its decision. If inducement is proved, BrightCloud will be liable under section 2(1) unless it proves that it had reasonable grounds to believe, and did believe, the representation was true until contracting (Misrepresentation Act 1967, s 2(1)). Fraud would require proof that the statement was knowingly or recklessly false, a demanding allegation not established merely by later breach (*Derry v Peek* (1889) 14 App Cas 337 (HL)).

The non-reliance clause does not automatically defeat the claim. A term which excludes or restricts liability or a remedy for pre-contract misrepresentation is effective only if reasonable (Misrepresentation Act 1967, s 3). The Court of Appeal treats a contractual estoppel or non-reliance device according to its substance, so drafting it as an agreed state of affairs does not evade section 3 (*First Tower Trustees Ltd v CDS (Superstores International) Ltd* [2018] EWCA Civ 1396, [2019] 1 WLR 637). Reasonableness will turn on bargaining strength, negotiation, legal advice, available alternatives, insurance and whether BrightCloud could fairly ask LexiAI to disclaim reliance on precise claims used to sell the service. BrightCloud cannot exclude its own fraud (*HIH Casualty and General Insurance Ltd v Chase Manhattan Bank* [2003] UKHL 6, [2003] 2 Lloyd's Rep 61). On the stated facts, section 3 gives LexiAI a strong answer to the blanket non-reliance clause, particularly for UK-only processing and quantified uptime.

References

Misrepresentation Act 1967, ss 2(1) and 3.
*Arnold v Britton* [2015] UKSC 36, [2015] AC 1619.
*Derry v Peek* (1889) 14 App Cas 337 (HL).
*First Tower Trustees Ltd v CDS (Superstores International) Ltd* [2018] EWCA Civ 1396, [2019] 1 WLR 637.
*HIH Casualty and General Insurance Ltd v Chase Manhattan Bank* [2003] UKHL 6, [2003] 2 Lloyd's Rep 61.
*Wood v Capita Insurance Services Ltd* [2017] UKSC 24, [2017] AC 1173.""",
    ),
    (
        "[CURRENT PART TARGET: 650 WORDS]\n[PLANNED LONG ANSWER: PART 2 OF 3]\n"
        "Scope: breach; incorporation and construction; exclusions and cap; third-party provider.\nOriginal request: " + ORIGINAL,
        """## Part 2: Breach and the Liability Clauses

BrightCloud's primary obligation is to provide “commercially reasonable hosting”. Repeated outages are capable of breaching that standard, but frequency, duration, maintenance windows, industry practice and the agreed service levels must be proved. Processing data outside the UK will be a straightforward breach if UK-only storage was contractual; if it remained only a representation, it instead supports misrepresentation remedies. BrightCloud cannot answer its own breach merely by identifying a third-party infrastructure provider. Contractual performance remains BrightCloud's responsibility unless the agreement validly allocates that risk through a force-majeure or subcontractor clause and the event falls within it.

Assuming LexiAI signed the written agreement, incorporation is unlikely to be the central dispute; signature ordinarily binds absent misrepresentation or another recognised vitiating factor (*L'Estrange v F Graucob Ltd* [1934] 2 KB 394 (CA)). The real questions are construction and statutory control. Exclusion clauses are construed as part of the contract as a whole. Clear language may allocate serious breach risk; there is no automatic rule that a “fundamental breach” destroys an exclusion (*Photo Production Ltd v Securicor Transport Ltd* [1980] AC 827 (HL)). Any genuine ambiguity may be resolved against the proferens, especially for negligence or unusually extensive protection, but contra proferentem is a residual tool after ordinary contextual construction.

The exclusions cover loss of profits, business and data, while the cap limits total liability to one monthly fee. Their breadth does not itself make them void. If LexiAI contracted on BrightCloud's written standard terms, section 3 subjects a term permitting substantially different or no performance to reasonableness (Unfair Contract Terms Act 1977, s 3). Any attempt to exclude negligence causing property loss is also subject to reasonableness (Unfair Contract Terms Act 1977, s 2(2)). The section 11 test asks whether the term was fair and reasonable when the contract was made, taking account of matters such as bargaining strength, inducement, knowledge, alternatives and practicability of insurance (Unfair Contract Terms Act 1977, s 11 and sch 2). A negotiated cap between commercially sophisticated parties is more defensible, and courts are cautious about undoing commercial risk allocation (*Watford Electronics Ltd v Sanderson CFL Ltd* [2001] EWCA Civ 317, [2001] 1 All ER (Comm) 696). Yet a cap of one monthly fee may be vulnerable if BrightCloud sold the service for sensitive legal data while leaving LexiAI with virtually no remedy for systemic outage or unlawful location of processing.

The provisions must be tested separately. A profit-loss exclusion may reasonably allocate consequential business risk even if the overall cap survives. Excluding data loss is harder to justify where data integrity is central to the service and BrightCloud can insure or manage that risk. The court will also ask whether the cap applies to misrepresentation and statutory claims on its language; the non-reliance clause cannot bypass the Misrepresentation Act 1967 section 3 analysis. LexiAI therefore has an arguable and fact-sensitive UCTA challenge, strongest against the nominal cap and any clause that reduces the promised service to no meaningful obligation.

References

Misrepresentation Act 1967, s 3.
Unfair Contract Terms Act 1977, ss 2(2), 3 and 11, sch 2.
*L'Estrange v F Graucob Ltd* [1934] 2 KB 394 (CA).
*Photo Production Ltd v Securicor Transport Ltd* [1980] AC 827 (HL).
*Watford Electronics Ltd v Sanderson CFL Ltd* [2001] EWCA Civ 317, [2001] 1 All ER (Comm) 696.""",
    ),
    (
        "[CURRENT PART TARGET: 650 WORDS]\n[PLANNED LONG ANSWER: PART 3 OF 3]\n"
        "Scope: causation, remoteness, excluded losses, penalty, invoice, remedies and final advice.\nOriginal request: " + ORIGINAL,
        """## Part 3: Loss, Penalty and Remedies

LexiAI must prove that each claimed loss was caused by BrightCloud's breach. Client losses require evidence linking particular outages or overseas processing to termination, rather than general dissatisfaction or LexiAI's own product defects. LexiAI must also mitigate by using available failover, notifying BrightCloud, preserving data and taking reasonable steps to retain clients. Recoverable loss must arise naturally or have been within the parties' reasonable contemplation when contracting (*Hadley v Baxendale* (1854) 9 Exch 341). BrightCloud knew it was hosting a legal-chatbot platform, so some business interruption and lost customers were foreseeable. The scale and duration of claimed future profits, however, may be too uncertain or remote unless BrightCloud knew the relevant client contracts and dependency on uninterrupted service.

The profit, business and data exclusions may therefore matter more than common-law remoteness. If incorporated, clearly drafted and reasonable under UCTA, they may bar the principal heads of loss even where factual causation is shown. LexiAI should plead direct wasted expenditure, service fees paid for defective performance, restoration costs and any other loss outside the wording. Misrepresentation damages may offer a different route, but the court will examine whether the exclusion and cap validly extend to that liability and whether section 3 permits them (Misrepresentation Act 1967, s 3). Rescission may be available for actionable misrepresentation, subject to affirmation, lapse, impossibility of substantial restoration and third-party rights; in an ongoing service contract, termination and damages may be more practical.

The GBP50,000 late-payment charge is vulnerable under the penalty rule. The rule applies to a secondary obligation triggered by breach, not to the price or another primary obligation. The question is whether the detriment is out of all proportion to BrightCloud's legitimate interest in timely payment (*Cavendish Square Holding BV v Makdessi; ParkingEye Ltd v Beavis* [2015] UKSC 67, [2016] AC 1172). BrightCloud has a legitimate interest in cash flow and collection costs, but a fixed GBP50,000 charge regardless of invoice size, delay or actual cost appears extravagant unless the monthly fee and commercial consequences are themselves very substantial. LexiAI's penalty defence is therefore strong on the stated facts. The underlying invoice does not disappear merely because the charge is penal.

LexiAI may withhold or set off sums only if the contract and applicable set-off rules permit it. It should notify breach, quantify a good-faith cross-claim, preserve monitoring and client evidence, and avoid assuming that every invoice is extinguished. BrightCloud can sue for the unpaid fee and rely on the cap and exclusions, but it faces material risk on misrepresentation, section 3 reasonableness, UCTA reasonableness and the penalty.

Overall, LexiAI's strongest routes are breach of the hosting standard or UK-processing term, statutory misrepresentation concerning the precise sales assurances, and invalidity of the blanket non-reliance clause. Recovery of lost profits and data loss is less certain because causation, remoteness and the liability wording all require evidence. BrightCloud is likely entitled to the net unpaid invoice, subject to any valid set-off, but the GBP50,000 charge is likely unenforceable unless it can justify proportionality to a substantial legitimate interest.

References

Misrepresentation Act 1967, s 3.
*Cavendish Square Holding BV v Makdessi; ParkingEye Ltd v Beavis* [2015] UKSC 67, [2016] AC 1172.
*Hadley v Baxendale* (1854) 9 Exch 341.""",
    ),
]


def row(user: str, assistant: str) -> str:
    return json.dumps(
        {"messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def main() -> int:
    output = Path("data/legal_answer_flow_v5_first_class_seed")
    output.mkdir(parents=True, exist_ok=True)
    train = [row(*PARTS[0]), row(*PARTS[1]), row(*PARTS[2])] * 3
    feedback_root = Path(
        os.environ.get("LEGAL_FEEDBACK_SOURCE", "source-materials")
    ) / "user's request record for improvements"
    accepted_path = next(feedback_root.glob("*/corrections/*fbrec_e0f276e479e740d7bf78ec0db44937e5.json"))
    accepted = json.loads(accepted_path.read_text(encoding="utf-8"))
    accepted_target = re.split(r"\(End of Answer\)", str(accepted["user_feedback"]), maxsplit=1, flags=re.I)[0].strip()
    accepted_prompt = "Write the complete corrected legal answer. Original request: " + str(accepted["question"])
    valid = [row(accepted_prompt, accepted_target)]
    test = [row(*PARTS[2])]
    for name, lines in (("train", train), ("valid", valid), ("test", test)):
        (output / f"{name}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "train": len(train), "valid": len(valid), "test": len(test)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
