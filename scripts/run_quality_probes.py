#!/usr/bin/env python3
"""Run hard quality probes against the current local model.

The script sends user-like questions to the local backend while forcing the
Own / Local provider, then stores answer metadata for review and feedback.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_APP_URL = "http://127.0.0.1:8766"
DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-Uncensored-4bit"


PROBES: list[dict[str, Any]] = [
    {
        "id": "criminal_intention_1000",
        "label": "Criminal-law essay, local adaptive split plan",
        "online_mode": "off",
        "message": (
            "Question 1 - Essay: Criminal Law. Suggested length: 1,000 words. "
            "'The law on intention is too uncertain because it allows juries to decide moral "
            "blameworthiness rather than applying a clear legal test.' Discuss. Consider direct "
            "intention, oblique intention, foresight of consequences, murder, and whether the "
            "current law gives juries too much flexibility. Use OSCOLA-style citations."
        ),
        "expected_split_parts": 2,
    },
    {
        "id": "trusts_certainty_1250",
        "label": "Equity and trusts problem question, local adaptive split plan",
        "online_mode": "off",
        "message": (
            "Question 2 - Problem: Equity and Trusts. Suggested length: 1,250 words. Lena declares "
            "that she is holding GBP80,000 'for my niece Maya, but only if she uses it wisely'. She "
            "also tells her solicitor that she wants 'most of my valuable paintings to go to my close "
            "friends'. Later, Lena transfers GBP50,000 to her brother Raj and says, 'Please look after "
            "this for Maya until she turns 25.' Raj spends GBP10,000 on personal expenses. Advise Maya "
            "and Lena's friends. Consider certainty of intention, subject matter and objects, fixed "
            "and discretionary trusts, and breach of trust. Use OSCOLA-style citations."
        ),
        "expected_split_parts": 2,
    },
    {
        "id": "land_lease_licence_1500",
        "label": "Land-law essay, local adaptive split plan",
        "online_mode": "off",
        "message": (
            "Question 3 - Essay: Land Law. Suggested length: 1,500 words. 'The distinction between "
            "leases and licences is essential in theory, but in practice it often creates uncertainty "
            "and unfairness.' Critically discuss. Consider exclusive possession, certainty of term, "
            "rent, sham agreements, lodgers and family arrangements, and why the distinction matters. "
            "Use OSCOLA-style citations."
        ),
        "expected_split_parts": 2,
    },
    {
        "id": "public_law_homelessness_1750",
        "label": "Public-law judicial-review problem question, local adaptive split plan",
        "online_mode": "off",
        "message": (
            "Question 4 - Problem: Public Law / Judicial Review. Suggested length: 1,750 words. The "
            "Minister for Housing introduces a policy allowing local authorities to remove homeless "
            "applicants from waiting lists if they reject one offer of temporary accommodation. The "
            "policy is introduced without consultation, despite prior government guidance promising "
            "consultation with charities and local authorities on major homelessness reforms. HomeFirst "
            "argues that the policy will disproportionately affect disabled applicants and single parents. "
            "The Minister says urgent action was needed because of pressure on public housing. Advise "
            "HomeFirst on possible grounds for judicial review, including standing, legitimate expectation, "
            "procedural fairness, relevant considerations, irrationality, proportionality/equality and remedies. "
            "Use OSCOLA-style citations."
        ),
        "expected_split_parts": 2,
    },
    {
        "id": "company_directors_2000",
        "label": "Company-law essay, local adaptive split plan",
        "online_mode": "off",
        "message": (
            "Question 5 - Essay: Company Law. Suggested length: 2,000 words. 'Directors' duties under "
            "the Companies Act 2006 appear strict, but in reality the law gives directors significant "
            "freedom to take commercial risks.' Critically evaluate. Consider the duty to promote the "
            "success of the company, reasonable care, skill and diligence, conflicts, benefits from "
            "third parties, shareholder ratification, business judgment and judicial reluctance to interfere. "
            "Use OSCOLA-style citations."
        ),
        "expected_split_parts": 2,
    },
    {
        "id": "tort_psychiatric_harm_650",
        "label": "Tort problem question, local single answer",
        "online_mode": "off",
        "message": (
            "650 words. Tort Law - Problem Question.\n"
            "Amira is a paramedic sent to a motorway collision caused when Ben, a delivery driver, "
            "looked at his phone and hit a minibus. Amira did not see the impact but arrived within "
            "minutes, treated several injured children, and later developed PTSD. One child, Cara, "
            "was initially stable but died in hospital two days later after a negligent delay by the "
            "hospital. Cara's mother watched the resuscitation attempt over a video call and also "
            "developed PTSD. Advise Amira and Cara's mother on negligence liability against Ben and "
            "the hospital. Use only authorities you can support from sources; do not invent page "
            "numbers or quotations."
        ),
        "expected_terms": [
            "duty",
            "breach",
            "causation",
            "remoteness",
            "psychiatric harm",
            "primary victim",
            "secondary victim",
            "Alcock",
            "Paul",
        ],
    },
    {
        "id": "contract_damages_exclusion_700",
        "label": "Contract problem question, local single answer",
        "online_mode": "off",
        "message": (
            "700 words. Contract Law - Problem Question.\n"
            "EcoBrew buys a custom roasting machine from FixTech for its new cafe. FixTech's sales "
            "email says delivery will be by 1 May and attaches standard terms containing a clause "
            "excluding 'all liability for indirect, consequential or economic loss'. The machine is "
            "delivered late and defective. EcoBrew loses ordinary cafe profits and also loses a "
            "lucrative supermarket launch that FixTech knew about only because EcoBrew mentioned it "
            "during negotiations. Advise on incorporation, construction/control of the exclusion "
            "clause, and damages/remoteness. Use full case citations where possible and do not invent "
            "exact quotations."
        ),
        "expected_terms": [
            "incorporation",
            "exclusion clause",
            "construction",
            "UCTA",
            "Hadley",
            "remoteness",
            "consequential loss",
            "mitigation",
        ],
    },
    {
        "id": "evidence_hearsay_700",
        "label": "Criminal-evidence problem question, local single answer",
        "online_mode": "off",
        "target_words": 700,
        "message": (
            "700 words. Criminal Evidence - Problem Question. At Dario's trial for robbery, the "
            "prosecution wants to rely on a written statement from an eyewitness who has moved abroad "
            "and refuses to travel. The statement names Dario, but the witness had previously argued "
            "with him. The prosecution also proposes a police officer's account of an anonymous call. "
            "Advise on hearsay admissibility, safeguards and fairness. Do not invent quotations or "
            "page references; use supported authorities and OSCOLA-style citations."
        ),
        "expected_terms": [
            "hearsay",
            "Criminal Justice Act 2003",
            "section 114",
            "admissibility",
            "fairness",
        ],
    },
    {
        "id": "family_financial_remedies_750",
        "label": "Family-law problem question, local single answer",
        "online_mode": "off",
        "target_words": 750,
        "message": (
            "750 words. Family Law - Problem Question. Priya and Sam are divorcing after a 16-year "
            "marriage. Priya paused her career to care for their two children; Sam has a high income, "
            "a pension and a pre-marital flat. Priya seeks a home for the children and continuing support. "
            "Advise on financial remedies and how the court should approach needs, sharing, compensation "
            "and pension provision. Use supported authorities, with OSCOLA-style citations."
        ),
        "expected_terms": [
            "Matrimonial Causes Act 1973",
            "section 25",
            "needs",
            "sharing",
            "fairness",
        ],
    },
    {
        "id": "employment_unfair_dismissal_750",
        "label": "Employment-law problem question, local single answer",
        "online_mode": "off",
        "target_words": 750,
        "message": (
            "750 words. Employment Law - Problem Question. Noor, a warehouse supervisor with six years' "
            "service, is dismissed after one customer complaint. Her employer did not investigate, let "
            "her see the allegation, or hold a hearing. It says trust has broken down. Advise Noor on "
            "unfair dismissal, procedure and remedies. Use supported authorities and OSCOLA-style citations."
        ),
        "expected_terms": [
            "Employment Rights Act 1996",
            "section 98",
            "fair reason",
            "procedure",
            "Polkey",
        ],
    },
    {
        "id": "ip_copyright_750",
        "label": "Intellectual-property problem question, local single answer",
        "online_mode": "off",
        "target_words": 750,
        "message": (
            "750 words. Intellectual Property - Problem Question. A freelance photographer licenses five "
            "images to a museum for its exhibition catalogue. The museum later posts cropped versions "
            "online and a retailer copies two images into an advert. Advise on copyright subsistence, "
            "ownership, licence scope, infringement and possible defences. Use supported authorities and "
            "OSCOLA-style citations."
        ),
        "expected_terms": [
            "copyright",
            "Copyright, Designs and Patents Act 1988",
            "infringement",
            "substantial part",
            "defence",
        ],
    },
    {
        "id": "tax_characterisation_700",
        "label": "Tax-law conceptual problem question, local single answer",
        "online_mode": "off",
        "target_words": 700,
        "message": (
            "700 words. Tax Law - Problem Question. Without using current rates, thresholds or allowances, "
            "explain how a lawyer would analyse the tax characterisation of (i) profits from a side "
            "business, (ii) rent from a property, and (iii) profit on a one-off sale of shares. Distinguish "
            "income from capital, identify the statutory framework that would need checking, and state "
            "what further facts matter. Do not invent current rates or legislation."
        ),
        "expected_terms": [
            "income tax",
            "capital gains tax",
            "trading income",
            "chargeable gain",
            "statutory",
        ],
    },
    {
        "id": "private_international_choice_law_750",
        "label": "Private-international-law problem question, local single answer",
        "online_mode": "off",
        "target_words": 750,
        "message": (
            "750 words. Private International Law - Problem Question. A business in England contracts "
            "online with a supplier in France. The contract selects French law but is silent on jurisdiction. "
            "A dispute over defective goods is brought in England. Explain the structured issues for "
            "jurisdiction and applicable law, including the post-Brexit position that must be verified. "
            "Do not state uncertain current rules as fact; distinguish verified law from matters requiring "
            "a current source."
        ),
        "expected_terms": [
            "jurisdiction",
            "choice of law",
            "applicable law",
            "Rome I",
            "post-Brexit",
        ],
    },
    {
        "id": "public_international_state_responsibility_750",
        "label": "Public-international-law problem question, local single answer",
        "online_mode": "off",
        "target_words": 750,
        "message": (
            "750 words. Public International Law - Problem Question. Security contractors hired by State A "
            "detain and mistreat nationals of State B during a border operation. State A says the contractors "
            "were private actors. Advise State B on attribution, breach, excuses and reparation in state "
            "responsibility. Use supported sources and clearly distinguish treaty obligations that need "
            "fact-specific verification."
        ),
        "expected_terms": [
            "state responsibility",
            "attribution",
            "internationally wrongful act",
            "breach",
            "reparation",
        ],
    },
    {
        "id": "human_rights_article_8_750",
        "label": "Human-rights problem question, local single answer",
        "online_mode": "off",
        "target_words": 750,
        "message": (
            "750 words. Human Rights - Problem Question. A council installs facial-recognition cameras "
            "outside a benefits office and retains all images for five years. It says this deters fraud. "
            "Advise a claimant on a challenge under Article 8 ECHR and the Human Rights Act 1998, including "
            "interference, legality, legitimate aim, necessity and proportionality. Use OSCOLA-style citations."
        ),
        "expected_terms": [
            "Article 8",
            "interference",
            "legality",
            "legitimate aim",
            "proportionality",
        ],
    },
    {
        "id": "competition_article_101_750",
        "label": "Competition-law problem question, local single answer",
        "online_mode": "off",
        "target_words": 750,
        "message": (
            "750 words. Competition Law - Problem Question. Three regional food wholesalers agree to stop "
            "discounting below a common price and exchange future pricing plans at monthly meetings. Advise "
            "on the structured analysis under Article 101 TFEU, including agreement, restriction, effect "
            "on trade and exemption. Use supported authorities and OSCOLA-style citations."
        ),
        "expected_terms": [
            "Article 101",
            "agreement",
            "restriction",
            "effect on trade",
            "exemption",
        ],
    },
    {
        "id": "media_privacy_750",
        "label": "Media-and-privacy-law problem question, local single answer",
        "online_mode": "off",
        "target_words": 750,
        "message": (
            "750 words. Media and Privacy Law - Problem Question. A newspaper receives a hospital employee's "
            "photograph and medical details of a well-known musician. It plans to publish them alongside a "
            "story that the musician cancelled concerts. Advise on misuse of private information and the "
            "balance between privacy and expression. Use supported authorities and OSCOLA-style citations."
        ),
        "expected_terms": [
            "reasonable expectation of privacy",
            "Article 8",
            "Article 10",
            "public interest",
            "balancing",
        ],
    },
    {
        "id": "medical_consent_750",
        "label": "Medical-law consent problem question, local single answer",
        "online_mode": "off",
        "target_words": 750,
        "message": (
            "750 words. Medical Law - Problem Question. Dr Chen recommends surgery to Alex but does not "
            "mention a small but serious risk of paralysis or a less invasive alternative. The risk materialises. "
            "Advise on consent, disclosure and causation. Use supported authorities and OSCOLA-style citations."
        ),
        "expected_terms": [
            "Montgomery",
            "material risk",
            "reasonable alternative",
            "consent",
            "causation",
        ],
    },
    {
        "id": "land_coownership_3000_split",
        "label": "Land-law long answer plan above 2,500 words",
        "online_mode": "off",
        "message": (
            "3,000 words. Land Law - Problem Question. Aisha and Ben buy a registered house as joint "
            "legal owners. Aisha pays 80% of the price; Ben pays 20%. Ben later moves out and orally "
            "promises Aisha she can keep the house if she pays the mortgage alone. Ben sells his beneficial "
            "interest to Chris, who wants an order for sale. Aisha's adult daughter Dana also lives there "
            "and paid for an extension. Advise on co-ownership, trusts of land, actual occupation, "
            "overreaching, proprietary estoppel, and sale."
        ),
        "expected_split_parts": 2,
    },
    {
        "id": "company_directors_5000_split",
        "label": "Company-law long answer plan above 5,000 words",
        "online_mode": "off",
        "message": (
            "5,000 words. Company Law essay. Critically evaluate whether directors' duties under the "
            "Companies Act 2006 give directors substantial practical latitude despite their apparently "
            "strict statutory wording. Address sections 172, 174, 175, 176 and 239, business judgment, "
            "shareholder ratification and enforcement."
        ),
        "expected_split_parts": 3,
    },
    {
        "id": "public_law_7500_split",
        "label": "Public-law plan above 5,000 words",
        "online_mode": "off",
        "message": (
            "7,500 words. Public Law essay. Critically analyse judicial review of automated local authority "
            "decision-making that affects homelessness and social-care applicants, covering consultation, "
            "fairness, reasons, equality, proportionality, data protection and remedies."
        ),
        "expected_split_parts": 3,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local-model quality probes.")
    parser.add_argument("--app-url", default=DEFAULT_APP_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training/legal_answer_flow_feedback_v2/quality_runs"),
    )
    parser.add_argument("--timeout-seconds", type=float, default=260.0)
    parser.add_argument("--only", action="append", help="Run only probe ID(s).")
    return parser.parse_args()


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    return json.loads(raw)


def word_count(text: str) -> int:
    return len([token for token in (text or "").split() if token.strip()])


def _expected_term_present(term: str, answer: str) -> bool:
    lowered = answer.lower()
    match = re.fullmatch(r"section\s+(\d+[a-z]?)", term.strip(), flags=re.I)
    if match:
        number = re.escape(match.group(1))
        return bool(re.search(rf"\b(?:section|s\.?)\s*{number}\b", lowered))
    return term.lower() in lowered


def evaluate_probe(probe: dict[str, Any], response: dict[str, Any], elapsed: float) -> dict[str, Any]:
    answer = str(response.get("answer") or "")
    expected_terms = probe.get("expected_terms") or []
    missing_terms = [
        term for term in expected_terms
        if not _expected_term_present(term, answer)
    ]
    citation_guard = response.get("citation_guard") or {}
    split_plan = response.get("split_plan") or None
    expected_split_parts = probe.get("expected_split_parts")
    issues: list[str] = []
    if response.get("provider") not in {"local", "backend"}:
        issues.append(f"unexpected_provider:{response.get('provider')}")
    if response.get("provider") == "local" and response.get("model") != DEFAULT_MODEL:
        issues.append(f"unexpected_model:{response.get('model')}")
    if citation_guard and not citation_guard.get("ok", True):
        issues.extend(str(item) for item in citation_guard.get("issues") or ["citation_guard_failed"])
    if missing_terms:
        issues.append("missing_expected_terms:" + ", ".join(missing_terms[:8]))
    target_words = int(probe.get("target_words") or 0)
    if target_words and word_count(answer) < int(target_words * 0.75):
        issues.append(f"under_target_words:{word_count(answer)}_of_{target_words}")
    lowered = answer.lower()
    if "[interactive codex supervisor handoff]" in lowered or "backend-composed prompt excerpt:" in lowered:
        issues.append("internal_handoff_or_prompt_leak")
    if len(re.findall(r"(?im)^\s*(?:references|bibliography):?\s*$", answer)) > 1:
        issues.append("duplicate_reference_sections")
    sentence_counts: dict[str, int] = {}
    for sentence in re.split(r"(?<=[.!?])\s+", answer):
        normalized = " ".join(sentence.lower().split())
        if len(normalized) < 60:
            continue
        sentence_counts[normalized] = sentence_counts.get(normalized, 0) + 1
    if any(count >= 3 for count in sentence_counts.values()):
        issues.append("repeated_sentence_loop")
    if re.search(r"[A-Za-z]{3,}-$", answer.strip()):
        issues.append("truncated_word_at_answer_end")
    if expected_split_parts is not None:
        actual = int((split_plan or {}).get("suggested_parts") or 0)
        if actual != int(expected_split_parts):
            issues.append(f"split_parts_expected_{expected_split_parts}_got_{actual}")
    return {
        "probe_id": probe["id"],
        "label": probe["label"],
        "elapsed_seconds": round(elapsed, 2),
        "provider": response.get("provider"),
        "model": response.get("model"),
        "word_count": word_count(answer),
        "answer_preview": answer[:1200],
        "split_plan": split_plan,
        "retrieval": response.get("retrieval"),
        "online_fallback": response.get("online_fallback"),
        "citation_guard": citation_guard,
        "missing_expected_terms": missing_terms,
        "issues": issues,
        "message_ids": response.get("message_ids"),
        "conversation_id": response.get("conversation_id"),
    }


def main() -> int:
    args = parse_args()
    selected = set(args.only or [])
    probes = [probe for probe in PROBES if not selected or probe["id"] in selected]
    if not probes:
        raise SystemExit("No probes selected.")

    run = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "app_url": args.app_url,
        "model": args.model,
        "results": [],
    }

    for probe in probes:
        payload = {
            "message": probe["message"],
            "provider": "local",
            "model_name": args.model,
            "online_mode": probe.get("online_mode", "off"),
            "use_sources_first": True,
            "answer_style": "legal_answer",
            "reference_style": "oscola",
            "privacy_mode": "saved",
        }
        print(f"Running {probe['id']} ...", flush=True)
        started = time.monotonic()
        try:
            response = post_json(
                f"{args.app_url.rstrip('/')}/api/chat",
                payload,
                timeout=args.timeout_seconds,
            )
            result = evaluate_probe(probe, response, time.monotonic() - started)
            result["ok"] = not result["issues"]
            result["raw_response"] = response
        except Exception as exc:
            result = {
                "probe_id": probe["id"],
                "label": probe["label"],
                "ok": False,
                "issues": [f"request_failed:{exc}"],
                "elapsed_seconds": round(time.monotonic() - started, 2),
            }
        run["results"].append(result)
        status = "ok" if result.get("ok") else "issues"
        print(f"Finished {probe['id']}: {status} ({result['elapsed_seconds']}s)", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_quality_run.json"
    output_path.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {output_path}", flush=True)
    issue_count = sum(1 for result in run["results"] if result.get("issues"))
    print(f"probes: {len(run['results'])}; probes_with_issues: {issue_count}", flush=True)
    return 1 if issue_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
