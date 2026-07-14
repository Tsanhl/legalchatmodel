#!/usr/bin/env python3
"""Exercise the running legal site exactly like a user, without retaining chats.

Every request is made through POST /api/chat in Private mode, consumes the SSE
response, runs the same deterministic release gates offline, writes an auditable
local test result, and permanently deletes the private conversation.  Generic
test prompts contain no user documents or identifiers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "legal_chat_ui"))
sys.path.insert(0, str(ROOT / "scripts"))

import guides  # noqa: E402
import pipeline  # noqa: E402
import server  # noqa: E402
from final_trial_sweep import QUESTIONS as LENGTH_QUESTIONS  # noqa: E402


BASE = "http://127.0.0.1:8765"
OUT_DIR = ROOT / "training" / "live_private_release_sweep"
REPORT = OUT_DIR / "report.jsonl"
OFFICIAL_HOSTS = (
    "legislation.gov.uk", "gov.uk", "bailii.org", "judiciary.uk",
    "supremecourt.uk", "parliament.uk", "sra.org.uk", "caa.co.uk", "justice.gov.uk",
    "electoralcommission.org.uk", "pensions-ombudsman.org.uk", "hcch.net",
    "sentencingcouncil.org.uk",
)
PRIVATE_RE = re.compile(
    r"(?:/Users/|\\Users\\|\[student\]|\bZ\d{6,8}\b|"
    r"\.docx(?:\.pdf)?\b|\bwriting guidance\b|·\s*indexed\b)", re.I,
)


# Direct practical enquiries deliberately use no requested length and no final
# References list.  Together they exercise the specialist guides missing from
# the original 100-question essay/problem bank.
GENERAL_ENQUIRIES: list[tuple[str, str]] = [
    ("aviation_law", "A passenger's international flight from London is cancelled and checked baggage is lost. Explain the main English-law and Montreal Convention routes, time limits, evidence and practical next steps."),
    ("civil_procedure_law", "A claimant served an English breach-of-contract claim, but the defendant says the particulars disclose no reasonable grounds. Explain strike out, summary judgment, evidence and costs in practical terms."),
    ("competition_law", "A small retailer suspects three suppliers are coordinating resale prices. Explain the Chapter I prohibition, evidence to preserve, CMA reporting and possible private remedies."),
    ("construction_law", "A contractor has not been paid under an English construction contract and wants adjudication. Explain the statutory right, timetable, enforcement and the main jurisdictional objections."),
    ("cultural_heritage_law", "A UK museum discovers that an acquired antiquity may have been unlawfully exported from its country of origin. Explain the legal and due-diligence steps it should take."),
    ("cybercrime_law", "An employee accessed a former employer's cloud account using an old password and downloaded files. Explain possible Computer Misuse Act liability, civil exposure and immediate practical steps."),
    ("election_law", "A local election candidate discovers a misleading anonymous online advert published during the campaign. Explain the principal English election-law routes, evidential needs and available complaints."),
    ("equality_law", "A disabled employee is refused home-working equipment recommended by occupational health. Explain reasonable adjustments, discrimination arising from disability, evidence and tribunal steps."),
    ("extradition_law", "A person arrested in England under a Part 1 extradition warrant fears prison mistreatment in the requesting state. Explain the hearing, human-rights objections, evidence and appeal route."),
    ("financial_regulation_law", "A UK start-up wants to operate an online investment platform. Explain how it should determine whether FCA authorisation, financial-promotion controls and client-money rules apply."),
    ("housing_law", "A private tenant in England receives a possession notice while serious disrepair remains unresolved. Explain validity checks, possible defences, counterclaims and urgent next steps."),
    ("insurance_law", "A small business failed to mention an earlier minor loss when buying insurance and the insurer now seeks to avoid the policy. Explain the Insurance Act 2015 analysis and remedies."),
    ("international_trade_law", "A UK exporter is told that a foreign government subsidy is harming sales. Explain the WTO and UK trade-remedies framework and the evidence needed before approaching the Trade Remedies Authority."),
    ("maritime_law", "Cargo is damaged during carriage from Liverpool under a bill of lading. Explain the Hague-Visby framework, who may sue, limitation, time bar and practical evidence."),
    ("mediation_law", "Two English companies are about to litigate a supply dispute. Explain mediation confidentiality, without-prejudice protection, settlement enforceability, costs consequences and practical preparation."),
    ("pensions_law", "A member of an occupational pension scheme believes the trustees misunderstood the scheme rules when refusing a benefit. Explain the internal dispute process, Ombudsman route, evidence and possible court issues."),
    ("private_international_law", "An English consumer bought goods from a French website with a French governing-law clause. Explain jurisdiction, applicable law and enforcement after Brexit."),
    ("public_procurement_law", "An unsuccessful bidder believes an English public authority changed its award criteria after tenders closed. Explain standstill, limitation, automatic suspension, disclosure and remedies."),
    ("sentencing_law", "A first-time offender has pleaded guilty in an English Crown Court. Explain how the court approaches offence category, culpability, harm, guilty-plea credit, mitigation and totality."),
    ("succession_wills", "A family finds a handwritten will signed by the deceased but witnessed by only one person. Explain formal validity, intestacy, possible rectification or construction issues and immediate probate steps."),
    ("tax_law", "An individual moving from Hong Kong to England asks when UK residence and remittance issues may arise. Give a cautious overview of the statutory residence test, records and need for current specialist advice."),
]


SQE_PROBES: list[tuple[str, str]] = [
    ("contract_law", "SQE single best answer. A posts acceptance on Tuesday. B emails a revocation on Wednesday, received immediately. A's letter arrives Thursday. When, if at all, was the contract formed? Give the answer first and explain why alternatives fail."),
    ("criminal_law", "SQE single best answer. D takes V's umbrella honestly believing it is D's own. Which element of theft is most clearly absent? Give one answer first and explain why."),
    ("land_law", "SQE single best answer. A registered proprietor grants B a legal easement expressly but it is not completed by registration. What is B's likely interest before registration? Give one answer first."),
    ("trusts_law", "SQE single best answer. T transfers shares to R to hold for B but fails to register R as shareholder and has done everything within T's power. What equitable principle is most relevant?"),
    ("tort_law", "SQE single best answer. A careless driver injures C; an unforeseeably fragile skull makes the injury much worse. Which remoteness rule applies? Give one answer first."),
    ("business_law", "SQE single best answer. A director has an undisclosed personal interest in a proposed company transaction. Which Companies Act 2006 duty is most directly engaged before the transaction?"),
    ("criminal_procedure_law", "SQE single best answer. A defendant charged with an either-way offence indicates a not-guilty plea. What allocation decision follows in the magistrates' court?"),
    ("evidence_law", "SQE single best answer. The prosecution wants to adduce a defendant's previous conviction to show propensity. Which statutory framework primarily governs admissibility?"),
    ("family_law", "SQE single best answer. An applicant seeks a divorce in England after one year of marriage. Must the applicant prove adultery, behaviour or separation under current law?"),
    ("human_rights_law", "SQE single best answer. A public authority acts incompatibly with a Convention right and no primary legislation compelled it. Which Human Rights Act 1998 provision supplies the direct unlawfulness route?"),
    ("intellectual_property_law", "SQE single best answer. An employee creates copyright software in the course of employment. Who is ordinarily the first owner under UK law?"),
    ("legal_ethics", "SQE single best answer. A solicitor discovers that a client intends to mislead the court using false evidence. Which duty takes priority and what must the solicitor do?"),
    ("public_law", "SQE single best answer. A minister uses a statutory power for a purpose Parliament did not authorise. Which judicial-review ground is most directly engaged?"),
    ("restitution_law", "SQE single best answer. P pays D under a fundamental mistake of fact and D has irreversibly changed position in good faith. Which defence is most relevant?"),
]


def request_json(path: str, payload: dict | None = None, method: str = "POST") -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


def post_chat(conversation_id: str, message: str, timeout: int = 14_400) -> tuple[str, list, list, str]:
    payload = json.dumps({
        "conversation_id": conversation_id,
        "message": message,
        "jurisdiction": "england_wales",
    }).encode()
    req = urllib.request.Request(
        BASE + "/api/chat", data=payload,
        headers={"Content-Type": "application/json"},
    )
    answer = ""
    sources: list = []
    statuses: list = []
    error = ""
    with urllib.request.urlopen(req, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data: "):
                continue
            try:
                event = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if "replace" in event:
                answer = event["replace"]
            elif "delta" in event:
                answer += event["delta"]
            if "sources" in event:
                sources = event["sources"] or []
            if "status" in event:
                statuses.append(event["status"])
            if "error" in event:
                error = event["error"]
    return answer, sources, statuses, error


def official_source_failures(sources: list) -> list[str]:
    failures: list[str] = []
    for source in sources:
        url = source.get("url", "") if isinstance(source, dict) else ""
        host = urllib.parse.urlparse(url).hostname or ""
        if not any(host == allowed or host.endswith("." + allowed) for allowed in OFFICIAL_HOSTS):
            failures.append(f"non-official source chip: {url or source!r}")
    return failures


def audit_answer(case_id: str, message: str, expected_slug: str, answer: str,
                 sources: list, statuses: list, error: str, elapsed: float) -> dict:
    target = pipeline.requested_word_count(message)
    slug = guides.detect_subject(message)
    body = server.Handler._without_reference_section(answer)
    failures: list[str] = []
    if error:
        failures.append("runtime error: " + error)
    if slug != expected_slug:
        failures.append(f"subject route {slug!r}, expected {expected_slug!r}")
    failures += server.Handler._generic_answer_failures(body, message, target=target)
    failures += server.Handler._complete_answer_failures(body, message)
    failures += server.Handler._subject_accuracy_failures(body, message, slug, "full answer")
    failures += official_source_failures(sources)
    for source in sources:
        if isinstance(source, dict) and not pipeline.official_result_matches_subject(slug, source):
            failures.append(f"official but subject-irrelevant source chip: {source.get('url') or source!r}")
    if PRIVATE_RE.search(answer):
        failures.append("private filename, path, identifier or internal label leaked")
    if not any("Searching indexed database + official sources" in status for status in statuses):
        failures.append("mandatory indexed/current-law search status was not emitted")
    if pipeline.needs_reference_list(message):
        if not re.search(r"(?im)^### References\s*$", answer):
            failures.append("essay/problem omitted the used-authority References section")
    elif re.search(r"(?im)^### References\s*$", answer):
        failures.append("general enquiry/SQE added a References list contrary to the default")
    failures = list(dict.fromkeys(failures))
    return {
        "id": case_id,
        "subject_expected": expected_slug,
        "subject_detected": slug,
        "requested_words": target,
        "body_words": len(body.split()),
        "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
        "answer_chars": len(answer),
        "seconds": round(elapsed, 2),
        "sources": sources,
        "statuses": statuses,
        "failures": failures,
        "passed": not failures,
    }


def run_case(case_id: str, message: str, expected_slug: str) -> dict:
    conversation = request_json(
        "/api/conversations",
        {"mode": "private", "jurisdiction": "england_wales"},
    )
    conversation_id = conversation["id"]
    started = time.time()
    answer = ""
    sources: list = []
    statuses: list = []
    error = ""
    delete_result: dict = {}
    try:
        answer, sources, statuses, error = post_chat(conversation_id, message)
        result = audit_answer(
            case_id, message, expected_slug, answer, sources, statuses, error,
            time.time() - started,
        )
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / f"{case_id}.md").write_text(answer, encoding="utf-8")
    except Exception as exc:
        result = {
            "id": case_id,
            "subject_expected": expected_slug,
            "seconds": round(time.time() - started, 2),
            "failures": [f"{type(exc).__name__}: {exc}"],
            "passed": False,
        }
    finally:
        try:
            delete_result = request_json(
                f"/api/conversations/{conversation_id}", None, method="DELETE"
            )
        except Exception as exc:
            delete_result = {"ok": False, "error": str(exc)}
    if not (delete_result.get("ok") and delete_result.get("permanent")):
        result.setdefault("failures", []).append(
            "private test conversation was not permanently deleted"
        )
        result["passed"] = False
    result["private_delete"] = delete_result
    with REPORT.open("a", encoding="utf-8") as output:
        output.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def cases_for_args(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    cases: list[tuple[str, str, str]] = []
    if args.lengths:
        start = max(args.start, 0)
        stop = min(args.stop if args.stop is not None else len(LENGTH_QUESTIONS), len(LENGTH_QUESTIONS))
        for index in range(start, stop):
            words, slug, register, stem = LENGTH_QUESTIONS[index]
            prompt = (
                f"Assume England and Wales law. {'Essay question' if register == 'essay' else 'Problem question'}. "
                f"Suggested length: {words:,} words. {stem} Default to full parenthetical OSCOLA."
            )
            cases.append((f"length_{words:05}_{slug}", prompt, slug))
    if args.general:
        start = max(args.start, 0)
        stop = min(args.stop if args.stop is not None else len(GENERAL_ENQUIRIES), len(GENERAL_ENQUIRIES))
        cases.extend(
            (f"general_{slug}", f"General legal enquiry. Subject: {slug.replace('_', ' ')}. Assume England and Wales law. {prompt}", slug)
            for slug, prompt in GENERAL_ENQUIRIES[start:stop]
        )
    if args.sqe:
        start = max(args.start, 0)
        stop = min(args.stop if args.stop is not None else len(SQE_PROBES), len(SQE_PROBES))
        cases.extend(
            (f"sqe_{slug}", f"Subject: {slug.replace('_', ' ')}. Assume England and Wales law. {prompt}", slug)
            for slug, prompt in SQE_PROBES[start:stop]
        )
    if args.case:
        cases = [case for case in cases if case[0] == args.case]
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", action="store_true")
    parser.add_argument("--general", action="store_true")
    parser.add_argument("--sqe", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--case")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    if not (args.lengths or args.general or args.sqe):
        args.lengths = args.general = args.sqe = True
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.fresh and REPORT.exists():
        REPORT.unlink()
    cases = cases_for_args(args)
    if not cases:
        raise SystemExit("No cases selected")
    results = [run_case(*case) for case in cases]
    summary = {
        "cases": len(results),
        "passed": sum(result.get("passed", False) for result in results),
        "failed": sum(not result.get("passed", False) for result in results),
    }
    print(json.dumps({"summary": summary}, indent=2), flush=True)
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
