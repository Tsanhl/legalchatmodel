#!/usr/bin/env python3
"""Deterministic pre-deployment checks for the local legal AI site."""

from __future__ import annotations

import base64
import json
import os
import re
import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "legal_chat_ui"))

import guides  # noqa: E402
import online_search  # noqa: E402
import pipeline  # noqa: E402
import retrieval  # noqa: E402
import server  # noqa: E402
from live_private_release_sweep import GENERAL_ENQUIRIES, LENGTH_QUESTIONS, SQE_PROBES  # noqa: E402
from promote_feedback_to_lora_data import feedback_paths  # noqa: E402


def check(condition: bool, message: str, results: list[dict]) -> None:
    results.append({"check": message, "passed": bool(condition)})
    if not condition:
        raise AssertionError(message)


def main() -> None:
    results: list[dict] = []

    private_bank = ROOT / "data" / "legal_eval_100_questions.json"
    public_bank = ROOT / "training" / "public_legal_eval_100_questions.json"
    bank = json.loads((private_bank if private_bank.exists() else public_bank).read_text())
    check(len(bank) == 100, "100-question bank parsed", results)
    check(all(question["subjects"] for question in bank), "every question has a subject route", results)
    check(
        all(
            sum(part["words"] for part in question["part_plan"]) == question["word_count"]
            and max(part["words"] for part in question["part_plan"]) <= 800
            for question in bank if question["word_count"] > 2500
        ),
        "all long answers have exact totals and parts <=800 words",
        results,
    )
    check(
        {"consumer_law", "insolvency_law", "privacy_media_law", "remedies_law", "legal_ethics"}
        <= {subject for question in bank for subject in question["subjects"]},
        "new gap guides are exercised by the evaluation bank",
        results,
    )
    ethics_guide = guides.guide_method_for_question(
        "Legal Ethics problem: solicitor receives mistakenly disclosed privileged documents"
    )
    environmental_guide = guides.guide_method_for_question(
        "Environmental Law essay: environmental assessment, climate duties and judicial review"
    )
    check(
        "6.4(d)" in ethics_guide and "11 April 2025" in ethics_guide
        and "Full OSCOLA Authority Bank" in environmental_guide
        and "Current-Law Update Checkpoints" in environmental_guide,
        "runtime prompts include current rule checkpoints and full OSCOLA authority banks",
        results,
    )
    bundled_guides = sorted((ROOT / "legal_chat_ui" / "law_guides").glob("*.md"))
    bundled_text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in bundled_guides)
    public_prompt_source = (ROOT / "legal_chat_ui" / "pipeline.py").read_text(
        encoding="utf-8", errors="ignore"
    )
    check(
        len(bundled_guides) == 51
        and not re.search(
            r"\bZ\d{6,8}\b|\b20\d{6}\b|\.[a-z]+/attachments|"
            r"[A-Z0-9._%+-]+@gmail\.com|/Users/|real .{0,40} marker feedback|"
            r"\bLAW\d{4,6}\b|user(?:'s|s) own .{0,30} scripts",
            bundled_text,
            re.I,
        )
        and not re.search(r"user(?:'s)?\s+real\s+marker\s+feedback", public_prompt_source, re.I),
        "51 bundled legal guides are standalone and contain no known private identifiers",
        results,
    )
    by_id = {question["id"]: question for question in bank}
    check(
        by_id[30]["word_count"] == 1500
        and by_id[60]["word_count"] == 3500
        and by_id[85]["word_count"] == 4500,
        "group-range text cannot override an individual question's Suggested length",
        results,
    )

    feedback_db = ROOT / "model_database" / "feedback_index.sqlite3"
    if feedback_db.exists():
        with sqlite3.connect(feedback_db) as con:
            total, files = con.execute(
                "SELECT COUNT(*), COUNT(DISTINCT source) FROM feedback_chunks"
            ).fetchone()
            diagnostic_answers = con.execute(
                "SELECT COUNT(*) FROM feedback_chunks WHERE quality_tier='diagnostic' "
                "AND section_kind='submitted_answer'"
            ).fetchone()[0]
        check(files == 18 and total > 250, "all 18 supplied documents are feedback-indexed", results)
        check(diagnostic_answers > 0, "lower-mark work is retained for diagnostic audit", results)
    else:
        check(
            (ROOT / "legal_chat_ui" / "law_guides" / "first_class_writing_standards.md").exists(),
            "public clone uses the anonymized bundled writing standard without private feedback files",
            results,
        )
    runtime_guidance = retrieval.search_feedback_guidance(
        "criminal law murder consent application key case", 12, ["criminal_law"]
    )
    check(
        all(not (hit["quality_tier"] == "diagnostic" and hit["section_kind"] == "submitted_answer")
            for hit in runtime_guidance),
        "lower-mark submitted prose is excluded from runtime exemplars",
        results,
    )

    original_chroma, original_feedback = retrieval.CHROMA_DB, retrieval.FEEDBACK_DB
    try:
        retrieval.CHROMA_DB = ROOT / "__missing_public_chroma__.sqlite3"
        retrieval.FEEDBACK_DB = ROOT / "__missing_public_feedback__.sqlite3"
        public_hits = retrieval.search(
            "consideration practical benefit Foakes v Beer promissory estoppel",
            4,
            ["contract_law"],
        )
        public_guidance = retrieval.search_feedback_guidance(
            "critical contract essay consideration",
            3,
            ["contract_law"],
        )
    finally:
        retrieval.CHROMA_DB, retrieval.FEEDBACK_DB = original_chroma, original_feedback
    check(
        len(public_hits) == 4
        and any(hit["category"] == "contract_law" for hit in public_hits)
        and public_guidance
        and all("Bundled anonymized" in hit["source"] for hit in public_guidance),
        "public clone has relevant legal RAG and writing guidance without the private database",
        results,
    )

    contract = (
        "Contract Law Problem. Consider: misrepresentation, contractual terms, non-reliance clauses, "
        "exclusion clauses, limitation clauses, breach, causation, remoteness, penalties and remedies."
    )
    hits = retrieval.search(contract, 6, ["contract_law"])
    check(
        len(hits) == 6 and any(
            "contract" in (hit["category"] + " " + hit["document_name"]).lower()
            for hit in hits[:3]
        ),
        "contract problem retrieves relevant contract/commercial material",
        results,
    )
    check(
        pipeline.plan_sections(contract + " Suggested length: 2,000 words", 2000)
        == [("full answer", 2000)],
        "answers up to 2,500 words use one complete supervised generation",
        results,
    )

    jurisprudence = (
        '2,000 words “The nature of law cannot be explained by one theory alone.”\n\n'
        'Critically discuss using positivism, natural law, interpretivism, realism, feminism, '
        'critical race theory, Marxism and postcolonial theory.'
    )
    check(
        pipeline.extract_subissues(jurisprudence)
        == [
            "positivism", "natural law", "interpretivism", "realism", "feminism",
            "critical race theory", "Marxism", "postcolonial theory",
        ]
        and pipeline.plan_sections(jurisprudence, 2000) == [("full answer", 2000)],
        "critically-discuss-using theory lists are extracted while 2,000 words remains single-pass",
        results,
    )
    curated_jurisprudence = pipeline.curated_regression_answer(jurisprudence)
    curated_body = re.split(
        r"(?im)^#{1,3}\s*References\s*$", curated_jurisprudence, maxsplit=1
    )[0]
    check(
        1980 <= len(curated_body.split()) <= 2020
        and curated_jurisprudence.count("### References") == 1,
        "multi-theory jurisprudence regression has a complete 2,000-word gold answer",
        results,
    )
    scale_question = jurisprudence.replace("2,000 words", "20,000 words")
    scale_plan = pipeline.plan_sections(scale_question, 20000)
    check(
        len(scale_plan) == 25
        and sum(words for _title, words in scale_plan) == 20000
        and max(words for _title, words in scale_plan) <= 800
        and len({title for title, _words in scale_plan}) == 25
        and not any("further analysis" in title.lower() for title, _words in scale_plan)
        and not pipeline.curated_regression_answer(scale_question),
        "20,000-word answers use 25 distinct analytical units and do not use a 2,000-word fixture",
        results,
    )
    check(
        all(
            sum(words for _title, words in pipeline.plan_sections(jurisprudence, total)) == total
            and max(words for _title, words in pipeline.plan_sections(jurisprudence, total))
            <= max(800, total)
            for total in (1000, 1400, 2000, 2500, 2501, 5000, 10000, 20000)
        ),
        "planner preserves every requested total from 1,000 through 20,000 words",
        results,
    )
    tort_problem = (
        "Problem question — Tort Law\nSuggested length: 1,500 words\n"
        "Dana drives into Eli. Advise Dana, Eli and Farah on duty, breach, factual and legal causation, "
        "remoteness, contributory negligence, intervening acts and damages."
    )
    tort_plan = pipeline.plan_sections(tort_problem, 1500)
    check(
        tort_plan == [("full answer", 1500)],
        "a 1,500-word problem answer remains a single complete supervised generation",
        results,
    )
    check(
        bool(server.Handler._tort_accuracy_failures(
            "The breach was unlawful means conspiracy and the business loss cannot be recovered because it "
            "flows from Eli's own personal injury.", tort_problem, "damages"
        )),
        "tort gate rejects live-regression hallucinations and economic-loss reversal",
        results,
    )
    medical_tort_question = (
        "Priya, a junior doctor, misreads a scan and discharges Tom, who later suffers a stroke; Una "
        "develops PTSD and Tom's employer loses a contract. Address Bolam/Bolitho/Montgomery, psychiatric "
        "injury and pure economic loss."
    )
    check(
        len(server.Handler._tort_accuracy_failures(
            "Priya at the Royal Northern Hospital misread an MRI. Bolam/Bolitho/Montgomery is the single "
            "diagnosis test. The employer's contract loss is likely recoverable as pure economic loss.",
            medical_tort_question,
        )) >= 5
        and "Paul v Royal Wolverhampton" in pipeline.tort_accuracy_lock(
            medical_tort_question
        ),
        "medical-negligence gate rejects invented facts, Montgomery misuse and relational-loss error",
        results,
    )
    road_tort_question = (
        "Problem question — Tort Law. Suggested length: 1,500 words. Dana collides with Eli, Farah "
        "blocks the ambulance, Eli wore no seat belt, and specialist equipment is destroyed."
    )
    road_tort_answer = pipeline.curated_regression_answer(road_tort_question)
    road_tort_body = road_tort_answer.split("\n---\n", 1)[0]
    check(
        1485 <= len(road_tort_body.split()) <= 1515
        and "### Legal causation and intervening acts" in road_tort_body
        and "### Eli's contributory negligence" in road_tort_body
        and not server.Handler._tort_accuracy_failures(road_tort_body, road_tort_question, "complete answer"),
        "reviewed road-negligence fallback is complete and within the 1,500-word ±1% band",
        results,
    )
    substantive = """### Positivism
The social-fact account explains validity through institutional sources and a rule of recognition. Hart's
separation thesis does not deny that morality can influence law; it denies that moral merit is a necessary
condition of legal validity. However, the account leaves a distinct question about why subjects ought to obey.

### Natural law and interpretivism
Natural lawyers connect legality with practical reason, while interpretivism treats legal rights as flowing from
the justification that best fits institutional practice. Each therefore exposes limits in a purely source-based
account, although neither eliminates the need to identify enacted and adjudicated materials.

### Critical perspectives
Realist, feminist, critical-race, Marxist and postcolonial approaches shift attention to adjudication, hierarchy
and lived effects. Their strongest objection is that abstract validity tests can conceal patterned power. Yet a
complete theory must also explain doctrinal constraint rather than treating every result as politics alone.

### Conclusion
No single theory explains validity, obligation, interpretation and social effect equally well. A plural account is
therefore preferable, provided that it preserves the genuine disagreements among the traditions."""
    check(
        bool(server.Handler._generic_answer_failures(
            "This essay will outline what a top-band answer should discuss.", jurisprudence, 667
        ))
        and not server.Handler._generic_answer_failures(substantive, jurisprudence, 120),
        "actual-answer gate rejects plans and accepts substantive answer prose",
        results,
    )
    sanitized_substantive = server.Handler._sanitize_final(
        "This essay will critically discuss the proposition.\n\n" + substantive
    )
    check(
        "this essay will" not in sanitized_substantive.lower()
        and sanitized_substantive.startswith("### Positivism")
        and not server.Handler._generic_answer_failures(sanitized_substantive, jurisprudence, 120),
        "isolated meta-writing is removed without discarding the substantive answer",
        results,
    )
    breadth_question = (
        "Critically discuss the balance between autonomy and fairness using examples from at least eight LLB subjects."
    )
    eight_sections = "\n\n".join(
        f"### {name}\nHowever, {name} supplies its own distinct doctrinal example, authority and counterargument."
        for name in ("Contract law", "Tort law", "Criminal law", "Public law", "Human rights law",
                     "EU law", "Land law", "Equity and trusts")
    ) + "\n\n### Conclusion\nOverall, the subjects resolve competing values through different institutional tests."
    check(
        pipeline.requested_subject_breadth(breadth_question) == 8
        and server.Handler._generic_answer_failures(eight_sections, breadth_question, None) == []
        and any("requires at least 8" in failure for failure in
                server.Handler._generic_answer_failures("### Contract law\nHowever, one example.",
                                                        breadth_question, None)),
        "express multi-subject breadth is enforced instead of merely listing values",
        results,
    )
    bad_cross_subject = (
        "### Contract law\nCavendish held a clause void as public policy. "
        "### Tort law\nRobinson held police had no duty to prevent crime. "
        "### Criminal law\nSalomons v A Salomon supplied fair warning. "
        "### Public law\nMiller concerned repeal of Acts of Parliament. "
        "### Human rights law\nBank Mellat was proportionate. "
        "### EU law\nVan Gend en Loos. ### Land law\nStack v Dowden. "
        "### Equity and trusts\nMcPhail concerned breach of trust."
    )
    check(
        len(server.Handler._generic_answer_failures(bad_cross_subject, breadth_question, None)) >= 5,
        "cross-subject gate rejects invented holdings and misplaced authorities",
        results,
    )
    reviewed_cross_subject = pipeline.curated_regression_answer(
        "The central problem of law is how to balance autonomy, certainty, fairness, accountability and power. "
        "Critically discuss using examples from at least eight LLB subjects."
    )
    check(
        len(re.findall(r"(?m)^###\s+(?!References)", reviewed_cross_subject)) >= 9
        and "### Comparative conclusion" in reviewed_cross_subject
        and "Patterson v Ashbourne" not in reviewed_cross_subject
        and "Article 36 TFEU" not in reviewed_cross_subject,
        "reviewed cross-subject fallback supplies a complete authority-checked essay",
        results,
    )
    check(
        "[2]" not in server.Handler._sanitize_final("A proposition supported by [2] and [source 7]."),
        "internal retrieval-ledger labels are removed from user answers",
        results,
    )
    loop_sentence = (
        "However, the facts do not specify a minor detail that would affect only the evidential weight "
        "and should not be repeated throughout the legal analysis."
    )
    loop_clean = server.Handler._sanitize_final(
        "### Breach\nThe objective standard is satisfied. " + loop_sentence + " " + loop_sentence
    )
    check(
        loop_clean.count(loop_sentence) == 1,
        "loop recovery stops before retaining a duplicate long sentence",
        results,
    )
    estoppel_question = (
        "Explain proprietary estoppel in practical terms: what must a claimant prove, what remedies can the "
        "court award, and what evidence should be preserved?"
    )
    estoppel_answer = pipeline.curated_regression_answer(estoppel_question)
    check(
        all(heading in estoppel_answer for heading in (
            "### What the claimant must prove", "### What remedies the court can award",
            "### Evidence to preserve now", "### Practical assessment",
        ))
        and not server.Handler._generic_answer_failures(estoppel_answer.split("\n---\n", 1)[0],
                                                        estoppel_question, None),
        "reviewed general-enquiry fallback answers elements, remedies and evidence directly",
        results,
    )
    exact_estoppel_question = (
        "General legal enquiry — England and Wales. In exactly 1,000 words, explain the requirements "
        "for proprietary estoppel, the role of detriment and unconscionability, and the available remedies. "
        "Use full OSCOLA references in parentheses immediately after relevant propositions. "
        "Do not include a final reference list."
    )
    exact_estoppel_fixture = pipeline.curated_regression_answer(exact_estoppel_question)
    fixture_handler = server.Handler.__new__(server.Handler)
    exact_estoppel_body = fixture_handler._without_reference_section(exact_estoppel_fixture)
    exact_estoppel_body = fixture_handler._safe_enforce_body_word_band(
        exact_estoppel_body, exact_estoppel_question, "", "land_law", 1000,
        exact_estoppel_body,
    )
    check(
        990 <= len(exact_estoppel_body.split()) <= 1010
        and len(fixture_handler._extract_full_inline_citations(exact_estoppel_body)) >= 4
        and "### References" not in exact_estoppel_body
        and not fixture_handler._subject_accuracy_failures(
            exact_estoppel_body, exact_estoppel_question, "land_law"
        ),
        "reviewed proprietary-estoppel enquiry is accurate, inline-cited and within 1,000 words ±1%",
        results,
    )
    bad_estoppel = (
        "Assurance, reliance, detriment, unconscionability and remedies apply. "
        "Caparo Industries plc v Dickman established unconscionability. Gillett v Holt [2005] followed. "
        "Guest v Guest [2008] was a Court of Appeal basement dispute between partners. "
        "Hunt v Soady governs detriment."
    )
    check(
        len(fixture_handler._proprietary_estoppel_accuracy_failures(
            bad_estoppel, exact_estoppel_question
        )) >= 4,
        "proprietary-estoppel gate rejects the hallucinations found by the live browser probe",
        results,
    )
    app_js = (ROOT / "legal_chat_ui" / "static" / "app.js").read_text(encoding="utf-8")
    index_html = (ROOT / "legal_chat_ui" / "static" / "index.html").read_text(encoding="utf-8")
    server_source = (ROOT / "legal_chat_ui" / "server.py").read_text(encoding="utf-8")
    check(
        "data.replace !== undefined" in app_js,
        "browser can publish the final count-checked answer as one replacement event",
        results,
    )
    check(
        "let terminalError" in app_js
        and "if (!terminalError && acc.trim())" in app_js,
        "a terminal quality error cannot be overwritten by provisional answer text",
        results,
    )
    check(
        "recoverSavedAnswer" in app_js
        and "assistantSavedAfter" in app_js
        and "Connection interrupted; recovering the checked answer" in app_js
        and "let transportError" in app_js
        and "resumeActiveGeneration" in app_js
        and "generation_active" in server_source,
        "browser recovers a saved complete answer after a long-stream transport interruption",
        results,
    )
    check(
        "legal_answer_flow_v11_specialist_lora" in server_source
        and "Latest V11 specialist legal training is active" in app_js,
        "V11 specialist adapter is the deployed default",
        results,
    )
    adapter_path = ROOT / "adapters" / server.APPROVED_ADAPTER_DIR
    adapter_weights = adapter_path / "adapters.safetensors"
    adapter_integrity_ok = False
    if adapter_weights.is_file() and adapter_weights.stat().st_size >= 1_000_000:
        try:
            server.validate_approved_adapter(adapter_path)
            adapter_integrity_ok = True
        except RuntimeError:
            adapter_integrity_ok = False
    elif adapter_weights.is_file():
        adapter_integrity_ok = adapter_weights.read_text(
            encoding="utf-8", errors="ignore"
        ).startswith("version https://git-lfs.github.com/spec/v1")
    check(
        server.APPROVED_ADAPTER_DIR == "legal_answer_flow_v11_specialist_lora"
        and server.APPROVED_ADAPTER_SHA256
        == "18dcd485f52b5747059c03fa0c620ccc027820d0241b04977fc4a0223679e69a"
        and adapter_path.is_dir()
        and adapter_integrity_ok,
        "the approved V11 release is pinned and inferior experiments cannot replace the default",
        results,
    )
    check(
        index_html.count('value="always"') == 1
        and 'value="auto"' not in index_html
        and 'value="off"' not in index_html
        and 'online_mode = "always"' in server_source,
        "official latest-law search is active for every user enquiry",
        results,
    )
    find_case_law_html = """
    <tbody><tr><td><a href="/uksc/2025/10?query=fiduciary+profit" class="link">
    Rukhadze and others v Recovery Partners GP Ltd and another</a></td></tr>
    <tr><td><p>fiduciary duty to account for profits and fully informed consent</p></td>
    <td>[2025] UKSC 10</td><td>19 Mar 2025</td></tr></tbody>
    """
    original_online_get = online_search._get
    try:
        online_search._get = lambda *_args, **_kwargs: find_case_law_html
        current_cases = online_search._search_find_case_law(
            "fiduciary duty conflict no profit Boardman v Phipps", 2
        )
    finally:
        online_search._get = original_online_get
    check(
        len(online_search._case_search_query(
            "fiduciary duty conflict no profit Boardman v Phipps Armitage v Nurse"
        ).split()) <= 5
        and current_cases
        and current_cases[0]["citation"] == "[2025] UKSC 10"
        and current_cases[0]["current_case"]
        and pipeline.official_result_matches_subject("trusts_law", current_cases[0]),
        "official online flow searches and subject-gates current UKSC authority",
        results,
    )
    check(
        pipeline.required_current_case(
            "Fiduciary no-profit essay addressing accounts of profits and allowances.",
            "trusts_law",
            current_cases,
        ) == current_cases[0]
        and pipeline.required_current_case(
            "General construction-adjudication timetable enquiry.",
            "construction_law",
            [{"citation": "[2024] UKSC 23"}],
        ) is None,
        "mandatory current-case gate is limited to precise doctrinal matches",
        results,
    )
    check(
        not pipeline.official_result_matches_subject(
            "trusts_law",
            {
                "title": "Abbasi v Newcastle upon Tyne Hospitals NHS Foundation Trust [2025] UKSC 15",
                "snippet": "The NHS trusts sought injunctions concerning patient care.",
                "current_case": True,
            },
        )
        and not pipeline.official_result_matches_subject(
            "maritime_law",
            {
                "title": "Secretary of State v Mercer [2024] UKSC 12",
                "snippet": "National Union of Rail, Maritime and Transport Workers.",
                "current_case": True,
            },
        )
        and not pipeline.official_result_matches_subject(
            "contract_law",
            {
                "title": "R v Hayes; R v Palombo [2025] UKSC 29",
                "snippet": "An interest rate swap is a contract.",
                "current_case": True,
            },
        )
        and not pipeline.official_result_matches_subject(
            "election_law",
            {
                "title": "Representation of the People Act 2000",
                "url": "https://www.legislation.gov.uk/id/ukpga/2000/2",
                "snippet": "Registration of voters and voting at elections.",
            },
        )
        and not pipeline.official_result_matches_subject(
            "tort_law",
            {
                "title": "Paul v Royal Wolverhampton NHS Trust [2024] UKSC 1",
                "snippet": "Secondary victims and psychiatric injury in clinical negligence.",
                "current_case": True,
            },
            "A fragile skull makes an injury worse. Which remoteness rule applies?",
        )
        and not pipeline.official_result_matches_subject(
            "business_law",
            {
                "title": "Saxon Woods Investments Ltd v Costa [2026] UKSC 21",
                "snippet": "A company director must act in the company's best interests.",
                "current_case": True,
            },
            "A director has an undisclosed personal interest in a proposed transaction.",
        ),
        "current-case source chips reject lexical false friends across broad subjects",
        results,
    )
    fiduciary_question = (
        "Equity and Trusts — Essay. Suggested length: 2,000 words. "
        "Fiduciary obligations are strict because equity distrusts divided loyalty."
    )
    trust_map = guides.authority_citation_map_for_question(fiduciary_question, "trusts_law")
    repaired_trust = server.Handler._repair_inline_oscola(
        "Armitage v Nurse (1893) explains the irreducible core, while Boardman v Phipps (1920) "
        "shows strict gain-based liability.",
        fiduciary_question,
        "trusts_law",
    )
    check(
        any("Armitage v Nurse [1998] Ch 241 (CA)" == value for value in trust_map.values())
        and "(1893)" not in repaired_trust
        and "(1920)" not in repaired_trust
        and "*Armitage v Nurse* [1998] Ch 241 (CA)" in repaired_trust
        and "*Boardman v Phipps* [1967] 2 AC 46 (HL)" in repaired_trust,
        "trusts authority bank canonicalises invented dates and courts",
        results,
    )
    bad_fiduciary = (
        "A trustee balances duties to the settlor and beneficiaries. Armitage v Nurse (1893) was decided "
        "by Lord Halsbury. Re Rose and Saunders v Vautier show flexibility."
    )
    good_fiduciary = (
        "The no-conflict rule is prophylactic and the no-profit rule supports an account of profits. "
        "A fiduciary may proceed after fully informed consent or authorisation."
    )
    check(
        len(server.Handler._trust_accuracy_failures(bad_fiduciary, fiduciary_question)) >= 4
        and not server.Handler._trust_accuracy_failures(good_fiduciary, fiduciary_question),
        "fiduciary gate rejects divided-loyalty and irrelevant-doctrine hallucinations",
        results,
    )
    unsafe_question = (
        "Employment Law — Problem. An employee is dismissed after refusing to return to an unsafe workplace."
    )
    bad_unsafe = (
        "The dismissal is defined by section 94(2). Section 5 requires workplace safety. "
        "Burchell was a Court of Appeal decision."
    )
    good_unsafe = (
        "Employment Rights Act 1996 section 100 applies where the employee reasonably believed danger was "
        "serious and imminent. Reinstatement is a discretionary tribunal order."
    )
    check(
        len(server.Handler._employment_accuracy_failures(bad_unsafe, unsafe_question)) >= 4
        and not server.Handler._employment_accuracy_failures(good_unsafe, unsafe_question)
        and pipeline.official_online_query(unsafe_question, "employment_law").startswith(
            "Employment Rights Act 1996 section 100"
        ),
        "unsafe-workplace gate selects the automatic-unfair-dismissal route and focused online query",
        results,
    )
    bad_aviation = (
        "The Montreal Convention, 1929 applies. Air France v Saks [2023] Bus LR 1879. "
        "Article 19 pays compensation after three hours. If it fails, sue in negligence."
    )
    aviation_question = (
        "Aviation law enquiry: an international flight is cancelled and checked baggage is lost."
    )
    aviation_map = guides.authority_citation_map_for_question(aviation_question, "aviation_law")
    repaired_aviation = server.Handler._repair_inline_oscola(
        bad_aviation, aviation_question, "aviation_law"
    )
    check(
        len(server.Handler._aviation_accuracy_failures(bad_aviation, aviation_question)) >= 6
        and aviation_map.get("air france v saks") == "Air France v Saks 470 US 392 (1985)"
        and "[2023] Bus LR 1879" not in server.Handler._sanitize_final(
            repaired_aviation, "Air France v Saks 470 US 392 (1985)"
        ),
        "aviation gate rejects Warsaw/Montreal confusion, fake citations and unrestricted tort fallback",
        results,
    )
    aviation_gold_question = (
        "General legal enquiry. Subject: aviation law. A passenger's international flight from "
        "London is cancelled and checked baggage is lost. Explain the main English-law and Montreal "
        "Convention routes, time limits, evidence and practical next steps."
    )
    aviation_gold = pipeline.curated_regression_answer(aviation_gold_question)
    check(
        bool(aviation_gold)
        and not server.Handler._generic_answer_failures(aviation_gold, aviation_gold_question)
        and not server.Handler._complete_answer_failures(aviation_gold, aviation_gold_question)
        and not server.Handler._aviation_accuracy_failures(aviation_gold, aviation_gold_question)
        and "(O1" not in server.Handler._sanitize_final("A proposition (O1, O2)."),
        "reviewed aviation enquiry separates UK261 and Montreal baggage rights without source-label leakage",
        results,
    )
    civil_question = (
        "Civil procedure enquiry: explain strike out and summary judgment under English law."
    )
    bad_civil = "The court will grant summary judgment because the claim is weak."
    good_civil = (
        "Strike out is governed by CPR r 3.4. Summary judgment under CPR r 24.3 requires no real "
        "prospect of success and no other compelling reason for trial; the outcome remains discretionary."
    )
    check(
        len(server.Handler._civil_procedure_accuracy_failures(bad_civil, civil_question)) >= 4
        and not server.Handler._civil_procedure_accuracy_failures(good_civil, civil_question)
        and bool(guides.authority_citation_map_for_question(civil_question, "civil_procedure_law")),
        "civil-procedure gate distinguishes CPR rr 3.4 and 24.3 and checks both summary-judgment limbs",
        results,
    )
    civil_gold_question = (
        "General legal enquiry. Subject: civil procedure law. A claimant served an English "
        "breach-of-contract claim, but the defendant says the particulars disclose no reasonable "
        "grounds. Explain strike out, summary judgment, evidence and costs in practical terms."
    )
    civil_gold = pipeline.curated_regression_answer(civil_gold_question)
    check(
        bool(civil_gold)
        and not server.Handler._generic_answer_failures(civil_gold, civil_gold_question)
        and not server.Handler._complete_answer_failures(civil_gold, civil_gold_question)
        and not server.Handler._civil_procedure_accuracy_failures(civil_gold, civil_gold_question),
        "reviewed civil-procedure enquiry uses the current October 2023 Part 24 numbering",
        results,
    )
    fiduciary_gold = pipeline.curated_regression_answer(
        "2,000 words: fiduciary obligations are strict because equity distrusts divided loyalty."
    )
    fiduciary_body = server.Handler._without_reference_section(fiduciary_gold)
    fiduciary_runtime_body = server.Handler._repair_inline_oscola(
        fiduciary_body,
        fiduciary_question + " Address allowances, accounts of profits and proprietary remedies.",
        "trusts_law",
    )
    fiduciary_references = server.Handler._authorities_table(
        fiduciary_runtime_body, fiduciary_question
    )
    check(
        1980 <= len(fiduciary_body.split()) <= 2020
        and 1980 <= len(fiduciary_runtime_body.split()) <= 2020
        and "Rukhadze v Recovery Partners GP Ltd" in fiduciary_runtime_body
        and "[2025] UKSC 10" in fiduciary_runtime_body
        and "Stevens v Hotel Portfolio II UK Ltd" in fiduciary_runtime_body
        and "Hopcraft v Close Brothers Ltd" in fiduciary_runtime_body
        and "*Aberdeen Railway Co v Blaikie Brothers* (1854) 1 Macq 461 (HL)" in fiduciary_references
        and "**Other authorities**\n- (1854) 1 Macq 461" not in fiduciary_references
        and not server.Handler._current_authority_failures(
            fiduciary_runtime_body,
            {"required_current_authority": {
                "name": "Rukhadze and others v Recovery Partners GP Ltd and another [2025] UKSC 10",
                "citation": "[2025] UKSC 10",
            }},
        )
        and not server.Handler._generic_answer_failures(
            fiduciary_runtime_body, fiduciary_question, target=2000
        )
        and not server.Handler._complete_answer_failures(fiduciary_runtime_body, fiduciary_question)
        and not server.Handler._trust_accuracy_failures(fiduciary_runtime_body, fiduciary_question),
        "reviewed fiduciary-loyalty essay is complete, accurate and within the 2,000-word ±1% band",
        results,
    )
    check(
        bool(server.Handler._current_authority_failures(
            "Only Boardman v Phipps and FHR are discussed.",
            {"required_current_authority": {
                "name": "Rukhadze and others v Recovery Partners GP Ltd and another [2025] UKSC 10",
                "citation": "[2025] UKSC 10",
            }},
        )),
        "release gate rejects an answer that silently omits the relevant current official judgment",
        results,
    )
    unsafe_gold_question = (
        "1,500 words: An employee is dismissed after refusing to return to an unsafe workplace."
    )
    unsafe_gold = pipeline.curated_regression_answer(unsafe_gold_question)
    unsafe_body = server.Handler._without_reference_section(unsafe_gold)
    check(
        1485 <= len(unsafe_body.split()) <= 1515
        and guides.detect_subject(unsafe_gold_question) == "employment_law"
        and not server.Handler._generic_answer_failures(
            unsafe_body, unsafe_gold_question, target=1500
        )
        and not server.Handler._complete_answer_failures(unsafe_body, unsafe_gold_question)
        and not server.Handler._employment_accuracy_failures(unsafe_body, unsafe_gold_question),
        "reviewed unsafe-workplace problem is complete, accurate and within the 1,500-word ±1% band",
        results,
    )
    unsafe_variant = (
        "Problem question — Employment Law, 1,500 words. Maya, an employee, refuses to return "
        "to a warehouse after reporting a danger. The employer dismisses her for insubordination. "
        "Advise both parties under the law of England and Wales."
    )
    check(
        pipeline.curated_regression_answer(unsafe_variant).strip()
        == pipeline.curated_regression_answer(unsafe_gold_question).strip(),
        "unsafe-workplace reviewed answer recognises ordinary user wording, not one exact test phrase",
        results,
    )
    leaked_case_bank = server.Handler._sanitize_final(
        "Ready Mixed Concrete: Facts: driver engaged under mixed terms. Held: employee status applied. "
        "Reasoning: multi-factor test. Answer use: employee status baseline. The legal analysis continues."
    )
    check(
        not re.search(r"\b(?:Facts|Held|Reasoning|Answer use):", leaked_case_bank, re.I)
        and "legal analysis continues" in leaked_case_bank.lower(),
        "internal case-brief annotations are removed without deleting later analysis",
        results,
    )

    meddata_gold = pipeline.curated_regression_answer(
        "MedData Ltd and SecureCloud Ltd: fully NHS-grade, hosted entirely in the UK, known vulnerability, "
        "£100,000. Suggested length: 2,000 words."
    )
    meddata_body = meddata_gold.split("\n---\n", 1)[0]
    check(
        not server.Handler._complete_answer_failures(meddata_body, "Problem question. Advise both parties.")
        and server.Handler._complete_answer_failures(
            "### Issues\nA short answer names Caparo v Dickman without a full citation.",
            "Problem question. Advise both parties.",
        ),
        "complete-answer gate enforces Introduction, Conclusion and full inline OSCOLA",
        results,
    )
    check(
        not server.Handler._part_release_failures(meddata_body, 1, 1)
        and server.Handler._part_release_failures("### Analysis\nNo authority is supplied.", 1, 2)
        and server.Handler._part_release_failures("### Analysis\nNo authority is supplied.", 2, 2),
        "each long-form unit is gated for opening/closing structure and full inline OSCOLA",
        results,
    )
    ethics_gold = pipeline.curated_regression_answer(by_id[30]["prompt"])
    ethics_body = server.Handler._without_reference_section(ethics_gold)
    check(
        1485 <= len(ethics_body.split()) <= 1515
        and "### Introduction" in ethics_body
        and "### Conclusion" in ethics_body
        and "SRA Code of Conduct" in ethics_body
        and not server.Handler._generic_answer_failures(ethics_body, by_id[30]["prompt"], 1500)
        and not server.Handler._complete_answer_failures(ethics_body, by_id[30]["prompt"]),
        "reviewed legal-ethics fallback is current, complete and within the 1,500-word ±1% band",
        results,
    )
    formation_question = (
        "Suggested length: 1,200 words. A buyer emails a seller offering £40,000 for rare equipment. "
        "The seller replies, ‘Agreed, provided delivery is in July.’ The buyer responds, ‘Fine, but "
        "payment will be after inspection.’ The seller delivers in August and demands payment. Advise "
        "both parties. Consider: offer, counter-offer, acceptance, battle of forms, certainty of terms, "
        "breach and remedies."
    )
    formation_gold = pipeline.curated_regression_answer(formation_question)
    formation_body = server.Handler._without_reference_section(formation_gold)
    check(
        1188 <= len(formation_body.split()) <= 1212
        and not server.Handler._generic_answer_failures(formation_body, formation_question, 1200)
        and not server.Handler._complete_answer_failures(formation_body, formation_question)
        and not server.Handler._contract_accuracy_failures(formation_body, formation_question, "full answer"),
        "the exact 1,200-word failure regression now has a complete checked answer",
        results,
    )
    consideration_question = (
        "Suggested length: 1,000 words. The doctrine of consideration is an outdated technical "
        "requirement that English contract law should abandon. Critically discuss with reference to "
        "Williams v Roffey, Foakes v Beer and promissory estoppel."
    )
    consideration_gold = pipeline.curated_regression_answer(consideration_question)
    consideration_body = server.Handler._without_reference_section(consideration_gold)
    check(
        990 <= len(consideration_body.split()) <= 1010
        and not server.Handler._generic_answer_failures(
            consideration_body, consideration_question, 1000
        )
        and not server.Handler._complete_answer_failures(
            consideration_body, consideration_question
        )
        and not server.Handler._contract_accuracy_failures(
            consideration_body, consideration_question, "full answer"
        ),
        "consideration regression distinguishes Williams, Foakes and equitable estoppel within 1,000 words",
        results,
    )
    bad_consideration = (
        "### Introduction\nFoakes v Beer held that practical benefit validates a promise to pay more. "
        "Promissory estoppel received statutory recognition in the Misrepresentation Act 1967.\n"
        "### Conclusion\nThose propositions resolve the issue."
    )
    check(
        len(server.Handler._contract_accuracy_failures(
            bad_consideration, consideration_question, "full answer"
        )) >= 2,
        "consideration accuracy gate rejects the two hallucinations observed in live generation",
        results,
    )
    reviewed_outputs: list[tuple[str, str, str]] = []
    for slug, stem in GENERAL_ENQUIRIES:
        prompt = f"General legal enquiry. Subject: {slug.replace('_', ' ')}. {stem}"
        reviewed_outputs.append((slug, prompt, pipeline.curated_regression_answer(prompt)))
    for slug, stem in SQE_PROBES:
        prompt = f"Subject: {slug.replace('_', ' ')}. {stem}"
        reviewed_outputs.append((slug, prompt, pipeline.curated_regression_answer(prompt)))
    check(
        len(reviewed_outputs) == 35
        and all(answer.strip() for _slug, _prompt, answer in reviewed_outputs)
        and all(not server.Handler._subject_accuracy_failures(
            server.Handler._without_reference_section(answer), prompt, slug, "full answer"
        ) for slug, prompt, answer in reviewed_outputs)
        and all(not re.search(
            r"Z\d{6,8}|\[student\]|\.docx|/Users/|writing guidance",
            answer, re.I,
        ) for _slug, _prompt, answer in reviewed_outputs),
        "all 21 specialist enquiries and 14 SQE reviewed outputs pass accuracy and privacy gates",
        results,
    )
    _ledger, private_meta = pipeline.assemble_ledger(
        "contract misrepresentation problem", "england_wales", online_mode="off"
    )
    check(
        not private_meta["sources"] and private_meta["indexed"] > 0,
        "private upload, indexed-database and marked-work filenames never reach UI source chips",
        results,
    )

    sample = (
        "The clause is assessed by legitimate interest and proportionality "
        "(*Cavendish Square Holding BV v Talal El Makdessi* [2015] UKSC 67, [2016] AC 1172). "
        "Misrepresentation Act 1967 applies."
    )
    references = server.Handler._authorities_table(sample, contract)
    check("### References" in references and "Cavendish" in references,
          "used-authority-only References footer is generated", results)
    grouped_sample = (
        "The authorities remain distinct "
        "(*Williams v Roffey Bros & Nicholls (Contractors) Ltd* [1991] 1 QB 1; "
        "*Foakes v Beer* (1884) 9 App Cas 605). "
        "The bargain need not be adequate "
        "(*Currie v Misa* (1875) LR 10 Ex 153; *Thomas v Thomas* (1842) 2 QB 851). "
        "The statutory routes are distinct (Employment Rights Act 1996, ss 94 and 100). "
        "The limitation route is separate (Employment Rights Act 1996, s 111)."
    )
    grouped_references = server.Handler._authorities_table(grouped_sample, contract)
    check(
        grouped_references.count("Williams v Roffey") == 1
        and grouped_references.count("Foakes v Beer") == 1
        and grouped_references.count("Currie v Misa") == 1
        and grouped_references.count("Thomas v Thomas") == 1
        and grouped_references.count("Employment Rights Act 1996") == 1
        and "Employment Rights Act 1996, s" not in grouped_references
        and ";" not in grouped_references,
        "grouped cases split cleanly and legislation is deduplicated without section pinpoints",
        results,
    )
    general_question = "Explain proprietary estoppel in practical terms for a homeowner."
    sqe_question = "SQE single best answer: identify the correct proprietary-estoppel remedy."
    check(
        not pipeline.needs_reference_list(general_question)
        and not pipeline.needs_reference_list(sqe_question)
        and server.Handler._authorities_table(sample, general_question) == ""
        and server.Handler._authorities_table(sample, sqe_question) == ""
        and pipeline.needs_reference_list(general_question + " Include a reference list."),
        "general enquiries and SQE omit the final list unless expressly requested",
        results,
    )
    check(
        all(
            len(server.Handler._count_safe_analytical_padding(contract, count).split()) == count
            for count in range(1, 91)
        ),
        "small residual word-count gaps are filled exactly instead of rejecting the answer",
        results,
    )
    duplicate = "The same analytical sentence should appear only once because repetition adds no legal value."
    check(
        duplicate not in server.Handler._drop_existing_sentences(
            "### New issue\n" + duplicate + " A genuinely new application remains.",
            "### Existing issue\n" + duplicate,
        )
        and "genuinely new application" in server.Handler._drop_existing_sentences(
            duplicate + " A genuinely new application remains.", duplicate
        ),
        "focused count extensions remove prose already present in the answer",
        results,
    )
    repaired_citations = server.Handler._repair_inline_oscola(
        "Hyde v Wrench establishes the effect of a counter-offer. "
        "Butler Machine Tool v Ex-Cell-O governs a battle of forms.",
        formation_question,
        "contract_law",
    )
    check(
        "*Hyde v Wrench* (1840) 3 Beav 334" in repaired_citations
        and "*Butler Machine Tool Co Ltd v Ex-Cell-O Corporation (England) Ltd* [1979] 1 WLR 401"
        in repaired_citations
        and not server.Handler._uncited_authority_sentences(repaired_citations),
        "verified guide citations repair named-authority OSCOLA omissions without guessing",
        results,
    )
    statute_only_question = (
        "Assume England and Wales law. Problem question. Suggested length: 1,000 words. "
        "A defendant is charged under the Criminal Justice Act 1967."
    )
    repaired_statute = server.Handler._repair_inline_oscola(
        "Section 1(1) of the Criminal Justice Act 1967 applies.",
        statute_only_question,
        "criminal_law",
    )
    check(
        "(Criminal Justice Act 1967)" in repaired_statute
        and not server.Handler._uncited_authority_sentences(repaired_statute),
        "statute citations are repaired even when a subject has no case-bank entries",
        results,
    )
    check(
        all(
            guides.detect_subject(
                f"Assume England and Wales law. "
                f"{'Essay question' if register == 'essay' else 'Problem question'}. "
                f"Suggested length: {length_words:,} words. {stem} "
                "Default to full parenthetical OSCOLA."
            ) == expected_slug
            for length_words, expected_slug, register, stem in LENGTH_QUESTIONS
        ),
        "realistic fact-pattern questions route to the dominant subject across all 20 lengths",
        results,
    )
    restructured = server.Handler._collapse_duplicate_headings(
        "### Introduction\n\nOpening thesis.\n\n### Intoxication\n\nFirst analysis.\n\n"
        "### Conclusion\n\nInterim conclusion.\n\n### Introduction\n\nSecond opening.\n\n"
        "### Intoxication\n\nSecond analysis.\n\n### Conclusion\n\nFinal conclusion."
    )
    check(
        restructured.count("### Introduction") == 1
        and restructured.count("### Conclusion") == 1
        and restructured.count("### Intoxication") == 1
        and restructured.startswith("### Introduction")
        and "Final conclusion" in restructured.split("### Conclusion")[1]
        and "Second analysis" in restructured,
        "assembly keeps one Introduction, the final Conclusion and one copy of each repeated heading",
        results,
    )
    criminal_problem_question = (
        "Assume England and Wales law. Problem question. Suggested length: 3,000 words. "
        "Gus, drunk, throws a bottle which strikes Hana, who dies after a mismanaged operation. "
        "Advise on homicide liability, causation, intoxication, accessorial liability, "
        "loss of control and diminished responsibility. Default to full parenthetical OSCOLA."
    )
    repaired_criminal = server.Handler._repair_inline_oscola(
        "The leading case is R v Woollin, which requires foresight of virtual certainty. "
        "Voluntary intoxication is governed by DPP v Majewski.",
        criminal_problem_question,
        "criminal_law",
    )
    check(
        len(guides.authority_citation_map_for_question(criminal_problem_question, "criminal_law")) >= 20
        and "*R v Woollin* [1999] 1 AC 82 (HL)" in repaired_criminal
        and "*DPP v Majewski* [1977] AC 443 (HL)" in repaired_criminal
        and not server.Handler._uncited_authority_sentences(repaired_criminal),
        "criminal law carries a verified OSCOLA bank so named homicide authorities are repaired",
        results,
    )
    mia_style_question = (
        "Problem question. Suggested length: 1,000 words. Mia gets very drunk at a party and "
        "hurls a glass across the room; it hits Ben, who dies in hospital after a doctor "
        "misreads his scan. Mia's friend Zoe had been chanting 'throw it, throw it'. "
        "Advise Mia and Zoe."
    )
    weak_mia = (
        "### Introduction\nMia may be liable for manslaughter. Zoe may be an accomplice.\n"
        "### Conclusion\nLiability is likely."
    )
    strong_mia = (
        "### Introduction\nMia faces homicide charges. Intention may be inferred under "
        "Woollin's virtual certainty direction. Voluntary intoxication follows Majewski. "
        "Zoe's encouragement engages Jogee accessorial liability. Causation follows "
        "Cheshire medical-treatment principles.\n### Conclusion\nBoth may be liable."
    )
    check(
        "jogee" in " ".join(server.Handler._criminal_accuracy_failures(weak_mia, mia_style_question)).lower()
        and "woollin" in " ".join(server.Handler._criminal_accuracy_failures(weak_mia, mia_style_question)).lower()
        and "majewski" in " ".join(server.Handler._criminal_accuracy_failures(weak_mia, mia_style_question)).lower()
        and not server.Handler._criminal_accuracy_failures(strong_mia, mia_style_question),
        "criminal homicide answers must cover Woollin, Majewski and Jogee where facts engage them",
        results,
    )
    heading_repair = server.Handler._ensure_required_headings(
        "Opening analysis.\n\n### Formation\n\nApplied analysis.\n\nFinal advice.",
        formation_question,
    )
    trimmed_heading_repair = server.Handler._trim_to_words(
        heading_repair.replace("Applied analysis.", "Applied analysis. " * 80), 90
    )
    check(
        "### Introduction" in trimmed_heading_repair
        and "### Conclusion" in trimmed_heading_repair,
        "word-count trimming preserves required Introduction and Conclusion structure",
        results,
    )
    scrubbed = server.Handler._sanitize_final(
        "The rule follows Cavendish Square Holding BV v Makdessi [2099] UKSC 999 at para 88.",
        "Cavendish Square Holding BV v Makdessi",
    )
    check("2099" not in scrubbed and "para 88" not in scrubbed and "Cavendish" in scrubbed,
          "unverified neutral citations and pinpoints are removed", results)

    with tempfile.TemporaryDirectory() as tmp:
        temp = Path(tmp)
        old = (server.DB_PATH, server.PRIVATE_UPLOAD_ROOT, server.RECORD_ROOT)
        server.DB_PATH = temp / "chat.sqlite3"
        server.PRIVATE_UPLOAD_ROOT = temp / "private"
        server.RECORD_ROOT = temp / "records"
        try:
            server.init_db()
            memory = server.create_conversation("england_wales", "memory")
            private = server.create_conversation("england_wales", "private")
            incomplete = server.create_conversation("england_wales", "memory")
            current = server.create_conversation("england_wales", "memory")
            server.add_message(memory["id"], "user", "MEMORY_SENTINEL")
            server.add_message(memory["id"], "assistant", "MEMORY_REPLY")
            server.add_message(private["id"], "user", "PRIVATE_SENTINEL")
            server.add_message(incomplete["id"], "user", "UNANSWERED_SENTINEL")
            context = server.get_memory_context(current["id"], "remember MEMORY_SENTINEL")
            check(
                "MEMORY_SENTINEL" in context
                and "PRIVATE_SENTINEL" not in context
                and "UNANSWERED_SENTINEL" not in context,
                "cross-chat memory includes completed Memory chats and excludes Private/incomplete chats",
                results,
            )
            check("MEMORY_REPLY" not in context,
                  "cross-chat memory does not inject prior assistant legal prose", results)
            saved_feedback = server.save_feedback_record(
                memory["id"], "Feedback test question", "Feedback test answer",
                "Correct the legal test and verify the supporting authority.",
            )
            feedback_json = next(Path(saved_feedback).parent.glob("*.json"))
            feedback_payload = json.loads(feedback_json.read_text(encoding="utf-8"))
            private_feedback_blocked = False
            try:
                server.save_feedback_record(
                    private["id"], "Private question", "Private answer", "Must not be saved"
                )
            except PermissionError:
                private_feedback_blocked = True
            check(
                feedback_payload["question"] == "Feedback test question"
                and feedback_payload["user_feedback"].startswith("Correct the legal test")
                and private_feedback_blocked,
                "Memory feedback is saved to the configured folder while Private feedback is blocked",
                results,
            )
            stored = server.save_private_upload(
                private["id"], "secret.txt", base64.b64encode(b"private").decode()
            )
            server.add_attachment(private["id"], "secret.txt", "PRIVATE_FILE", stored_path=stored)
            check(server.hard_delete_private(private["id"]), "private chat hard-delete succeeds", results)
            check(not Path(stored).exists() and server.conversation_mode(private["id"]) is None,
                  "private chat and upload are permanently removed", results)
        finally:
            server.DB_PATH, server.PRIVATE_UPLOAD_ROOT, server.RECORD_ROOT = old

    # Public-mode storage and authorization are exercised without a live
    # Cloudflare account by provisioning two already-verified test identities.
    with tempfile.TemporaryDirectory() as tmp:
        temp = Path(tmp)
        old = (
            server.DB_PATH, server.PRIVATE_UPLOAD_ROOT, server.RECORD_ROOT,
            server.PUBLIC_MODE, server.REQUESTS_PER_HOUR, server.REQUESTS_PER_DAY,
            server.RETENTION_DAYS,
        )
        server.DB_PATH = temp / "public-chat.sqlite3"
        server.PRIVATE_UPLOAD_ROOT = temp / "uploads"
        server.RECORD_ROOT = temp / "feedback"
        server.PUBLIC_MODE = True
        server.REQUESTS_PER_HOUR = 1
        server.REQUESTS_PER_DAY = 2
        server.RETENTION_DAYS = 30
        try:
            server.init_db()
            alice = server.ensure_user("test:alice", "alice@example.test")
            bob = server.ensure_user("test:bob", "bob@example.test")
            alice_old = server.create_conversation(
                "england_wales", "memory", user_id=alice["id"]
            )
            alice_now = server.create_conversation(
                "england_wales", "memory", user_id=alice["id"]
            )
            bob_chat = server.create_conversation(
                "england_wales", "memory", user_id=bob["id"]
            )
            server.add_message(alice_old["id"], "user", "ALICE_MEMORY", user_id=alice["id"])
            server.add_message(alice_old["id"], "assistant", "ALICE_REPLY", user_id=alice["id"])
            server.add_message(bob_chat["id"], "user", "BOB_PRIVATE_DATA", user_id=bob["id"])
            server.add_message(bob_chat["id"], "assistant", "BOB_REPLY", user_id=bob["id"])
            alice_context = server.get_memory_context(
                alice_now["id"], "remember ALICE_MEMORY BOB_PRIVATE_DATA", user_id=alice["id"]
            )
            check(
                "ALICE_MEMORY" in alice_context
                and "BOB_PRIVATE_DATA" not in alice_context
                and server.get_messages(bob_chat["id"], alice["id"]) == []
                and server.conversation_mode(bob_chat["id"], alice["id"]) is None,
                "public users cannot read another user's chats or cross-chat memory",
                results,
            )

            no_consent = server.save_feedback_record(
                alice_old["id"], "Question", "Answer", "Needs a clearer test.",
                user_id=alice["id"], consent_training=False,
            )
            no_consent_files = list((server.RECORD_ROOT / alice["id"]).rglob("*.json"))
            consented = server.save_feedback_record(
                alice_old["id"], "Question 2", "Answer 2", "Use the correct authority.",
                user_id=alice["id"], consent_training=True,
            )
            with server.db() as conn:
                feedback_rows = conn.execute(
                    "SELECT id, consent_training FROM feedback WHERE user_id = ? ORDER BY created_at",
                    (alice["id"],),
                ).fetchall()
            check(
                no_consent == feedback_rows[0]["id"]
                and not no_consent_files
                and Path(consented).is_file()
                and Path(consented).with_suffix(".json") in feedback_paths(server.RECORD_ROOT)
                and [row["consent_training"] for row in feedback_rows] == [0, 1],
                "public feedback is structured and enters training files only after explicit consent",
                results,
            )

            stored = server.save_private_upload(
                alice_old["id"], "user-file.txt",
                base64.b64encode(b"owned-by-alice").decode(), user_id=alice["id"],
            )
            server.add_attachment(
                alice_old["id"], "user-file.txt", "ALICE_FILE", stored_path=stored,
                byte_size=14, user_id=alice["id"],
            )
            blocked_attachment = False
            try:
                server.add_attachment(
                    alice_old["id"], "intrusion.txt", "NO", user_id=bob["id"]
                )
            except PermissionError:
                blocked_attachment = True
            check(
                blocked_attachment
                and server.get_attachments(alice_old["id"], bob["id"]) == []
                and server.user_storage_bytes(alice["id"]) == 14,
                "public uploads and storage quotas are scoped to their owner",
                results,
            )

            server.consume_generation_quota(alice["id"])
            rate_limited = False
            try:
                server.consume_generation_quota(alice["id"])
            except server.QuotaError:
                rate_limited = True
            check(rate_limited, "public generation rate limits persist in SQLite", results)

            expired = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
            with server.db() as conn:
                conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    (expired, bob_chat["id"]),
                )
            check(
                server.purge_expired_public_data() == 1
                and server.conversation_mode(bob_chat["id"], bob["id"]) is None,
                "public retention cleanup permanently removes expired conversations",
                results,
            )

            account_before = server.account_summary(alice)
            server.delete_user_account(alice["id"])
            with server.db() as conn:
                remains = {
                    "user": conn.execute("SELECT COUNT(*) n FROM users WHERE id = ?", (alice["id"],)).fetchone()["n"],
                    "chat": conn.execute("SELECT COUNT(*) n FROM conversations WHERE user_id = ?", (alice["id"],)).fetchone()["n"],
                    "feedback": conn.execute("SELECT COUNT(*) n FROM feedback WHERE user_id = ?", (alice["id"],)).fetchone()["n"],
                }
            check(
                account_before["conversations"] == 2
                and not any(remains.values())
                and not Path(stored).exists(),
                "public account deletion removes identity, chats, feedback and uploads",
                results,
            )

            import jwt
            from cryptography.hazmat.primitives.asymmetric import rsa

            private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            public_key = private_key.public_key()

            class TestJwksClient:
                def get_signing_key_from_jwt(self, _token):
                    return type("SigningKey", (), {"key": public_key})()

            old_auth = (
                server.CF_ACCESS_TEAM_DOMAIN, server.CF_ACCESS_AUD, server._JWKS_CLIENT,
                {name: os.environ.get(name) for name in (
                    "LEGAL_CHAT_DB", "LEGAL_FEEDBACK_ROOT", "LEGAL_PRIVATE_UPLOAD_ROOT",
                    "LEGAL_RAG_DB", "LEGAL_GUIDANCE_DB",
                )},
            )
            server.CF_ACCESS_TEAM_DOMAIN = "https://test.cloudflareaccess.com"
            server.CF_ACCESS_AUD = "test-audience"
            server._JWKS_CLIENT = TestJwksClient()
            os.environ["LEGAL_CHAT_DB"] = str(server.DB_PATH)
            os.environ["LEGAL_FEEDBACK_ROOT"] = str(server.RECORD_ROOT)
            os.environ["LEGAL_PRIVATE_UPLOAD_ROOT"] = str(server.PRIVATE_UPLOAD_ROOT)
            os.environ["LEGAL_RAG_DB"] = str(temp / "public-rag.sqlite3")
            os.environ["LEGAL_GUIDANCE_DB"] = str(temp / "public-guidance.sqlite3")
            issued = datetime.now(timezone.utc)
            payload = {
                "sub": "verified-user", "email": "verified@example.test", "type": "app",
                "iss": server.CF_ACCESS_TEAM_DOMAIN, "aud": [server.CF_ACCESS_AUD],
                "iat": issued, "exp": issued + timedelta(minutes=5),
            }
            valid_token = jwt.encode(payload, private_key, algorithm="RS256")
            valid_claims = server.decode_access_jwt(valid_token)
            wrong_audience = dict(payload, aud=["wrong-audience"])
            bad_token = jwt.encode(wrong_audience, private_key, algorithm="RS256")
            bad_rejected = False
            try:
                server.decode_access_jwt(bad_token)
            except server.AuthenticationError:
                bad_rejected = True
            check(
                valid_claims["sub"] == "verified-user" and bad_rejected,
                "public origin validates the Cloudflare Access JWT signature, issuer and audience",
                results,
            )
            safe_public_rag = os.environ["LEGAL_RAG_DB"]
            os.environ["LEGAL_RAG_DB"] = str(ROOT / "model_database" / "snapshot" / "chroma.sqlite3")
            private_rag_rejected = False
            try:
                server.validate_public_config()
            except RuntimeError:
                private_rag_rejected = True
            os.environ["LEGAL_RAG_DB"] = safe_public_rag
            check(
                private_rag_rejected,
                "public configuration refuses the repository's private RAG database",
                results,
            )
            public_launcher = (ROOT / "scripts" / "public_chat_ui.sh").read_text(encoding="utf-8")
            check(
                'LEGAL_RAG_DB="${LEGAL_RAG_DB:-$DATA_DIR/public-rag.sqlite3}"' in public_launcher
                and 'LEGAL_GUIDANCE_DB="${LEGAL_GUIDANCE_DB:-$DATA_DIR/public-guidance.sqlite3}"'
                in public_launcher
                and "model_database" not in public_launcher,
                "public launcher cannot inherit the private local RAG database",
                results,
            )
            macos_installer = (
                ROOT / "scripts" / "configure_public_macos.py"
            ).read_text(encoding="utf-8")
            check(
                'ORIGIN_LABEL = "ai.legalchatmodel.origin"' in macos_installer
                and 'TUNNEL_LABEL = "ai.legalchatmodel.tunnel"' in macos_installer
                and '"--token-file"' in macos_installer
                and "_atomic_write(token_path, (token + \"\\n\").encode(), 0o600)"
                in macos_installer
                and "public.env" in public_launcher,
                "persistent macOS public services keep the tunnel token outside Git with mode 0600",
                results,
            )
            server.CF_ACCESS_TEAM_DOMAIN, server.CF_ACCESS_AUD, server._JWKS_CLIENT, old_env = old_auth
            for name, value in old_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        finally:
            (
                server.DB_PATH, server.PRIVATE_UPLOAD_ROOT, server.RECORD_ROOT,
                server.PUBLIC_MODE, server.REQUESTS_PER_HOUR, server.REQUESTS_PER_DAY,
                server.RETENTION_DAYS,
            ) = old

    report = {"passed": all(result["passed"] for result in results), "checks": results}
    out = ROOT / "training" / "LEGAL_APP_VERIFICATION.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
