"""
Reusable backend quality controls for legal answer generation.

This file stores privacy-safe rules derived from local course/feedback
materials and operational QA requirements. It deliberately contains no private
source paths, filenames, marker wording, or user-specific facts.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


NO_SYLLABUS_LIMIT_MARKERS: tuple[str, ...] = (
    "no syllabus limit",
    "no limitation",
    "no limit",
    "broad-all",
    "broad all",
    "broad research",
    "outside syllabus",
    "not limited to syllabus",
)

COURSE_BOUND_MARKERS: tuple[str, ...] = (
    "course-bound",
    "course bound",
    "module syllabus",
    "stay within the module",
    "stay within syllabus",
    "within the syllabus",
    "only in syllabus",
)

FRESHNESS_TRIGGERS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "law_medicine",
        ("assisted dying", "abortion reform", "hfea", "organ allocation", "capacity code", "deemed consent"),
        "Law and Medicine current-law points can change. Verify current statutes, bills, official guidance, and appellate status before stating the rule.",
    ),
    (
        "competition_law",
        ("digital markets", "dma", "dmcc", "article 102", "self-preferencing", "platform", "gatekeeper"),
        "Competition and digital-market regulation is current-law sensitive. Check official CMA/EU materials, legislation commencement, and recent judgments/decisions where available.",
    ),
    (
        "biolaw_ai_data",
        ("ai", "artificial intelligence", "data protection", "medical device", "algorithm", "cybersecurity", "gdpr"),
        "AI/data/medical-device law is current-law sensitive. Verify statutory status, regulator guidance, and current enforcement position before relying on it.",
    ),
    (
        "tax_law",
        ("tax", "hmrc", "vat", "capital gains", "inheritance tax", "corporation tax", "income tax", "threshold", "rate"),
        "Tax rates, thresholds, reliefs, penalties and HMRC powers are current-law sensitive. Verify current legislation, HMRC guidance and tribunal/appellate authority before stating figures or deadlines.",
    ),
    (
        "intellectual_property_law",
        ("ai copyright", "text and data mining", "trade mark reform", "patent ai", "digital copyright", "online safety"),
        "IP and AI/copyright points can change through legislation, regulator guidance and appellate authority. Verify current UK/EU status before stating reform-sensitive rules.",
    ),
    (
        "eu_law",
        ("retained eu law", "assimilated law", "post-brexit", "withdrawal agreement", "windsor framework", "charter"),
        "EU/UK post-Brexit status is current-law sensitive. Verify whether the issue is EU law, assimilated/retained EU law, Withdrawal Agreement/Windsor Framework, or domestic replacement.",
    ),
    (
        "employment_law",
        ("employment rights bill", "worker status", "unfair dismissal", "flexible working", "redundancy", "whistleblowing"),
        "Employment law reform and statutory thresholds can change. Verify current legislation, commencement, ACAS/government guidance, and leading appellate authority.",
    ),
    (
        "sqe2",
        ("sqe2", "sqe 2", "kaplan", "sra", "assessment specification", "written skills"),
        "SQE2 assessment rules are specification-sensitive. Verify official SRA/Kaplan structure and criteria where the user asks for current assessment detail.",
    ),
)

GENERAL_CURRENT_LAW_MARKERS: tuple[str, ...] = (
    "latest",
    "current",
    "recent",
    "new law",
    "new case",
    "updated",
    "2025",
    "2026",
)

SOURCE_QUALITY_LABELS: tuple[tuple[str, str], ...] = (
    ("official_primary", "legislation, rules, official regulator material, court judgments, treaty text, or official assessment specification"),
    ("course_material", "module handbook, lecture/tutorial material, seminar worksheet, or official course guidance"),
    ("feedback_marking", "formative/summative feedback, performance indicators, sample answer commentary, or marking criteria"),
    ("secondary_commentary", "textbook, journal article, practitioner commentary, or explanatory note"),
    ("weak_or_noise", "duplicate file, filename-only hit, uncited web snippet, outdated note, irrelevant subject, or source with insufficient metadata"),
)

ANTI_GENERIC_QUALITY_RULES: tuple[str, ...] = (
    "apply to facts before moving to the next issue",
    "rank likely/arguable/weak outcomes where facts permit",
    "state remedy/next step before ending",
    "do not survey the topic when the user asked for advice or a focused essay",
    "do not use vague policy unless tied to doctrine, authority, evidence, or the statutory test",
    "run a specialist accuracy pass for exact statutory gateway, procedural requirement, remedy route, and current-law status",
)

SUBJECT_TEMPLATE_BY_SLUG: Dict[str, str] = {
    "law_medicine": "\n".join([
        "[SUBJECT TEMPLATE — LAW AND MEDICINE]",
        "- Course-bound essay: Part I thesis + statutory route -> 2-3 focused syllabus examples -> ethical/legal critique -> reform/no-reform conclusion.",
        "- No-limit essay: use the same structure, but label wider material and verify current-law status before relying on it.",
        "- Problem answer: patient/person -> capacity/decision-maker -> statutory route -> authority -> lawful treatment/court outcome -> ethics/practical step.",
        "- Do not let autonomy language replace the statutory test; use autonomy to evaluate the test after applying it.",
        "- In every substantive example, name the governing route where it exists, e.g. Mental Capacity Act 2005, Human Tissue Act 2004, Abortion Act 1967, or Human Fertilisation and Embryology Act 1990.",
    ]),
    "contract_law": "\n".join([
        "[SUBJECT TEMPLATE — CONTRACT LAW]",
        "- Problem: formation/status -> term/representation -> breach/vitiating factor -> exclusion/unfairness -> causation/remoteness/mitigation -> remedy.",
        "- Essay: define the proposition and defend a thesis about certainty, autonomy, reliance, fairness, risk allocation, or remedial coherence.",
        "- Term/representation discipline: classify the statement by timing, expertise, importance, reduction into writing and reliance before remedy.",
        "- Exclusion/non-reliance discipline: incorporation -> construction -> UCTA/CRA or misrepresentation control -> effect on remedy; do not jump straight to fairness.",
        "- Damages discipline: expectation/reliance election, remoteness, mitigation, causation, agreed damages/penalty and equitable remedy must be separated.",
    ]),
    "competition_law": "\n".join([
        "[SUBJECT TEMPLATE — COMPETITION LAW]",
        "- Article 102 problem: undertaking -> market definition -> dominance -> abuse theory -> foreclosure/exploitation evidence -> effect -> objective justification/efficiencies -> enforcement/remedy.",
        "- Article 101 problem: undertaking/agreement -> object/effect -> appreciability -> exemption -> enforcement/remedy.",
        "- Essay: state the competition-law goal, then test whether doctrine/economics/regulation serves it without becoming a policy-only answer.",
        "- Define `effects-based` before using it: distinguish actual effects, likely effects, capability to foreclose, presumptions of harm, as-efficient-competitor analysis, consumer welfare, and protection of the competitive process.",
        "- Article 102 digital/platform sequence: dominance and special responsibility -> precise abuse label -> economic mechanism (network effects, data, defaults, ranking, traffic diversion, scale/scope) -> foreclosure of efficient rivals -> objective necessity/efficiencies -> remedy.",
        "- Smart-device/voice-assistant sequence: undertaking -> product/ecosystem channel -> compulsory pre-installation/defaults/interoperability -> tying or technological tying before weaker self-preferencing labels -> exploitative data/risk terms -> security justification.",
        "- Google Shopping discipline: self-preferencing is not per se abusive; analyse leveraging from general search into comparison shopping, unequal ranking/display, traffic dependence, potential foreclosure, and why Bronner indispensability is not automatically the governing test for non-refusal platform design cases.",
        "- Digital Markets Act discipline: label the DMA as ex ante gatekeeper regulation and Article 102 as ex post enforcement; explain overlap without treating one as replacing the other.",
        "- Current Article 102 policy discipline: the 2009 Guidance remains relevant until replaced, but answers on current digital/exclusionary abuse must mention and verify the Commission's first-Guidelines process where relevant.",
    ]),
    "pensions_law": "\n".join([
        "[SUBJECT TEMPLATE — PENSIONS LAW]",
        "- Problem: scheme type -> deed/rules -> dates/member status -> power/process -> statutory overlay -> assumptions/calculations -> remedy.",
        "- Equalisation/amendment: Barber timing -> scheme amendment route -> statutory protection -> actuarial/commutation issue -> practical consequence.",
        "- Non-financial investment: retirement-benefit purpose/Pensions Act 2004 s 255 -> financially material vs non-financial motive -> member consensus across all beneficiary classes -> risk/quantum of detriment -> DB/DC/default-fund split -> trustee process and operational advice.",
    ]),
    "commercial_law": "\n".join([
        "[SUBJECT TEMPLATE — COMMERCIAL LAW]",
        "- Problem: transaction type -> contract timeline -> title/property -> risk/delivery/acceptance -> breach/default -> insolvency/priority -> remedy.",
        "- Sale of goods: implied term, description/quality/fitness/title, rejection/acceptance, damages and retention of title must be separated.",
        "- Nemo dat/title: original title holder -> transfer chain -> statutory/common-law exception -> good faith/possession/title document -> final proprietary and personal remedies.",
        "- ROT/security discipline: clause construction, identifiable goods, mixed/manufactured goods, proceeds, registration/security characterisation and insolvency effect.",
    ]),
    "employment_law": "\n".join([
        "[SUBJECT TEMPLATE — EMPLOYMENT LAW]",
        "- Problem: claimant status -> qualifying period/time limit -> statutory route -> employer defence/process -> causation/remedy adjustment -> practical order.",
        "- Status discipline: employee, worker, applicant, former employee and self-employed contractor have different gateways and remedies.",
        "- Discrimination discipline: classify claim -> comparator or PCP -> burden -> justification/defence -> employer liability -> remedies.",
        "- Dismissal discipline: reason, investigation, procedure, range of reasonable responses, Polkey/contributory fault and reinstatement/compensation must be separated.",
    ]),
    "family_law": "\n".join([
        "[SUBJECT TEMPLATE — FAMILY LAW]",
        "- Problem: dispute type/order sought -> statutory gateway -> welfare/needs/threshold factors -> proportionality/safeguards -> specific order.",
        "- Children: identify the order first, then welfare paramountcy, checklist, risk evidence, contact safeguards and practical drafting.",
        "- Finance: assets/resources -> needs -> sharing/compensation -> departure factors -> nuptial agreement/conduct only where legally material -> final order.",
        "- Public-law children: threshold is separate from welfare and proportionality; do not jump to care/adoption outcome before proving threshold.",
    ]),
    "criminal_law": "\n".join([
        "[SUBJECT TEMPLATE — CRIMINAL LAW]",
        "- Problem: go straight to offence-by-offence analysis: actus reus -> mens rea -> causation -> defence -> likely charge/outcome.",
        "- Essay: identify the fault line, then evaluate culpability, fair labelling, autonomy, harm, prevention, or criminalisation limits.",
    ]),
    "land_law": "\n".join([
        "[SUBJECT TEMPLATE — LAND LAW]",
        "- Problem: registered/unregistered land -> right asserted -> creation/formality -> protection/registration -> priority -> remedy.",
        "- Essay: target the quotation/proposition and evaluate certainty, fairness, registration, numerus clausus, or family-home protection.",
        "- Registered-land problem final pass: ask whether any lease has expired or is about to expire; if business occupation is present, consider Landlord and Tenant Act 1954 Part II continuation/security unless validly contracted out.",
        "- Co-ownership priority final pass: purchase-price contribution may support a resulting trust as well as evidence for common-intention constructive trust; keep acquisition, quantification, priority, and sale proceeds separate.",
        "- Lender priority final pass: if an overriding beneficial interest binds a later registered charge, explain whether the charge bites only on the registered proprietor's beneficial share and how overreaching or sale proceeds change the practical outcome.",
        "- Easement/estoppel final pass: separate right of way from parking; test whether parking leaves reasonable use; visibility/use is not the same as protection, but consider and reject any actual-occupation argument expressly.",
        "- Option/notice final pass: estate contracts should be protected by notice; if late protection is attempted, discuss official search with priority and interim relief before completion.",
        "- Overreaching final pass: appointment and payment to two trustees must be valid; overreaching clears beneficial interests under a trust only, not leases, easement-type claims, or options.",
    ]),
    "tort_law": "\n".join([
        "[SUBJECT TEMPLATE — TORT LAW]",
        "- Negligence problem: duty category -> breach -> factual causation -> scope/remoteness -> defences -> damages.",
        "- Defamation/privacy: protected interest -> threshold -> defence/balancing -> remedy.",
    ]),
    "business_law": "\n".join([
        "[SUBJECT TEMPLATE — BUSINESS / COMPANY LAW]",
        "- Problem: vehicle/status -> actor authority -> duty/breach -> approval/filing/ratification -> enforcement route -> remedy/practical risk.",
        "- Directors/minorities: separate company loss, personal prejudice, derivative claim, unfair prejudice, ratification, and insolvency/creditor-interest routes.",
    ]),
    "intellectual_property_law": "\n".join([
        "[SUBJECT TEMPLATE — INTELLECTUAL PROPERTY LAW]",
        "- Problem: identify the right first -> subsistence/validity -> ownership -> infringement/restricted act -> defence/exception -> remedy.",
        "- Copyright: work category, originality, authorship/ownership, restricted act, substantial part, fair dealing/licence and remedy must be visible.",
        "- Trade mark/passing off: sign/use/goods or services -> origin/reputation function -> confusion/unfair advantage/due cause -> defence/remedy.",
        "- Patent: claim construction, patentability/validity, infringement, insufficiency/obviousness challenge and remedy; keep UK/EU/US positions separate.",
    ]),
    "tax_law": "\n".join([
        "[SUBJECT TEMPLATE — TAX LAW]",
        "- Problem: taxpayer -> chargeable event -> charging provision -> timing/computation -> relief/exemption -> anti-avoidance -> compliance/remedy.",
        "- Current figures discipline: do not invent rates, thresholds, dates or penalties; verify them or state assumptions and legal effect without figures.",
        "- Avoidance discipline: characterise the tax advantage -> purposive construction/Ramsay -> transfer pricing or other TAAR/specific anti-avoidance/profit-diversion route -> GAAR -> disclosure/penalties/settlement/remedy.",
        "- IP/offshore discipline: identify DEMPE functions, control of risk, funding, real decision-makers/personnel, arm's-length valuation/royalty, and whether intermediaries perform real functions before invoking GAAR.",
        "- GAAR discipline: GAAR is a high-threshold abusive-arrangements route, not a catch-all for ordinary avoidance or a substitute for transfer pricing.",
        "- Treaty/admin discipline: apply domestic charge first, then treaty allocation or HMRC power, statutory preconditions, time limit, appeal forum and remedy.",
    ]),
    "environmental_law": "\n".join([
        "[SUBJECT TEMPLATE — ENVIRONMENTAL LAW / NUISANCE]",
        "- Nuisance problem: standing/proprietary interest -> private nuisance for amenity harm -> negligence for personal injury/causation -> statutory nuisance -> regulator/JR route -> ranked remedies.",
        "- Permit discipline: environmental permits and substantial compliance are relevant but not conclusive; do not treat them as a tort defence unless the statutory scheme clearly excludes common-law rights.",
        "- Statutory nuisance discipline: local-authority complaint/investigation -> abatement notice -> appeal or best-practicable-means point where relevant -> resident magistrates' court route if the authority fails to act.",
        "- Regulatory/JR discipline: courts review legal error, irrationality, failure to consider material evidence, failure to follow policy, or inadequate investigation; they do not simply remake technical enforcement decisions.",
    ]),
    "succession_wills": "\n".join([
        "[SUBJECT TEMPLATE — WILLS / SUCCESSION]",
        "- Probate validity problem: formal validity/due execution -> testamentary capacity -> knowledge and approval -> undue influence/fraud -> effect of invalidity -> any Inheritance Act 1975 claim.",
        "- Burden discipline: propounder proves due execution and capacity; real doubt or suspicious circumstances require affirmative proof on the relevant issue, especially knowledge and approval.",
        "- Capacity discipline: Banks v Goodfellow is time-specific; good-and-bad-day evidence requires analysis of lucid interval at execution, not a generic dementia conclusion.",
        "- Suspicion discipline: carer involvement, dramatic departure from an earlier will, and major beneficiary participation mainly sharpen knowledge and approval; they do not prove probate undue influence without coercion.",
        "- 1975 Act discipline: identify applicant category and maintenance-only standard for non-spouses before discussing adult-child or carer fairness.",
    ]),
    "trusts_law": "\n".join([
        "[SUBJECT TEMPLATE — EQUITY / TRUSTS]",
        "- Problem: property/asset-by-asset -> creation/formality/constitution -> duty/breach -> tracing/priority -> personal/proprietary remedy.",
        "- Essay: defend a thesis about certainty, conscience, loyalty, proprietary remedies, charitable purpose, or commercial flexibility.",
    ]),
    "evidence_law": "\n".join([
        "[SUBJECT TEMPLATE — EVIDENCE LAW]",
        "- Problem: classify evidence -> statutory/common-law gateway -> exclusion/fairness discretion -> weight/direction -> likely ruling.",
        "- Essay: test truth-finding against reliability, Article 6 fairness, jury protection, managerial discretion, and party autonomy.",
        "- Identification final pass: after Turnbull, check PACE Code D procedure, pre-identification contamination, social-media exposure, video-ID compliance, exclusion/withdrawal, and jury warning.",
        "- Expert-evidence final pass: require transparent methodology, qualifications, limitations, non-overstatement, and no usurpation of the jury's identification function.",
        "- Confession final pass: separate PACE s 76(2)(a) oppression from s 76(2)(b) unreliability; threats/inducements often fit unreliability more clearly than narrow oppression, with s 78 as fallback fairness control.",
        "- Silence final pass: for CJPOA 1994 s 34, identify the later fact relied on, whether the accused could reasonably have mentioned it when questioned, and whether the adverse inference is fair.",
        "- Hearsay final pass: for absent accomplice/co-suspect material, stress motive to shift blame, lack of cross-examination, centrality to the case, and Article 6 fairness safeguards.",
        "- Bad-character final pass: first classify defendant/co-defendant/non-defendant status, then choose gateway; old/dissimilar convictions may be weak propensity evidence despite superficial similarity.",
    ]),
    "public_law": "\n".join([
        "[SUBJECT TEMPLATE — PUBLIC LAW / JUDICIAL REVIEW]",
        "- Problem: power source -> amenability/standing/time -> ground of review -> intensity -> remedy/discretion.",
        "- Essay: defend a thesis about legality, accountability, separation of powers, rights protection, deference, or institutional competence.",
    ]),
    "eu_law": "\n".join([
        "[SUBJECT TEMPLATE — EU LAW]",
        "- Problem: instrument/status -> scope -> direct effect or enforcement route -> justification/proportionality -> remedy.",
        "- Internal market: freedom -> measure/restriction -> discrimination/market access -> justification -> suitability/necessity/balance.",
        "- Remedies: direct effect, indirect effect, state liability, preliminary reference and supremacy must be route-specific, not generic.",
        "- UK-linked questions: label EU law, assimilated/retained EU law, Withdrawal Agreement/Windsor Framework or domestic replacement before applying doctrine.",
    ]),
    "private_international_law": "\n".join([
        "[SUBJECT TEMPLATE — PRIVATE INTERNATIONAL LAW]",
        "- Problem: jurisdiction/service -> stay/forum -> applicable law -> interim relief -> recognition/enforcement.",
        "- Keep merits, forum, governing law, and enforcement separate unless the question expressly merges them.",
    ]),
    "public_international_law": "\n".join([
        "[SUBJECT TEMPLATE — PUBLIC INTERNATIONAL LAW]",
        "- Problem: source -> jurisdiction -> attribution -> breach -> circumstances precluding wrongfulness/immunity -> consequences/remedies/enforcement.",
        "- State responsibility: attribution, breach, invocation, reparation and countermeasures must be separate; do not use jus cogens as a shortcut.",
        "- Use of force: Article 2(4), Security Council/self-defence, necessity, proportionality, imminence/evidence and collective-security limits.",
        "- IHL: conflict classification, status, targeting, proportionality, precautions, command/criminal responsibility and enforcement route.",
    ]),
    "mediation_law": "\n".join([
        "[SUBJECT TEMPLATE — MEDIATION]",
        "- Problem: agreement to mediate -> mediation process/confidentiality -> mediator conduct -> settlement route -> enforcement/refusal ground.",
        "- Essay: separate process value from legal enforceability and from mandatory-ADR/access-to-court concerns.",
    ]),
    "biolaw_ai_data": "\n".join([
        "[SUBJECT TEMPLATE — BIOLAW / AI / DATA]",
        "- Essay/problem: technology -> actor -> affected interest -> legal regime -> safety/consent/equality/accountability gap -> governance response.",
        "- Verify current-law status for AI, data protection, medical-device, and regulator guidance claims.",
        "- Dissertation-scale AI/data work: distinguish public availability from contextual reuse; define theoretical labels; label source status as enacted law, bill, guidance, settlement, litigation, academic claim or proposal.",
    ]),
}

TOPIC_MARKING_RUBRIC_BY_SLUG: Dict[str, str] = {
    "law_medicine": "\n".join([
        "[TOPIC-SPECIFIC MARKING RUBRIC — LAW AND MEDICINE]",
        "- First-class/top-band script: exact statutory route before ethics, two or three examples in depth, course/no-limit scope labelled, and a definite reform or no-reform thesis.",
        "- Penalise: autonomy rhetoric without the governing legal test; course-bound drift into excluded medical-law areas; copying model phrasing instead of adapting structure.",
        "- Niche traps: advance decision validity/applicability, DBD/DCD and living/deceased transplantation distinctions, AA 1967 s.1(1)(a) vs s.1(1)(d), HFEA Schedule 3 consent and s.13(5).",
    ]),
    "contract_law": "\n".join([
        "[TOPIC-SPECIFIC MARKING RUBRIC — CONTRACT LAW]",
        "- First-class/top-band script: follows the contractual chronology, classifies the legal route before remedy, and explains why the chosen remedy follows from breach/vitiation.",
        "- Penalise: fairness rhetoric without doctrine, remedy before liability, misrepresentation and contractual term merged, consumer/business controls treated as interchangeable.",
        "- Niche traps: entire-agreement/non-reliance clauses, bars to rescission, innominate-term termination, penalty/agreed damages, frustration limits and remoteness/scope of duty in damages.",
    ]),
    "competition_law": "\n".join([
        "[TOPIC-SPECIFIC MARKING RUBRIC — COMPETITION LAW]",
        "- First-class/top-band script: market definition serves the theory of harm; dominance and abuse are separate; economic evidence, counterfactual, effects, justification, and remedy are all visible.",
        "- Penalise: policy-only digital-market discussion; self-preferencing labels without foreclosure mechanism; excessive-pricing claims without costs, margins, comparators, and unfairness analysis.",
        "- Niche traps: ranking/default placement/interoperability, data-access refusals, margin squeeze, tying vs bundling, objective necessity vs efficiency justification.",
        "- Smart Robo-style trap: classify compulsory pre-installation/technological tying and default architecture before self-preferencing; treat hacking/security facts as evidence for quality, unfair terms or justification, not as automatic abuse.",
        "- Article 102 essays must define `effects-based` and distinguish actual effects, likely effects, capability, presumptions, AEC analysis, consumer welfare and competitive-process protection.",
        "- Google Shopping / Bronner trap: do not treat every self-preferencing case as a refusal-to-supply/essential-facilities case; explain why Bronner is central only where compulsory access is truly sought.",
        "- Current-law trap: mention the draft/forthcoming Article 102 exclusionary-abuse Guidelines process for current answers and avoid presenting the 2009 Guidance as the final modern framework.",
    ]),
    "land_law": "\n".join([
        "[TOPIC-SPECIFIC MARKING RUBRIC — LAND LAW]",
        "- First-class/top-band script: start with registered/unregistered status, identify every proprietary right separately, run creation/protection/priority/remedy in order, and use facts to choose likely outcomes.",
        "- Penalise: applying unregistered principles to registered land; treating actual occupation and overreaching as the same; ignoring Schedule 3 para 2 exceptions or options/easements separately.",
        "- Niche traps: LRA 2002 s.27 registrable dispositions, s.29 priority, notices/restrictions, overreaching under LPA 1925 ss.2 and 27, proprietary estoppel remedy proportionality, TOLATA outcome.",
        "- Lease-expiry trap: where a fixed-term business lease was granted years ago, check expiry and Landlord and Tenant Act 1954 Part II security before assuming either continuing legal lease or vacant possession.",
        "- Beneficial-interest trap: purchase-money contributions raise resulting-trust analysis; actual occupation and lender priority must then be separated from quantification and overreaching.",
        "- Completion-priority trap: options/notices and late protection require official-search-with-priority analysis where completion is pending.",
    ]),
    "commercial_law": "\n".join([
        "[TOPIC-SPECIFIC MARKING RUBRIC — COMMERCIAL LAW]",
        "- First-class/top-band script: builds a title/risk chronology, separates proprietary from contractual claims, and states insolvency priority consequences.",
        "- Penalise: title conclusions without timeline, breach-of-quality analysis where title is the real issue, ROT clauses assumed effective without construction/security analysis.",
        "- Niche traps: nemo dat exceptions, retention of title over mixed goods/proceeds, rejection versus acceptance, documentary sales, undisclosed agency and insolvency administrator/liquidator consequences.",
    ]),
    "employment_law": "\n".join([
        "[TOPIC-SPECIFIC MARKING RUBRIC — EMPLOYMENT LAW]",
        "- First-class/top-band script: starts with status and time limit, selects the statutory route, applies the burden/defence, and finishes with remedy adjustments.",
        "- Penalise: contract labels treated as decisive, every employment right surveyed, discrimination without comparator/PCP/burden/justification, dismissal without process and range of responses.",
        "- Niche traps: worker status after platform-control cases, automatic unfair dismissal, pregnancy/redundancy priority, reasonable adjustments, Polkey/contribution, severance of restrictive covenants.",
    ]),
    "family_law": "\n".join([
        "[TOPIC-SPECIFIC MARKING RUBRIC — FAMILY LAW]",
        "- First-class/top-band script: identifies the order sought, applies the statutory checklist/factors, gives safeguards, and states the precise order/remedy.",
        "- Penalise: generic welfare/fairness, adult-rights analysis displacing child welfare, financial remedy without asset schedule, public child answer without threshold.",
        "- Niche traps: relocation/contact safeguards, domestic-abuse protective order enforcement, nuptial agreement weight, conduct threshold in finance, adoption/proportionality and parental responsibility.",
    ]),
    "tort_law": "\n".join([
        "[TOPIC-SPECIFIC MARKING RUBRIC — TORT LAW]",
        "- First-class/top-band script: identify the tort and protected interest first, then run the correct specialist test rather than generic negligence; keep duty, breach, causation, scope, remoteness, defences, and damages distinct.",
        "- Penalise: Caparo as a universal formula after Robinson; foreseeability-only psychiatric harm; public-authority duty without acts/omissions and operational/policy distinction; defamation without serious-harm analysis.",
        "- Niche traps: omissions/third-party harm exceptions, primary/secondary psychiatric victims and Alcock controls, Barclays/Morrison limits on vicarious liability, nuisance/Rylands protected interest, DA 2013 serious harm.",
    ]),
    "business_law": "\n".join([
        "[TOPIC-SPECIFIC MARKING RUBRIC — BUSINESS / COMPANY LAW]",
        "- First-class/top-band script: separate company, director, shareholder, creditor and office-holder routes; identify approvals and filings; state who can sue and what remedy follows.",
        "- Penalise: moralised director blame without statutory duty/remedy; treating majority control as curing conflicts; using veil piercing where agency, trust, tort or statute is the real route.",
        "- Niche traps: Model Articles quorum/conflicts, CA 2006 ss.171-177 and ss.190-196, ratification limits, derivative permission, unfair prejudice, creditor-interest duty and insolvency voidable transactions.",
    ]),
    "intellectual_property_law": "\n".join([
        "[TOPIC-SPECIFIC MARKING RUBRIC — INTELLECTUAL PROPERTY LAW]",
        "- First-class/top-band script: identifies the IP right and jurisdiction, keeps subsistence/validity separate from infringement, and only then analyses defences and remedies.",
        "- Penalise: broad AI/copyright commentary without restricted act, trade-mark confusion without sign/goods/use analysis, patent conclusion without claim construction.",
        "- Niche traps: substantial-part qualitative copying, fair dealing purpose/fairness, communication to the public, trade mark functions/reputation/due cause, obviousness/sufficiency and AI authorship/inventorship limits.",
    ]),
    "tax_law": "\n".join([
        "[TOPIC-SPECIFIC MARKING RUBRIC — TAX LAW]",
        "- First-class/top-band script: identifies taxpayer, charge, timing, computation assumptions, relief, anti-avoidance route, compliance consequence and appeal/remedy.",
        "- Penalise: policy-only answers, invented rates/thresholds, GAAR as a catch-all, treaty analysis before domestic charge, transfer pricing left vague, and calculations hidden in prose.",
        "- Niche traps: Ramsay as purposive construction not discretion, Mayes limits, sham/evasion/avoidance distinctions, VAT place/time of supply, HMRC discovery/follower/penalty safeguards.",
        "- Offshore IP trap: sequence ordinary construction/Ramsay before transfer pricing/specific rules and then GAAR; assess DEMPE, control of risk, arm's-length royalty and intermediary functions.",
    ]),
    "environmental_law": "\n".join([
        "[TOPIC-SPECIFIC MARKING RUBRIC — ENVIRONMENTAL LAW / NUISANCE]",
        "- First-class/top-band script: separates standing, amenity nuisance, personal injury causation, statutory nuisance, regulatory discretion/JR, and remedies.",
        "- Penalise: treating permit compliance as a complete defence, merging odour/noise with respiratory injury, and using judicial review as merits appeal.",
        "- Niche traps: proprietary interest in private nuisance, Barr v Biffa permit point, Manchester Ship Canal statutory-scheme point, EPA 1990 ss 79/82 procedure, and evidence for contamination/health harm.",
    ]),
    "succession_wills": "\n".join([
        "[TOPIC-SPECIFIC MARKING RUBRIC — WILLS / SUCCESSION]",
        "- First-class/top-band script: states due execution and burden structure, applies Banks v Goodfellow to execution time, then centres suspicious circumstances on knowledge and approval before weaker undue influence.",
        "- Penalise: treating golden rule as validity rule, importing inter vivos undue influence presumptions, and discussing adult-child 1975 Act claims without maintenance need.",
        "- Niche traps: lucid intervals, solicitor file quality, carer arranging appointment, revocation by invalid later will, and Ilott maintenance-only limits for non-spouse applicants.",
    ]),
    "trusts_law": "\n".join([
        "[TOPIC-SPECIFIC MARKING RUBRIC — EQUITY / TRUSTS]",
        "- First-class/top-band script: run asset-by-asset, classify the equitable route first, keep creation/constitution/formality separate from remedy, and explain why proprietary relief matters.",
        "- Penalise: fairness labels without the trust trigger; remedy labels without insolvency consequence; tracing before establishing breach or proprietary base.",
        "- Niche traps: purpose-trust beneficiary principle and capriciousness, three certainties and administrative workability, secret-trust communication/acceptance timing, mixed-fund tracing and volunteer/value priority.",
    ]),
    "evidence_law": "\n".join([
        "[TOPIC-SPECIFIC MARKING RUBRIC — EVIDENCE LAW]",
        "- First-class/top-band script: classify the evidence precisely, use the statutory gateway, decide admissible/excluded/admitted with limits, then state warning, direction or trial consequence.",
        "- Penalise: 'relevant therefore admissible'; Article 6 as a free-standing override before domestic route; merging hearsay, previous statements, bad character and confession evidence.",
        "- Niche traps: CJA 2003 hearsay gateways and interests of justice, bad-character gateways and fairness exclusion, PACE ss.76/78, Turnbull identification warnings, expert reliability and vulnerable-witness limits.",
        "- Identification trap: Turnbull is not enough where video identification/social-media contamination facts raise PACE Code D compliance and procedural fairness.",
        "- Confession trap: threats or pressure may be stronger under s.76(2)(b) unreliability than narrow Fulling-style oppression; always keep s.78 separate.",
        "- Silence/bad-character trap: s.34 needs a later relied-on fact; bad-character gateways depend on defendant/co-defendant/non-defendant status and on probative value, not mere criminal flavour.",
    ]),
    "public_law": "\n".join([
        "[TOPIC-SPECIFIC MARKING RUBRIC — PUBLIC LAW / JUDICIAL REVIEW]",
        "- First-class/top-band script: identify decision-maker, power source, claimant, ground, review intensity, remedy and discretion; distinguish legality from merits and policy critique.",
        "- Penalise: generic unfairness/irrationality without a recognised ground; proportionality where the context does not justify it; planning answers without statutory hooks.",
        "- Niche traps: legitimate expectation clarity/reliance/overriding interest, fettering and relevant/irrelevant considerations, reasons and consultation, HRA proportionality, quashing/declaration/mandatory order limits.",
    ]),
    "eu_law": "\n".join([
        "[TOPIC-SPECIFIC MARKING RUBRIC — EU LAW]",
        "- First-class/top-band script: identifies the instrument and enforcement route, applies restriction/justification/proportionality or direct-effect/remedy in the correct order, and labels post-Brexit status.",
        "- Penalise: generic supremacy, directive horizontal direct effect errors, Keck/Cassis used without measure classification, and pre-Brexit propositions applied to UK facts without status check.",
        "- Niche traps: vertical/horizontal direct effect, indirect effect limits, state liability conditions, Article 34 selling arrangements/use restrictions, citizenship status and preliminary-reference duty/discretion.",
    ]),
    "biolaw_ai_data": "\n".join([
        "[TOPIC-SPECIFIC MARKING RUBRIC — BIOLAW / AI / DATA]",
        "- First-class/top-band script: identifies the technology, actor, affected interest, legal regime, current-law status, concrete gap and proportionate governance response.",
        "- Penalise: speculative technology claims, ethics without legal route, bias claims without mechanism, and reform proposals that ignore cost, delay or regulatory burden.",
        "- Niche traps: controller/processor and Article 9 status, DPIA/automated decision-making, medical-device validation, cybersecurity, post-market monitoring, AI risk classification and accountability gaps.",
        "- Dissertation-scale trap: theory must do legal work, current cases/statutes/reports must be status-checked, comparative claims need a mechanism, and reform labels need implementation detail.",
    ]),
    "private_international_law": "\n".join([
        "[TOPIC-SPECIFIC MARKING RUBRIC — PRIVATE INTERNATIONAL LAW]",
        "- First-class/top-band script: maps each forum, separates jurisdiction/applicable law/enforcement, and gives a practical litigation sequence.",
        "- Penalise: merits merged with jurisdiction, EU instruments used without post-Brexit status, forum convenience asserted without connecting factors and justice analysis.",
        "- Niche traps: service-out gateways, exclusive jurisdiction/arbitration clauses, Rome I/Rome II exceptions, anti-suit/freezing relief, recognition defences and foreign mandatory rules.",
        "- Post-Brexit essay trap: compare front-end permission/service/forum openness with back-end recognition/enforcement limits; Hague 2019 is partial repair, not a full Brussels/Lugano substitute.",
    ]),
    "public_international_law": "\n".join([
        "[TOPIC-SPECIFIC MARKING RUBRIC — PUBLIC INTERNATIONAL LAW]",
        "- First-class/top-band script: identifies the source of law, separates jurisdiction/attribution/breach/immunity/remedy, and states enforcement limits realistically.",
        "- Penalise: soft law treated as binding, jus cogens used as automatic answer, immunity and jurisdiction merged, private-international-law concepts imported without reason.",
        "- Niche traps: effective/control attribution, due diligence, countermeasures, necessity/proportionality in self-defence, ratione personae/material immunity and IHL classification/status.",
    ]),
    "criminal_law": "\n".join([
        "[TOPIC-SPECIFIC MARKING RUBRIC — CRIMINAL LAW]",
        "- First-class/top-band script: choose the plausible offence first, then AR/MR/causation/defence; rank strong, arguable and weak charges instead of analysing everything equally.",
        "- Penalise: long generic introductions, every theoretical offence at equal weight, consent/self-defence/intoxication discussed without the precise statutory or common-law trigger.",
        "- Niche traps: sports consent and lawful recognised categories, transferred malice, oblique intent, Jogee secondary liability, loss of control/diminished responsibility, theft dishonesty and appropriation.",
    ]),
    "pensions_law": "\n".join([
        "[TOPIC-SPECIFIC MARKING RUBRIC — PENSIONS LAW]",
        "- First-class/top-band script: identify scheme type, deed/rules, power, process, statutory overlay, dates, calculations, member status and practical remedy.",
        "- Penalise: undefined acronyms, conclusions without calculations, loose pinpoints, and treating member consensus or trustee discretion as self-proving.",
        "- Niche traps: Barber/equalisation timing, section 67 protected modifications, amendment powers, trustee impartiality, DB/DC distinction, employer covenant and Ombudsman/remedy route.",
        "- Non-financial investment trap: test Pensions Act 2004 s 255, the two formulations of financial-detriment risk, informed survey evidence, all beneficiary classes, DB/DC/default-fund differences and SIP/process steps.",
    ]),
    "mediation_law": "\n".join([
        "[TOPIC-SPECIFIC MARKING RUBRIC — MEDIATION]",
        "- First-class/top-band script: separate mediation process, mediator conduct, confidentiality/privilege, settlement formation, enforcement route and refusal ground; use arbitration only as a comparison, not as a substitute.",
        "- Penalise: generic praise of settlement culture, treating mediated settlements as arbitral awards, ignoring Article 5 refusal grounds, or discussing confidentiality without evidence/use consequences.",
        "- Niche traps: Singapore Convention critical mass, Article 5(1)(e) serious breach/mediator standards uncertainty, New York Convention benchmark, without-prejudice limits, mandatory ADR/access-to-court proportionality.",
    ]),
}

TOPIC_SPECIALIST_SEQUENCE_RULES: tuple[tuple[tuple[str, ...], str, tuple[str, ...]], ...] = (
    (("tax_",), "TAX", (
        "characterise tax advantage/charge -> purposive construction/Ramsay -> transfer pricing/TAAR/specific rule -> GAAR -> disclosure, penalties, appeal and settlement/remedy",
        "avoid GAAR-as-catch-all; identify the strongest statutory route and the taxpayer's best commercial-purpose answer",
    )),
    (("succession_",), "WILLS / SUCCESSION", (
        "formal validity -> capacity -> knowledge and approval -> undue influence/fraud -> invalidity consequence -> Inheritance Act 1975",
        "state burdens, execution-time capacity/lucid interval, and adult-child maintenance limits where relevant",
    )),
    (("generic_environmental_law", "climate_"), "ENVIRONMENTAL", (
        "standing/source -> private or regulatory route -> proof/causation -> statutory enforcement/JR -> injunction, damages, declaration or enforcement remedy",
        "treat permits, regulator discretion and current climate/statutory status as technical gateways, not policy slogans",
    )),
    (("competition_",), "COMPETITION", (
        "market definition -> dominance/agreement gateway -> abuse/object-effect theory -> foreclosure/consumer harm -> justification/exemption -> enforcement/remedy",
        "for digital/platform facts, include economics, data/ranking/default effects, and current DMCC/DMA/CMA/EU status where relevant",
    )),
    (("medical_", "clinical_negligence"), "LAW AND MEDICINE", (
        "patient/status -> consent/capacity/statutory gateway -> breach/lawfulness -> causation/best interests -> remedy or court order",
        "keep autonomy, battery/negligence, MCA best interests, and course-bound/current-law issues separate",
    )),
    (("land_", "generic_land_law"), "LAND", (
        "registered/unregistered status -> right asserted -> creation/formality -> protection/registration -> priority -> TOLATA/sale/injunction/remedy",
        "separate actual occupation, overreaching, notices/restrictions, quantification and proceeds",
    )),
    (("equity_",), "EQUITY / TRUSTS", (
        "asset/right -> creation/formality/constitution -> beneficial entitlement or duty -> breach/tracing/priority -> personal/proprietary remedy",
        "do not let conscience/fairness replace the precise equitable trigger or insolvency consequence",
    )),
    (("contract_",), "CONTRACT", (
        "formation/status -> term/representation -> breach/vitiating factor -> exclusion/unfairness -> causation/remoteness/mitigation -> remedy",
        "separate classification and remedy before using fairness language",
    )),
    (("commercial_", "generic_commercial_law", "aviation_", "maritime_", "insurance_", "banking_", "secured_transactions_"), "COMMERCIAL", (
        "transaction/instrument -> title/risk/obligation -> breach/default -> defences/limits -> insolvency/priority -> remedy/enforcement",
        "for specialist regimes, identify the convention/statute, time limit and exclusive remedy before common-law analysis",
    )),
    (("consumer_", "generic_consumer_protection_law", "product_liability_"), "CONSUMER / PRODUCT LIABILITY", (
        "consumer/product status -> statutory quality/safety/unfairness gateway -> defect/non-conformity -> causation -> trader/producer defence -> remedy/enforcement",
        "keep CRA rights, unfair terms, strict product liability and negligence distinct",
    )),
    (("civil_procedure_",), "CIVIL PROCEDURE / DISPUTE RESOLUTION", (
        "claim/status -> limitation/forum/track -> pleadings/disclosure/evidence -> applications/sanctions/ADR -> costs and final remedy/order",
        "keep merits, case management, Denton relief, summary judgment/strike-out, settlement and costs consequences separate",
    )),
    (("company_", "partnership_", "insolvency_", "corporate_bhr_"), "BUSINESS / COMPANY / INSOLVENCY", (
        "actor/status -> duty/authority/approval -> breach/transaction validity -> standing/enforcement route -> remedy, contribution or insolvency consequence",
        "separate company loss, shareholder prejudice, creditor interests, office-holder claims and veil/direct-duty routes",
    )),
    (("employment_",), "EMPLOYMENT", (
        "status -> time/eligibility -> statutory route -> employer reason/defence/process -> causation/adjustment -> remedy",
        "separate ordinary unfair dismissal, automatic unfairness, discrimination, whistleblowing, equal pay and restrictive covenants",
    )),
    (("family_",), "FAMILY", (
        "order/route sought -> statutory gateway -> welfare/needs/threshold factors -> proportionality/safeguards -> specific order",
        "keep children, finance, abduction, cohabitation and public-law threshold analysis separate",
    )),
    (("criminal_evidence", "evidence_"), "EVIDENCE", (
        "classify evidence -> statutory/common-law gateway -> exclusion/fairness discretion -> weight/warning/direction -> likely ruling/trial consequence",
        "keep admissibility, exclusion, weight and jury direction distinct",
    )),
    (("criminal_", "generic_criminal_law", "generic_sentencing_law", "generic_criminology"), "CRIMINAL / SENTENCING", (
        "offence/route -> actus reus -> mens rea -> causation -> defence/partial defence -> conviction/sentence consequence",
        "rank plausible charges and keep criminalisation theory separate from doctrinal liability",
    )),
    (("tort_",), "TORT", (
        "protected interest/tort -> duty or special gateway -> breach/interference -> causation/scope/remoteness -> defences -> damages/injunction",
        "avoid using generic negligence where occupiers, nuisance, defamation, psychiatric harm or economic loss has a specialist gateway",
    )),
    (("public_law_automated",), "PUBLIC LAW / DATA GOVERNANCE", (
        "power/source -> automated tool role -> legality/fairness/reasons -> rights/data-protection route -> human review -> remedy/systemic reform",
        "verify DUAA/UK GDPR current status and define meaningful human involvement concretely",
    )),
    (("public_law_", "generic_administrative_law", "generic_access_to_justice", "generic_rule_of_law", "constitutional_", "generic_devolution_law", "human_rights_"), "PUBLIC / CONSTITUTIONAL / HUMAN RIGHTS", (
        "decision/power source -> amenability/standing/time -> ground/right -> intensity/proportionality/deference -> remedy/discretion -> constitutional evaluation",
        "separate legality, merits, rule-of-law theory, rights review and political-constitutional limits",
    )),
    (("generic_planning_law",), "PLANNING", (
        "planning power -> development plan/statutory hook -> material considerations/consultation/reasons -> merits/legal-error boundary -> statutory/JR remedy",
        "do not turn disagreement with planning balance into illegality without a recognisable ground",
    )),
    (("generic_housing_law",), "HOUSING", (
        "occupier/status -> statutory protection/notice -> possession/repair/homelessness duty -> proportionality/defence -> order/remedy",
        "keep tenancy status, public-law duties and possession discretion separate",
    )),
    (("education_",), "EDUCATION", (
        "decision/status -> statutory or policy duty -> procedural fairness/SEND/discrimination route -> proportionality/welfare -> appeal/JR/remedy",
        "separate merits challenge from legal error and identify the correct tribunal, panel or court route",
    )),
    (("prison_",), "PRISON", (
        "state power -> prison rule/policy -> common-law fairness/HRA ground -> security/deference -> remedy and residual discretion",
        "separate treatment conditions, disciplinary fairness, release/recall and parole-type review",
    )),
    (("public_procurement_",), "PUBLIC PROCUREMENT", (
        "procurement regime/date -> duty/breach -> standstill/limitation -> causation/loss of chance -> automatic suspension or damages/set-aside remedy",
        "verify current Procurement Act/transition status and keep process legality distinct from commercial merits",
    )),
    (("pensions_",), "PENSIONS", (
        "scheme type -> deed/rules/date/member status -> power/process -> statutory overlay -> calculation/assumptions -> Ombudsman/court/practical remedy",
        "make Barber/equalisation, section 67, DB/DC and employer-covenant issues visible where relevant",
    )),
    (("legal_ethics_",), "LEGAL ETHICS / PROFESSIONAL REGULATION", (
        "client/retainer -> duty or code provision -> conflict/confidentiality/undertaking risk -> consent/exception -> stop-act-report-remedy",
        "separate regulatory breach, negligence, fiduciary conflict and privilege/confidentiality consequences",
    )),
    (("generic_charity_law",), "CHARITY", (
        "charitable purpose/public benefit -> trustee power/duty -> Commission/regulatory issue -> cy-près or enforcement route -> remedy/governance step",
        "separate validity, administration, political purpose, private benefit and trustee breach",
    )),
    (("generic_financial_regulation_law",), "FINANCIAL REGULATION", (
        "regulated activity/status -> rule/source -> breach or prudential risk -> supervisory/enforcement power -> private-law consequence/remedy",
        "keep authorisation, conduct rules, prudential standards and enforcement sanctions separate",
    )),
    (("eu_", "generic_eu_law"), "EU / POST-BREXIT", (
        "instrument/status -> scope/effect -> breach/restriction -> justification/proportionality -> enforcement/remedy -> UK post-Brexit status",
        "label EU law, assimilated/retained law, Withdrawal Agreement/Windsor Framework or domestic replacement before applying doctrine",
    )),
    (("private_international_law", "conflict_of_laws"), "PRIVATE INTERNATIONAL LAW", (
        "jurisdiction/service -> stay/forum -> applicable law -> interim relief -> recognition/enforcement",
        "keep merits, forum, governing law and enforcement separate; check post-Brexit instrument status",
    )),
    (("international_commercial_arbitration",), "ARBITRATION", (
        "agreement/scope -> seat/law -> tribunal jurisdiction/procedure -> challenge/enforcement gateway -> public-policy/due-process limit -> remedy",
        "verify current Arbitration Act amendments and keep seat law distinct from enforcement convention",
    )),
    (("generic_mediation_law",), "MEDIATION", (
        "agreement to mediate -> process/confidentiality -> mediator conduct -> settlement formation -> enforcement/refusal ground",
        "use arbitration only as comparison; do not treat mediated settlements as awards",
    )),
    (("public_international_law", "generic_international_law", "international_human_rights_", "ihl_", "climate_state_responsibility"), "PUBLIC INTERNATIONAL LAW / IHL", (
        "source -> jurisdiction/attribution/status -> breach -> exception/immunity/defence -> responsibility/reparation/enforcement",
        "separate treaty/custom, attribution, due diligence, immunity, IHL status and enforcement limits",
    )),
    (("generic_international_trade_law", "wto_"), "INTERNATIONAL TRADE / WTO", (
        "covered measure -> obligation/breach -> exception/security/justification -> necessity/good faith -> retaliation/compliance consequence",
        "work from treaty text before policy critique",
    )),
    (("immigration_", "refugee_", "generic_extradition_law"), "IMMIGRATION / ASYLUM / EXTRADITION", (
        "statutory/treaty route -> eligibility/barrier -> human-rights or non-refoulement risk -> evidence/assurances -> appeal/JR/remedy",
        "separate asylum, deportation, extradition, Article 3, Article 8 and fair-trial risks",
    )),
    (("ip_", "patent_"), "INTELLECTUAL PROPERTY", (
        "right/jurisdiction -> subsistence/validity -> ownership/entitlement -> infringement/restricted act -> defence/exception -> remedy",
        "separate copyright, trade mark, patent and passing-off functions rather than using generic IP policy",
    )),
    (("defamation_", "generic_freedom_of_expression_law"), "MEDIA / EXPRESSION", (
        "protected interest/speech -> threshold -> meaning/interference -> defence/justification/proportionality -> remedy/injunction",
        "keep defamation, misuse of private information, confidentiality and public-law expression routes separate",
    )),
    (("data_protection", "ai_", "generic_ai_law", "equality_substantive_framework"), "AI / DATA / EQUALITY", (
        "actor/system -> affected data/equality interest -> statutory basis -> discrimination/processing risk -> safeguards/audit/human review -> remedy/reform",
        "verify current GDPR/DUAA/AI status and identify the concrete proxy, opacity or proof-burden problem",
    )),
    (("cyber", "cybercrime"), "CYBER / COMPUTER MISUSE", (
        "conduct/system -> offence/civil route -> jurisdiction/evidence -> attribution/intent -> defence/safeguard -> remedy/enforcement",
        "separate criminal misuse, data breach, harassment, ransomware payment and cross-border cooperation issues",
    )),
    (("construction_",), "CONSTRUCTION", (
        "contract machinery -> time/delay/EOT -> defects/specification -> payment/adjudication -> loss measure -> practical remedy",
        "do not analyse delay or defects as bare common-law breach where contract mechanisms control the result",
    )),
    (("restitution_",), "RESTITUTION", (
        "enrichment -> at claimant's expense -> unjust factor -> defence/change of position -> proprietary/personal remedy",
        "separate mistake, failure of basis, duress and illegality rather than relying on broad fairness",
    )),
    (("sports_",), "SPORTS GOVERNANCE", (
        "body/rule -> private-law/public-law/competition route -> procedural fairness/proportionality -> autonomy/deference -> remedy",
        "separate natural justice, restraint of trade, anti-doping and competition arguments",
    )),
    (("cultural_",), "CULTURAL HERITAGE", (
        "object/provenance -> title/export/import rule -> limitation/good faith -> restitution/return framework -> remedy/diplomatic consequence",
        "separate ownership, public-law controls, international instruments and ethical repatriation arguments",
    )),
    (("space_law",), "SPACE LAW", (
        "actor/object -> treaty obligation -> attribution/jurisdiction -> damage/liability -> registration/mitigation -> claim/enforcement route",
        "separate OST duties, Liability Convention, registration and domestic licensing",
    )),
    (("election_",), "ELECTION LAW", (
        "regulated actor -> campaign finance/rules -> breach -> enforcement body -> sanction/remedy/election validity",
        "separate criminal, civil/regulatory and democratic-legitimacy consequences",
    )),
    (("generic_agency_law",), "AGENCY", (
        "principal-agent relationship -> actual authority -> apparent authority -> ratification -> fiduciary breach -> third-party and internal remedies",
        "keep external liability separate from internal indemnity/accountability",
    )),
    (("jurisprudence_", "generic_legal_history", "statutory_interpretation"), "LEGAL THEORY / HISTORY / INTERPRETATION", (
        "thesis/concept -> competing theory/rule -> authority or historical evidence -> counterargument -> evaluative conclusion",
        "define technical concepts and link examples to the thesis rather than writing a chronology",
    )),
)


def build_topic_specialist_sequence_block(topic: str = "", query: str = "") -> str:
    topic_key = (topic or "").strip().lower()
    low = (query or "").strip().lower()
    for prefixes, label, rules in TOPIC_SPECIALIST_SEQUENCE_RULES:
        if topic_key:
            matched = any(topic_key.startswith(prefix) for prefix in prefixes)
        else:
            matched = any(prefix and prefix in low for prefix in prefixes)
        if matched:
            return "\n".join([
                f"[SPECIALIST SEQUENCE MATRIX — {label}]",
                *[f"- {rule}" for rule in rules],
            ])
    return "\n".join([
        "[SPECIALIST SEQUENCE MATRIX — GENERAL LEGAL]",
        "- classify route/status -> exact gateway/test -> factual application -> counterargument -> remedy/procedure -> final practical outcome",
        "- if no subject-specific sequence matched, make the missing assumptions explicit and avoid inventing specialist law",
    ])


GOLDEN_OUTPUT_AUDIT_SUITE: tuple[Dict[str, Any], ...] = (
    {
        "name": "law_medicine_course_bound_autonomy",
        "prompt": "Law and Medicine course-bound essay: critically examine bodily autonomy using two or three syllabus examples only.",
        "must_show": ("course-bound", "two or three focused examples", "statutory route", "ethics"),
    },
    {
        "name": "law_medicine_no_limit_autonomy",
        "prompt": "Law and Medicine no syllabus limit essay: critically examine bodily autonomy using medical law more widely.",
        "must_show": ("no syllabus limit", "wider material", "current-law verification", "statutory route"),
    },
    {
        "name": "competition_article_102_problem",
        "prompt": "Competition Law Article 102 problem: advise on self-preferencing, data access, dominance, effects and objective justification.",
        "must_show": ("dominance", "abuse theory", "effects", "objective justification", "remedy"),
    },
    {
        "name": "land_registered_priority_gold_shape",
        "prompt": "Land Law problem: advise on registered land priority involving an unregistered easement, a lease, an option, actual occupation and overreaching.",
        "must_show": ("registered status", "right-by-right priority", "actual occupation", "overreaching", "remedy"),
    },
    {
        "name": "trusts_tracing_secret_trust_gold_shape",
        "prompt": "Trusts Law problem: advise on purpose trusts, secret trusts, certainty, tracing through a mixed fund and proprietary remedies.",
        "must_show": ("beneficiary principle", "secret-trust timing", "three certainties", "mixed-fund tracing", "personal/proprietary remedy"),
    },
    {
        "name": "company_conflicts_insolvency_gold_shape",
        "prompt": "Company Law problem: advise on director conflict, related-party transaction approval, minority remedies and insolvency-office-holder claims.",
        "must_show": ("client/claim owner", "director duties", "approval route", "minority remedy", "insolvency remedy"),
    },
    {
        "name": "evidence_admissibility_gold_shape",
        "prompt": "Evidence Law problem: advise on hearsay, bad character, confession, identification and expert evidence in one criminal trial.",
        "must_show": ("classify evidence", "statutory gateway", "fairness exclusion", "weight/direction", "likely ruling"),
    },
    {
        "name": "public_law_legitimate_expectation_gold_shape",
        "prompt": "Public Law problem: advise on legitimate expectation, improper pressure, reasons, proportionality and JR remedies.",
        "must_show": ("power source", "review ground", "clarity/reliance/fairness", "intensity", "discretionary remedy"),
    },
    {
        "name": "pensions_nra_equalisation_gold_shape",
        "prompt": "Pensions Law problem: advise on NRA equalisation, Barber-window benefits, amendment formalities, section 67 and visible calculation method.",
        "must_show": ("scheme type", "Barber timing", "section 67", "visible workings", "practical trustee steps"),
    },
    {
        "name": "mediation_singapore_convention_gold_shape",
        "prompt": "International Commercial Mediation essay: evaluate the Singapore Convention using the New York Convention benchmark, Article 5 and confidentiality tensions.",
        "must_show": ("Singapore Convention", "New York Convention benchmark", "Article 5", "confidentiality", "critical thesis"),
    },
    {
        "name": "sqe2_written_marking",
        "prompt": "SQE2 legal writing marking: mark my answer against the criteria and give a corrected model.",
        "must_show": ("criterion-by-criterion", "A-F", "recipient focus", "corrected high-scoring answer"),
    },
)

LIVE_PROMPT_AUDIT_SUITE: tuple[Dict[str, Any], ...] = (
    {
        "name": "Law and Medicine course-bound essay",
        "prompt": "Law and Medicine - Essay Question. Stay within the module syllabus. Critically examine whether English law protects bodily autonomy adequately.",
        "checks": ("stays within syllabus", "uses 2-3 examples", "does not drift into Montgomery/negligence unless necessary", "takes a thesis"),
    },
    {
        "name": "Law and Medicine no-limit essay",
        "prompt": "Law and Medicine broad-all / no syllabus limit essay. Critically examine bodily autonomy across English medical law.",
        "checks": ("labels wider material", "verifies current law", "still uses focused structure", "does not become a generic survey"),
    },
    {
        "name": "Competition Article 102 problem",
        "prompt": "Competition Law - Problem Question. Advise on dominance, self-preferencing, refusal/access terms, effects and objective justification.",
        "checks": ("separates dominance and abuse", "chooses abuse theory", "uses effects evidence", "states remedy/enforcement risk"),
    },
    {
        "name": "SQE2 written marking/practice",
        "prompt": "SQE2 legal research practice and marking prompt: generate a hard task, then mark the candidate answer against criteria.",
        "checks": ("uses correct skill criteria", "task is harder than sample but answerable", "marks against A-F", "gives next targeted practice"),
    },
    {
        "name": "Land Law registered-priority stress",
        "prompt": "Land Law problem: registered land with unregistered easement, long lease, option, actual occupation and possible overreaching.",
        "checks": ("registered/unregistered status first", "separate each right", "creation/protection/priority/remedy", "actual occupation and overreaching kept separate"),
    },
    {
        "name": "Tort psychiatric/public-authority stress",
        "prompt": "Tort problem: police non-attendance, negligent arrest injury, psychiatric harm to relative and officer.",
        "checks": ("recognised psychiatric injury", "primary/secondary victim distinction", "omission/public authority route", "operational/policy distinction"),
    },
    {
        "name": "Company conflict/insolvency stress",
        "prompt": "Company Law problem: conflicted related-party transaction, dividends near insolvency, minority shareholder and liquidator remedies.",
        "checks": ("director-specific duties", "approvals and ratification", "derivative/unfair prejudice routes", "creditor-interest and insolvency remedies"),
    },
    {
        "name": "Trusts creation/tracing stress",
        "prompt": "Trusts problem: purpose trusts, secret trust, uncertain beneficiaries, mixed-fund tracing and proprietary remedies.",
        "checks": ("creation/formality/constitution separate", "beneficiary principle", "secret-trust timing", "personal/proprietary remedies"),
    },
    {
        "name": "Evidence admissibility stress",
        "prompt": "Evidence Law problem: hearsay, bad character, confession, identification and expert evidence in one criminal trial.",
        "checks": ("classifies each item", "statutory gateways", "PACE/CJA and Article 6 sequence", "ruling plus direction/safeguard"),
    },
    {
        "name": "Public Law legitimate-expectation stress",
        "prompt": "Public Law problem: policy promise, immediate licence revocation, ministerial pressure, reasons and remedies.",
        "checks": ("power source and ground", "clarity/reliance/fairness", "overriding public interest", "discretionary remedy"),
    },
    {
        "name": "Pensions NRA/equalisation stress",
        "prompt": "Pensions Law problem: DB scheme with unequal historic NRAs, Barber equalisation, late deed amendment, section 67 and actuarial/value changes.",
        "checks": ("scheme rules and dates first", "Barber-window timing", "section 67/accrued rights", "visible workings", "trustee practical steps"),
    },
    {
        "name": "Pensions non-financial investment stress",
        "prompt": "Pensions Law problem: trustees want ethical divestment despite weak employer covenant, incomplete member survey and possible financial detriment.",
        "checks": ("financial and non-financial factors separated", "member consensus tested", "financial detriment assessed", "DB/DC and employer covenant context", "process safeguards"),
    },
    {
        "name": "Mediation Singapore Convention stress",
        "prompt": "International Commercial Mediation essay: critically assess whether the Singapore Convention solves enforcement of international mediated settlements.",
        "checks": ("New York Convention used only as benchmark", "enforceability gap", "Article 5 refusal grounds", "mediator standards", "confidentiality/evidence tension"),
    },
    {
        "name": "Mediation process/enforcement stress",
        "prompt": "International Commercial Mediation problem: tiered mediation clause, early litigation, mediator conflict, confidential caucus statements and short settlement term sheet.",
        "checks": ("agreement to mediate/stay", "mediator conduct", "confidentiality and without-prejudice", "settlement contract/enforcement", "cross-border route"),
    },
)


def law_medicine_syllabus_mode(query: str) -> str:
    low = (query or "").lower()
    if any(marker in low for marker in NO_SYLLABUS_LIMIT_MARKERS):
        return "no_limit"
    if any(marker in low for marker in COURSE_BOUND_MARKERS):
        return "course_bound"
    return "default_course_bound"


def build_law_medicine_syllabus_mode_block(query: str, slug: str) -> str:
    if slug != "law_medicine":
        return ""
    mode = law_medicine_syllabus_mode(query)
    if mode == "no_limit":
        return "\n".join([
            "[LAW AND MEDICINE SOURCE MODE: NO SYLLABUS LIMIT]",
            "- The course-bound exclusions do not apply because the user asked for broad/no-limit analysis.",
            "- Still distinguish course-core material from wider English medical-law material.",
            "- Verify current-law status before relying on recent statutes, bills, regulator guidance, or reform proposals.",
        ])
    label = "COURSE-BOUND" if mode == "course_bound" else "DEFAULT COURSE-BOUND"
    return "\n".join([
        f"[LAW AND MEDICINE SOURCE MODE: {label}]",
        "- Stay within medical ethics, consent/refusal/capacity, end-of-life, transplantation, abortion, and reproductive medicine unless the prompt expressly expands scope.",
        "- Do not drift into clinical negligence/Montgomery, mental health law, deprivation of liberty, public health, US law, or surrogacy unless directly necessary.",
        "- Use wider material only as a brief contrast if it is needed to answer the exact question.",
        "- For each chosen syllabus example, identify the governing legal route before critique: common-law consent/refusal, Mental Capacity Act 2005, Human Tissue Act 2004, Abortion Act 1967, or Human Fertilisation and Embryology Act 1990 as applicable.",
    ])


def build_source_quality_priority_block() -> str:
    labels = "\n".join(f"- {label}: {meaning}" for label, meaning in SOURCE_QUALITY_LABELS)
    return "\n".join([
        "[SOURCE QUALITY PRIORITY GATE]",
        "Before drafting, classify retrieved material by source quality and use the highest-value material first:",
        labels,
        "Priority order: official_primary and feedback_marking/course_material for assessment technique; official_primary for legal rules; secondary_commentary for critique; weak_or_noise only to trigger better retrieval, not to support claims.",
        "Do not treat every RAG chunk equally. If a source is only a filename/snippet/noisy duplicate, do not cite it or let it control the answer.",
    ])


def build_full_rag_answer_flow_block() -> str:
    return "\n".join([
        "[FULL RAG ANSWER FLOW — MANDATORY]",
        "- First classify the user request: essay, problem question, chat explanation, SQE task, document amendment, marking, or question generation.",
        "- Then detect subject area, jurisdiction, citation style, privacy mode, document scope, and whether the user gave explicit instructions overriding defaults.",
        "- Use selected chat uploads first where present. Only selected/ticked uploads are in scope; unticked uploads must be ignored.",
        "- For selected PDFs/DOCX/text, use extracted relevant page/paragraph/sentence evidence before unsupported model recall. Cite exact pages/paragraphs only where the extracted source location is explicit.",
        "- Run indexed legal RAG before generation: BM25 exact keyword/legal-reference search plus vector semantic search.",
        "- Merge BM25 and vector results, rerank/filter for relevance, subject fit, source quality, current-law status, and document diversity.",
        "- Build an internal source ledger before drafting: each material proposition should map to uploaded evidence, indexed RAG, code/subject guide, or verified online-source context.",
        "- If indexed/uploaded coverage is thin, outdated, or current-law-sensitive, use online official-source fallback where configured/allowed; prefer primary/official sources.",
        "- If the source base remains insufficient, state the limitation and narrow the answer rather than inventing authorities, facts, citations, page numbers, or quotations.",
        "- Generate the answer only after retrieval and source-ledger checks, using the matched answer guide for essay/problem/SQE/chat/doc-amend shape.",
        "- Apply citation guard: no fake citations, no fake pinpoints, no raw local paths, no internal retrieval labels, no reconstructed quotations.",
        "- Run a supervisor pass before final output: legal accuracy, exact test, issue coverage, source support, current-law status, citation placement, guide compliance, conclusion/outcome, and privacy leakage.",
        "- If the supervisor pass finds a material failure, revise before output. Do not present an answer as source-grounded or online-checked unless that route actually ran.",
        "- Final output should be source-traceable and candid about uncertainty. Do not promise absolute certainty where source coverage or law-currentness does not support it.",
    ])


def build_reference_citation_policy_block() -> str:
    return "\n".join([
        "[REFERENCE / CITATION POLICY]",
        "- Default referencing style is inline OSCOLA unless the user expressly requests another style.",
        "- Put the full OSCOLA reference immediately after the sentence or proposition it supports, not as a loose source dump.",
        "- If the user explicitly requests Harvard, APA, MLA, Chicago or another style, follow that style throughout.",
        "- Provide a bibliography/reference list only when the user asks for one or when the requested style/task clearly requires it.",
        "- Pinpoint discipline is strict: cite exact pages, paragraphs, schedules or clauses only where the exact location is verifiable from indexed RAG, uploaded source text, or supplied official/search-backed context.",
        "- If an exact pinpoint is not verifiable, cite only the case, statute, article, journal, book or web source level; never invent page numbers, paragraph numbers, quotations or metadata.",
        "- Where indexed coverage is thin and a search-backed source/link is genuinely available, include the real clickable link for that proposition rather than fabricating an OSCOLA pinpoint.",
        "- Uploaded chat documents are transient selected sources. Use only selected attachments for that chat turn; do not treat unticked uploads as sources and do not describe them as globally indexed.",
        "- Where selected uploaded PDFs/DOCX/text provide extracted page/paragraph/sentence excerpts, use those excerpts as evidence and cite page/paragraph pinpoints only when the location is explicitly present.",
        "- If the user asks which exact sentences are quoted, quote only exact sentences visible in retrieved/indexed/source excerpts; do not reconstruct, paraphrase, or invent quotations.",
        "- Explicit user instructions on citation style, quote handling, bibliography/reference-list heading, language, structure, scope, or exclusions override default house style unless they conflict with no-fabrication or safety rules.",
    ])


def build_answer_specificity_gate() -> str:
    return "\n".join([
        "[ANSWER SPECIFICITY / ANTI-GENERIC GATE]",
        "- Every major issue must apply to facts before moving to the next issue.",
        "- Rank likely/arguable/weak outcomes where facts permit; do not leave all arguments artificially equal.",
        "- State remedy/next step before ending, including practical consequence, enforcement route, or residual evidence needed.",
        "- Do not survey the topic when the user asked for advice or a focused essay; select the legal issues that answer the prompt.",
        "- Do not use vague policy unless tied to doctrine, authority, evidence, or the statutory test.",
        "- Run a specialist accuracy pass for exact statutory gateway, procedural requirement, remedy route, and current-law status.",
    ])


def build_specialist_accuracy_pass_gate() -> str:
    return "\n".join([
        "[SPECIALIST ACCURACY PASS — TOP FIRST-CLASS STANDARD]",
        "- Before finalising, run a silent subject-specialist check for: exact statutory gateway, procedural requirement, evidential threshold, remedy route, timing/date trap, and current-law status.",
        "- For problem questions, classify the legal route first, state the exact test, apply the strongest facts, state the best counterargument, rank likely/arguable/weak, then give remedy and practical next step.",
        "- For essays, define the central contested concept, identify the live doctrinal/policy tension, steel-man the counterargument, use authority economically, and explain why the final thesis follows.",
        "- Replace broad correctness with technical exactness: if a rule depends on status, timing, commencement, registration, source hierarchy, party role, or procedural step, say so explicitly.",
        "- Final paragraph discipline: strongest route, weakest route, likely remedy/outcome, residual risk, and any missing evidence must be clear before the answer ends.",
    ])


def build_source_freshness_gate(query: str, slug: str = "", topic: str = "") -> str:
    low = (query or "").lower()
    slug_low = (slug or "").lower()
    topic_low = (topic or "").lower()
    triggered: List[str] = []
    if any(marker in low for marker in GENERAL_CURRENT_LAW_MARKERS):
        triggered.append("The user used current/latest/recent language, so current-law verification is required.")
    for target_slug, markers, instruction in FRESHNESS_TRIGGERS:
        target_matches_context = slug_low == target_slug or topic_low.startswith(target_slug) or target_slug in low
        marker_matches = any(marker in low for marker in markers)
        if target_matches_context or marker_matches:
            triggered.append(instruction)
    if not triggered:
        return ""
    deduped: List[str] = []
    seen = set()
    for item in triggered:
        key = re.sub(r"\s+", " ", item.strip().lower())
        if key and key not in seen:
            seen.add(key)
            deduped.append(item)
    return "\n".join([
        "[CURRENT-LAW / FRESHNESS GATE]",
        "- Use online search or official/current sources where available before stating any unstable rule.",
        "- Prefer official primary sources first: legislation.gov.uk, courts/tribunals, SRA/Kaplan, CMA/EU Commission, ICO, ACAS/government/regulator guidance as relevant.",
        "- If current verification is unavailable, say so and use calibrated wording instead of presenting the point as settled.",
        *[f"- {item}" for item in deduped],
    ])


def build_subject_answer_template_block(slug: str, query: str = "") -> str:
    return SUBJECT_TEMPLATE_BY_SLUG.get((slug or "").strip().lower(), "")


def build_topic_marking_rubric_block(slug: str, topic: str = "") -> str:
    slug_key = (slug or "").strip().lower()
    if slug_key in TOPIC_MARKING_RUBRIC_BY_SLUG:
        return TOPIC_MARKING_RUBRIC_BY_SLUG[slug_key]
    topic_key = (topic or "").strip().lower()
    if topic_key.startswith(("company_", "business_", "partnership_", "insolvency_")):
        return TOPIC_MARKING_RUBRIC_BY_SLUG.get("business_law", "")
    if topic_key.startswith(("equity_", "trust")):
        return TOPIC_MARKING_RUBRIC_BY_SLUG.get("trusts_law", "")
    if topic_key.startswith("public_law") or topic_key.startswith(("generic_judicial_review", "generic_constitutional_law")):
        return TOPIC_MARKING_RUBRIC_BY_SLUG.get("public_law", "")
    return ""


def build_answer_quality_addon_blocks(query: str, slug: str = "", topic: str = "") -> str:
    blocks = [
        build_law_medicine_syllabus_mode_block(query, slug),
        build_full_rag_answer_flow_block(),
        build_source_quality_priority_block(),
        build_reference_citation_policy_block(),
        build_answer_specificity_gate(),
        build_specialist_accuracy_pass_gate(),
        build_topic_specialist_sequence_block(topic=topic, query=query),
        build_source_freshness_gate(query, slug=slug, topic=topic),
        build_topic_marking_rubric_block(slug, topic=topic),
        build_subject_answer_template_block(slug, query=query),
    ]
    return "\n\n".join(block for block in blocks if block.strip())


def build_golden_output_audit_prompt() -> str:
    lines = [
        "[GOLDEN OUTPUT AUDIT SUITE]",
        "Use these prompts for manual/live QA. A pass means the output is specific, structured, source-aware, and top-band in style rather than a generic legal summary.",
    ]
    for item in GOLDEN_OUTPUT_AUDIT_SUITE:
        lines.append(f"- {item['name']}: {item['prompt']} | must show: {', '.join(item['must_show'])}")
    return "\n".join(lines)


def build_live_prompt_audit_checklist() -> str:
    lines = [
        "[LIVE PROMPT AUDIT CHECKLIST]",
        "Run these prompts through the app after prompt-layer changes and manually score the checks.",
    ]
    for item in LIVE_PROMPT_AUDIT_SUITE:
        lines.append(f"- {item['name']}: {item['prompt']} | checks: {', '.join(item['checks'])}")
    return "\n".join(lines)


def build_sqe2_practice_marking_loop_block(skill_label: str = "") -> str:
    label = (skill_label or "the selected written skill").strip()
    return "\n".join([
        "[SQE2 HARD PRACTICE + MARKING LOOP]",
        f"- Generate practice around {label}, the requested practice area, and the user's weak/wrong/niche focus.",
        "- Make tasks harder than official samples but still answerable from the facts/sources provided.",
        "- If the user is being tested, withhold answers unless they expressly request model answers or marking points.",
        "- When the user submits an answer, mark criterion-by-criterion on the A-F scale, identify missed law/ethics/practical steps, then provide a corrected high-scoring answer.",
        "- End marking feedback with `Next targeted practice`: one concise recommendation for the next harder task based on the weakest criterion.",
    ])
