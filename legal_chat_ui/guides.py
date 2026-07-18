"""Subject detection + guide-based structuring + supervisor quality blocks.

- detect_subject(question)  -> a subject slug (matches law_guides/<slug>.md)
- guide_method(slug)        -> the guide's 'how to structure the answer' sections
- supervisor_quality(...)   -> curated quality-control blocks for the supervisor

The structuring text comes from the bundled, anonymized law guides; the quality
gates come from the reused legal_answer_quality_controls module.
"""

from __future__ import annotations

import re
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
BUNDLED_GUIDES_DIR = APP_DIR / "law_guides"
LEGACY_GUIDES_DIR = APP_DIR.parent / "model_database" / "snapshot" / "law_guides"
GUIDES_DIR = BUNDLED_GUIDES_DIR if BUNDLED_GUIDES_DIR.is_dir() else LEGACY_GUIDES_DIR

try:
    import legal_answer_quality_controls as _q
    _Q_OK = True
except Exception:
    _q = None
    _Q_OK = False

# Distinctive keywords -> guide slug. ``detect_subject`` keeps the first match
# for compatibility; ``detect_subjects`` returns every relevant guide for
# integrated/mega questions.
_SUBJECT_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("aviation_law", ("aviation law", "air passenger", "international flight", "montreal convention", "air carrier", "checked baggage")),
    ("construction_law", ("construction law", "construction contract", "construction adjudication", "housing grants construction", "building contractor")),
    ("cultural_heritage_law", ("cultural heritage", "antiquity", "museum acquisition", "unlawfully exported", "dealing in cultural objects")),
    ("cybercrime_law", ("cybercrime", "computer misuse", "unauthorised access", "hacking", "malware", "cloud account")),
    ("election_law", ("election law", "election candidate", "electoral commission", "campaign advert", "corrupt practice", "illegal practice")),
    ("extradition_law", ("extradition", "part 1 warrant", "extradition act", "requesting state")),
    ("insurance_law", ("insurance law", "insurance act 2015", "insurer", "policyholder", "insurance policy", "avoid the policy")),
    ("international_trade_law", ("international trade", "trade remedies", "trade remedies authority", "wto", "anti-dumping", "countervailing")),
    ("maritime_law", ("maritime law", "bill of lading", "hague-visby", "carriage of goods by sea", "cargo damage")),
    ("public_procurement_law", ("public procurement", "procurement act", "standstill period", "unsuccessful bidder", "award criteria")),
    ("legal_ethics", ("legal ethics", "professional ethics", "professional conduct", "solicitor ethics",
                       "duties to the court", "duty to the court", "professional discipline", "sra principles", "client privilege")),
    ("law_medicine", ("medical", "medicine", "clinical negligence", "consent to treatment", "capacity act", "bolam")),
    ("privacy_media_law", ("defamation", "misuse of private information", "media law", "journalist", "press freedom", "online safety")),
    ("biolaw_ai_data", ("data protection", "data breach", "gdpr", "artificial intelligence", " ai ", "biolaw", "robotics", "facial recognition", "algorithmic")),
    ("consumer_law", ("consumer law", "consumer rights", "unfair commercial practice", "unfair trading", "misleading advertising", "trader and consumer")),
    ("contract_law", ("contract", "consideration", "offer and acceptance", "counter-offer", "counter offer",
                       "battle of forms", "certainty of terms", "payment term", "misrepresentation",
                       "breach of contract", "exclusion clause", "frustration")),
    ("tort_law", ("negligence", "duty of care", "nuisance", "tort", "occupiers", "defamation", "vicarious liability",
                  "psychiatric injury", "pure economic loss")),
    ("trusts_law", ("trust", "equity", "fiduciary", "constructive trust", "resulting trust", "beneficiary")),
    ("land_law", ("land", "easement", "covenant", "mortgage", "adverse possession", "co-ownership", "registered land", "lease")),
    ("criminal_procedure_law", ("criminal procedure", "criminal trial", "fitness to plead", "plea and trial", "criminal disclosure", "bail application")),
    ("criminal_law", ("criminal", "murder", "theft", "mens rea", "actus reus", "manslaughter", "self-defence",
                      "homicide", "diminished responsibility", "loss of control", "accessorial liability",
                      "intoxication")),
    ("evidence_law", ("evidence law", "hearsay", "admissibility", "burden of proof", "witness competence", "bad character evidence")),
    ("competition_law", ("competition", "cartel", "coordinating resale prices", "resale price maintenance", "chapter i prohibition", "abuse of dominance", "article 101", "article 102", "merger control",
                         "tfeu", "market definition", "ca 1998", "predatory pricing")),
    ("commercial_law", ("commercial", "sale of goods", "agency", "carriage", "bills of lading", "romalpa",
                        "nemo dat", "retention of title", "cif", "passing of property")),
    ("insolvency_law", ("insolvency", "liquidation", "administrator", "wrongful trading", "fraudulent trading", "company moratorium", "creditor losses")),
    ("business_law", ("company law", "director", "shareholder", "corporate group", "corporate governance", "piercing the veil",
                      "corporate opportunity", "corporate veil", "companies act 2006", "ca 2006")),
    ("intellectual_property_law", ("copyright", "patent", "trade mark", "trademark", "passing off", "design right", "intellectual property", "copyright software",
                                   "originality", "cdpa", "computer-generated works")),
    ("employment_law", ("employment law", "unfair dismissal", "redundancy", "worker status", "discrimination at work",
                        "unsafe workplace", "refusing to return", "serious and imminent danger",
                        "whistleblowing", "employee status", "self-employed", "trade union", "trade-union")),
    ("family_law", ("family law", "divorce", "financial remedy", "child arrangements", "ancillary relief", "matrimonial")),
    ("tax_law", ("tax", "hmrc", "vat", "capital gains", "income tax", "avoidance", "statutory residence test", "tax residence")),
    ("pensions_law", ("pension", "occupational scheme", "trustee of the scheme", "auto-enrolment")),
    ("environmental_law", ("environmental", "pollution", "planning permission", "private nuisance", "rylands")),
    ("succession_wills", ("wills law", "valid will", "handwritten will", "witnessed will", "succession", "intestacy", "probate", "testator", "executor")),
    ("public_law", ("judicial review", "wednesbury", "administrative", "ultra vires", "public law", "constitutional",
                    "parliamentary sovereignty", "rule of law", "separation of powers", "royal prerogative",
                    "ouster clause", "prorogation", "devolution", "legitimate expectation")),
    ("eu_law", ("eu law", "european union", "direct effect", "supremacy", "preliminary reference")),
    ("private_international_law", ("conflict of laws", "private international", "jurisdiction clause", "applicable law", "rome i")),
    ("public_international_law", ("international law", "treaty", "state responsibility", "customary international", "un charter")),
    ("mediation_law", ("mediation", "adr", "arbitration", "settlement negotiation")),
    ("human_rights_law", ("human rights", "echr", "convention right", "proportionality", "strasbourg",
                          "article 8", "article 10", "freedom of expression", "right to private life",
                          "margin of appreciation", "human rights act")),
    ("restitution_law", ("unjust enrichment", "restitution", "change of position", "mistaken payment", "pays under a mistake", "quantum meruit")),
    ("remedies_law", ("law of remedies", "remedies essay", "choice of remedy", "damages, injunctions", "specific performance", "rescission", "tracing", "declarations")),
    ("equality_law", ("equality act", "protected characteristic", "direct discrimination", "indirect discrimination",
                      "reasonable adjustments", "victimisation", "harassment at work")),
    ("immigration_refugee_law", ("immigration", "asylum", "refugee", "deportation", "leave to remain", "removal directions",
                                 "refugee convention", "nationality and borders act", "illegal migration act")),
    ("housing_law", ("housing", "homelessness", "secure tenancy", "assured shorthold", "possession order", "disrepair")),
    ("jurisprudence_law", ("jurisprudence", "legal positivism", "natural law", "dworkin", "legal theory", "rule of recognition",
                           "interpretivism")),
    ("civil_procedure_law", ("civil procedure", "summary judgment", "case management", "part 36", "disclosure obligations")),
    ("sentencing_law", ("sentencing", "sentencing council", "custodial sentence", "guilty-plea credit", "offence category", "culpability and harm", "sentencing guidelines", "dangerous offender", "dangerousness")),
    ("financial_regulation_law", ("financial regulation", "investment platform", "fca authorisation", "financial promotion", "client money", "fsma", "market abuse", "authorised person", "fca handbook")),
]

_EXPLICIT_SUBJECTS: tuple[tuple[str, str], ...] = (
    ("legal ethics", "legal_ethics"), ("professional ethics", "legal_ethics"),
    ("contract law", "contract_law"), ("tort law", "tort_law"),
    ("criminal procedure", "criminal_procedure_law"), ("criminal law", "criminal_law"),
    ("public law", "public_law"), ("administrative law", "public_law"),
    ("human rights law", "human_rights_law"), ("land law", "land_law"),
    ("equity and trusts", "trusts_law"), ("trusts law", "trusts_law"),
    ("company law", "business_law"),
    ("insolvency law", "insolvency_law"), ("commercial law", "commercial_law"),
    ("eu law", "eu_law"), ("public international law", "public_international_law"),
    ("private international law", "private_international_law"),
    ("jurisprudence", "jurisprudence_law"), ("legal theory", "jurisprudence_law"),
    ("evidence law", "evidence_law"), ("civil procedure", "civil_procedure_law"),
    ("family law", "family_law"), ("employment law", "employment_law"),
    ("medical law", "law_medicine"), ("intellectual property", "intellectual_property_law"),
    ("data protection", "biolaw_ai_data"), ("privacy / media law", "privacy_media_law"),
    ("environmental law", "environmental_law"), ("immigration law", "immigration_refugee_law"),
    ("tax law", "tax_law"), ("aviation law", "aviation_law"),
    ("construction law", "construction_law"), ("cultural heritage law", "cultural_heritage_law"),
    ("cybercrime law", "cybercrime_law"), ("election law", "election_law"),
    ("extradition law", "extradition_law"), ("insurance law", "insurance_law"),
    ("international trade law", "international_trade_law"), ("maritime law", "maritime_law"),
    ("public procurement law", "public_procurement_law"), ("pensions law", "pensions_law"),
    ("sentencing law", "sentencing_law"), ("succession law", "succession_wills"),
    ("financial regulation law", "financial_regulation_law"),
)


def detect_subject(question: str) -> str:
    t = f" {(question or '').lower()} "
    for phrase, slug in _EXPLICIT_SUBJECTS:
        if _keyword_matches(t, phrase):
            return slug
    # Realistic questions rarely name their subject, and a single shared word
    # ("contract", "criminal", "bill of lading") routinely appears outside its
    # own field.  Rank subjects by the number of distinct keyword hits so the
    # dominant subject wins; a tie keeps the historical first-match order.
    best_slug, best_hits = "", 0
    for slug, kws in _SUBJECT_KEYWORDS:
        hits = sum(1 for kw in kws if _keyword_matches(t, kw))
        if hits > best_hits:
            best_slug, best_hits = slug, hits
    return best_slug


def _keyword_matches(text: str, keyword: str) -> bool:
    """Phrase match without false positives such as ``tax`` in ``statutory``."""
    value = keyword.strip().lower()
    if not value:
        return False
    return bool(re.search(rf"(?<!\w){re.escape(value)}(?!\w)", text))


def detect_subjects(question: str, limit: int = 10) -> list[str]:
    """Return all relevant subject guides, in priority order, without duplicates."""
    text = f" {(question or '').lower()} "
    subjects: list[str] = []
    for slug, keywords in _SUBJECT_KEYWORDS:
        if any(_keyword_matches(text, keyword) for keyword in keywords) and slug not in subjects:
            subjects.append(slug)
        if len(subjects) >= limit:
            break
    return subjects


def _section(text: str, heading: str) -> str:
    """Return the body of a '## heading' section from a guide markdown."""
    m = re.search(rf"##\s+{re.escape(heading)}\s*\n(.*?)(?=\n##\s|\Z)", text, re.S | re.I)
    return m.group(1).strip() if m else ""


def writing_standards(max_chars: int = 3200) -> str:
    """Return the anonymized, distilled first-class writing standards."""
    path = GUIDES_DIR / "first_class_writing_standards.md"
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]


def guide_method(slug: str) -> str:
    """The 'how to structure the answer' part of the guide for this subject."""
    if not slug:
        return ""
    path = GUIDES_DIR / f"{slug}.md"
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    parts = []
    for h in (
        "Answer Method",
        "For Each Topic Include",
        "Case Brief Bank",
        "Full OSCOLA Authority Bank",
        "Current-Law Update Checkpoints",
        "Current SRA Rule Bank — verified 14 July 2026",
        "Strong First-Class Accuracy Pass",
        "Anonymized Quality Upgrade",
        "Anonymized Output Architecture",
        "Avoid",
    ):
        body = _section(text, h)
        if body:
            parts.append(f"{h}:\n{body}")
    title = slug.replace("_", " ").title()
    return f"ANSWER GUIDE — {title} (structure the answer this way):\n" + "\n\n".join(parts) if parts else ""


def guide_methods(slugs: list[str], max_chars: int = 9000) -> str:
    """Combine guides for multi-subject questions within a bounded prompt budget."""
    blocks: list[str] = []
    remaining = max_chars
    for slug in slugs:
        block = guide_method(slug)
        if not block:
            continue
        if len(block) > remaining:
            block = block[:remaining].rsplit("\n", 1)[0]
        if block:
            blocks.append(block)
            remaining -= len(block)
        if remaining <= 300:
            break
    return "\n\n".join(blocks)


def guide_method_for_question(question: str, primary: str | None = None,
                              max_chars: int = 9000) -> str:
    """Return every relevant guide, keeping an explicitly supplied primary first."""
    slugs = detect_subjects(question)
    if primary:
        slugs = [primary] + [slug for slug in slugs if slug != primary]
    return guide_methods(slugs, max_chars=max_chars)


def authority_citation_map_for_question(question: str, primary: str | None = None) -> dict[str, str]:
    """Extract verified full case citations from the relevant guide banks.

    Values are copied verbatim from ``Full OSCOLA Authority Bank`` sections;
    this gives the release pipeline a deterministic way to repair a model that
    names a verified case but forgets the required parenthetical citation.
    """
    slugs = detect_subjects(question)
    if primary:
        slugs = [primary] + [slug for slug in slugs if slug != primary]
    mapping: dict[str, str] = {}
    # Party-v-party and Re-style short titles (Re Rose, Re Baden, etc.).
    pattern = re.compile(
        r"\*((?:Re\s+[^*\n]{2,120}?|[^*\n]{2,180}?\bv\s+[^*\n]{2,180}?))\*\s+"
        r"((?:(?:\[(?:18|19|20)\d{2}\]|\((?:18|19|20)\d{2}\))[^;\n]{0,150}|"
        r"\d+\s+US\s+\d+\s+\((?:18|19|20)\d{2}\)[^;\n]{0,60}))"
    )

    def index_citation(name: str, details: str) -> None:
        details = re.sub(r"\s+", " ", details).strip().rstrip(". ")
        normalized_name = re.sub(r"\s+", " ", name).strip()
        full = f"{normalized_name} {details}"
        key = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
        if not key or not details:
            return
        mapping[key] = full
        # Models commonly omit corporate suffixes while retaining both
        # recognisable party names (for example, "Butler Machine Tool v
        # Ex-Cell-O").  Index that conservative alias to the same
        # verified citation; never guess aliases that drop a party.
        corporate = {
            "co", "company", "corp", "corporation", "inc", "limited", "ltd",
            "llp", "llc", "plc", "gmbh", "ag", "sa", "nv", "england",
        }
        alias = " ".join(token for token in key.split() if token not in corporate)
        if (" v " in f" {alias} " or alias.startswith("re ")) and alias != key:
            mapping.setdefault(alias, full)

        # A model will often use the conventional shortened party
        # names ("Williams v Roffey") rather than reproduce every
        # corporate suffix and descriptor.  Index conservative one-
        # and two-token roots on *both* sides of v.  Keeping both
        # parties is important: a bare surname is too easy to match in
        # ordinary prose and could attach the wrong authority.
        parties = re.split(r"\s+v\s+", alias, maxsplit=1)
        if len(parties) == 2:
            left = [t for t in parties[0].split() if t not in {"the"}]
            right = [t for t in parties[1].split() if t not in {"the"}]
            for left_n in (1, 2):
                for right_n in (1, 2):
                    if len(left) >= left_n and len(right) >= right_n:
                        short = f"{' '.join(left[:left_n])} v {' '.join(right[:right_n])}"
                        if len(short) >= 7:
                            mapping.setdefault(short, full)
        elif alias.startswith("re "):
            # Re Baden's Deed Trusts (No 2) → also index "re baden".
            tokens = [t for t in alias.split()[1:] if t not in {"the", "no", "and"}]
            if tokens:
                mapping.setdefault("re " + tokens[0], full)
                if len(tokens) >= 2:
                    mapping.setdefault("re " + " ".join(tokens[:2]), full)

        # Judicial-review style titles: R (Applicant) v Respondent → also
        # index "applicant v respondent" (models almost never write the R ()).
        jr = re.match(
            r"^(?:r|r on the application of)\s+(.+?)\s+v\s+(.+)$",
            key,
        )
        if jr:
            applicant = jr.group(1).strip()
            respondent = jr.group(2).strip()
            if applicant and respondent:
                mapping.setdefault(f"{applicant} v {respondent}", full)
                app_toks = [t for t in applicant.split() if t not in {"the"}]
                resp_toks = [t for t in respondent.split() if t not in {"the"}]
                if app_toks and resp_toks:
                    mapping.setdefault(
                        f"{app_toks[0]} v {' '.join(resp_toks[:2])}", full
                    )
                    mapping.setdefault(
                        f"{' '.join(app_toks[:2])} v {' '.join(resp_toks[:2])}", full
                    )

        # A few authorities are universally cited by a distinctive
        # conventional short title that omits one party altogether.
        # These aliases are explicit so the repair never guesses.
        conventional = {
            "central london property trust ltd v high trees house ltd": ("high trees",),
            "central london property trust v high trees house": ("high trees",),
            "mcphail v doulton": ("re baden",),
            "r jackson v attorney general": ("jackson v attorney general", "jackson"),
            "r miller v secretary of state for exiting the european union": (
                "miller v secretary of state for exiting the european union",
                "miller no 1",
                "miller (no 1)",
            ),
            "r miller v the prime minister": (
                "miller v the prime minister",
                "miller no 2",
                "miller (no 2)",
                "cherry miller",
            ),
            "r privacy international v investigatory powers tribunal": (
                "privacy international v investigatory powers tribunal",
                "privacy international",
            ),
        }
        for short in conventional.get(key, ()):
            mapping.setdefault(short, full)

    for slug in slugs:
        path = GUIDES_DIR / f"{slug}.md"
        if not path.is_file():
            continue
        bank = _section(path.read_text(encoding="utf-8", errors="ignore"),
                        "Full OSCOLA Authority Bank")
        for name, details in pattern.findall(bank):
            index_citation(name, details)
    return mapping


def supervisor_quality(question: str, slug: str) -> str:
    """Curated quality-control blocks for the supervisor pass (kept within budget)."""
    if not _Q_OK:
        return ""
    blocks = [
        _q.build_reference_citation_policy_block(),
        _q.build_answer_specificity_gate(),
        _q.build_specialist_accuracy_pass_gate(),
        _q.build_topic_marking_rubric_block(slug, topic=""),
    ]
    text = "\n\n".join(b for b in blocks if b and b.strip())
    return text[:5000]  # bound the prompt


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Explain duty of care in negligence and foreseeability"
    slug = detect_subject(q)
    print("question:", q)
    print("detected subject:", slug or "(none)")
    print("\n--- guide method (first 700 chars) ---\n", guide_method(slug)[:700])
    print("\n--- supervisor quality (first 500 chars) ---\n", supervisor_quality(q, slug)[:500])
