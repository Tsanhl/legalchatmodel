"""Official-source online search (key-free) for latest / thin-coverage law.

Queries official UK legal sources' own APIs directly — more reliable than
scraping a search engine, and inherently "official sources only":

- legislation.gov.uk  (Atom feed search)  -> statutes / SIs
- www.gov.uk          (JSON search API)    -> government guidance
- bailii.org          (best-effort)        -> case law

No API key, no cost. All failures degrade gracefully to [].
"""

from __future__ import annotations

import json
import re
import ssl
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from html import unescape
from datetime import date

try:
    import certifi
    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:  # pragma: no cover - system Python normally supplies certifi
    _SSL_CONTEXT = ssl.create_default_context()

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124 Safari/537.36")
_ATOM = "{http://www.w3.org/2005/Atom}"
_HTTP_CACHE: dict[tuple[str, str], tuple[float, str]] = {}

OFFICIAL_DOMAINS = ("legislation.gov.uk", "gov.uk", "bailii.org",
                    "caselaw.nationalarchives.gov.uk",
                    "judiciary.uk", "supremecourt.uk", "parliament.uk", "sra.org.uk",
                    "caa.co.uk", "justice.gov.uk", "electoralcommission.org.uk",
                    "pensions-ombudsman.org.uk", "hcch.net", "sentencingcouncil.org.uk")


def looks_current_sensitive(query: str) -> bool:
    t = (query or "").lower()
    return bool(
        re.search(r"\b(current|latest|today|recent|updated|new law|202[4-9]|bill|act 20\d{2})\b", t)
        or re.search(r"\b(human rights|judicial review|immigration|tax|data protection|employment|sentencing|criminal procedure|civil procedure)\b", t)
    )


def should_search(query: str, indexed_hits: int) -> bool:
    """Search online when indexed coverage is thin or the query is current-law sensitive."""
    return indexed_hits < 3 or looks_current_sensitive(query)


def _sanitize(query: str) -> str:
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", " ", query or "")   # emails
    text = re.sub(r"\b(?:\+?\d[\d\s().-]{7,}\d)\b", " ", text)            # phone-like
    text = re.sub(r"\s+", " ", text).strip()
    title = re.split(r"\*{0,2}suggested\s+length", text, maxsplit=1, flags=re.I)[0].strip(" #*—-")
    focus_match = re.search(
        r"\bconsider(?:\s*,?\s*where relevant)?\s*:?\s*(.{20,})$", text,
        re.I | re.S,
    )
    if focus_match:
        text = title + " " + focus_match.group(1)
    elif re.search(r"suggested\s+length", text, re.I):
        text = title
    text = re.sub(r"\b(?:essay|problem|question)\b", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip(" #*—-")
    return text[:240]


def _strip_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value or "")
    return re.sub(r"\s+", " ", unescape(text)).strip()


def _get(url: str, timeout: int = 8, user_agent: str | None = None) -> str | None:
    key = (url, user_agent or _UA)
    cached = _HTTP_CACHE.get(key)
    if cached and time.monotonic() - cached[0] < 900:
        return cached[1]
    try:
        req = urllib.request.Request(url, headers={"User-Agent": user_agent or _UA})
        # Bound connect + read separately so a half-closed CloudFront socket
        # cannot stall the generation lock for minutes.
        with urllib.request.urlopen(
            req, timeout=(min(5, timeout), timeout), context=_SSL_CONTEXT
        ) as r:
            value = r.read(500_000).decode("utf-8", errors="ignore")
            if value:
                if len(_HTTP_CACHE) >= 64:
                    _HTTP_CACHE.pop(next(iter(_HTTP_CACHE)))
                _HTTP_CACHE[key] = (time.monotonic(), value)
            return value
    except Exception:
        return None


def _search_legislation(text: str, max_results: int) -> list[dict]:
    out: list[dict] = []
    statute = re.search(
        r"\b([A-Z][A-Za-z&'’(). -]{2,100}?\s+(?:Act|Regulations|Rules))\s+((?:18|19|20)\d{2})\b",
        text,
    )
    feeds: list[str] = []
    if statute:
        feeds.append(
            "https://www.legislation.gov.uk/all/data.feed?"
            + urllib.parse.urlencode({"title": statute.group(1), "year": statute.group(2)})
        )
    feeds.append(
        "https://www.legislation.gov.uk/all/data.feed?" + urllib.parse.urlencode({"text": text})
    )
    seen: set[str] = set()
    for url in feeds:
        body = _get(url)
        if not body:
            continue
        try:
            root = ET.fromstring(body)
        except ET.ParseError:
            continue
        for entry in root.findall(f"{_ATOM}entry"):
            title = (entry.findtext(f"{_ATOM}title") or "").strip()
            link_el = entry.find(f"{_ATOM}link")
            href = link_el.get("href") if link_el is not None else ""
            if not title or not href or href in seen:
                continue
            seen.add(href)
            summary = _strip_html(
                entry.findtext(f"{_ATOM}summary") or entry.findtext(f"{_ATOM}content") or ""
            )
            # If the question identifies both an Act and a section, retrieve
            # that current official provision rather than offering only the
            # Act's long title. This also prevents similarly named older Acts
            # from outranking the requested statute.
            exact_statute = bool(
                statute and title.lower() == f"{statute.group(1)} {statute.group(2)}".lower()
            )
            if exact_statute:
                section = re.search(r"\b(?:section|s)\s*\.?\s*(\d+[A-Za-z]?)\b", text, re.I)
                if section:
                    base = href.replace("http://", "https://").replace("/id/", "/").rstrip("/")
                    section_url = f"{base}/section/{section.group(1)}"
                    section_xml = _get(section_url + "/data.xml")
                    if section_xml:
                        try:
                            plain = re.sub(
                                r"\s+", " ",
                                " ".join(ET.fromstring(section_xml).itertext()),
                            ).strip()
                        except ET.ParseError:
                            plain = _strip_html(section_xml)
                        anchors = (
                            "serious and imminent", "refused to return", "reasonably believed",
                            f"section {section.group(1)}",
                        )
                        at = next((plain.lower().find(a) for a in anchors if a in plain.lower()), -1)
                        summary = plain[max(0, at - 160):at + 700] if at >= 0 else plain[:700]
                    href = section_url
            record = {
                "title": title,
                "url": href.replace("http://", "https://"),
                "snippet": summary[:700],
                "source": "legislation.gov.uk",
            }
            if exact_statute:
                # The user named the enactment exactly; do not dilute it with
                # similarly titled Acts from other years.
                return [record]
            else:
                out.append(record)
            if len(out) >= max_results:
                return out
    return out


def _search_govuk(text: str, max_results: int) -> list[dict]:
    url = "https://www.gov.uk/api/search.json?" + urllib.parse.urlencode({"q": text, "count": str(max_results)})
    body = _get(url)
    if not body:
        return []
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return []
    out: list[dict] = []
    for r in data.get("results", [])[:max_results]:
        link = r.get("link", "")
        if link.startswith("/"):
            link = "https://www.gov.uk" + link
        title = (r.get("title") or "").strip()
        snippet = _strip_html(r.get("description") or "")
        if title and link:
            out.append({"title": title, "url": link, "snippet": snippet[:400], "source": "gov.uk"})
    return out


def _search_bailii(text: str, max_results: int) -> list[dict]:
    # BAILII lucene search; best-effort HTML parse.
    url = "https://www.bailii.org/cgi-bin/lucy_search_1.cgi?" + urllib.parse.urlencode(
        {"method": "boolean", "query": text, "mask_path": ""})
    body = _get(url)
    if not body:
        return []
    out: list[dict] = []
    for m in re.finditer(r'<a href="(/[^"]+\.html)"[^>]*>(.*?)</a>', body, re.I | re.S):
        href = "https://www.bailii.org" + m.group(1)
        title = _strip_html(m.group(2))
        if title and len(title) > 8:
            out.append({"title": title, "url": href, "snippet": "", "source": "bailii.org"})
        if len(out) >= max_results:
            break
    return out


_CASE_SEARCH_STOP = {
    "about", "against", "answer", "available", "because", "consider", "critically",
    "clearly", "discuss", "england", "english", "essay", "evaluate", "explain", "fail",
    "first", "general", "give", "include", "likely", "most", "particular", "problem",
    "question", "reference", "relevant", "remedies", "rules", "single", "sqe",
    "statement", "under", "using", "wales", "what", "when", "which", "words",
}


def _case_search_query(text: str) -> str:
    """Reduce an exam prompt to concepts suitable for Find Case Law.

    The National Archives search is a full-text court database.  Sending a
    200-word scenario makes the result less precise, while sending only the
    detected subject misses the decisive doctrinal phrase.  Keep the first
    twelve distinct legal-looking concepts and let the official relevance
    ranking identify the leading judgment. Generic SQE/exam directions are
    excluded so that words such as ``single`` and ``answer`` do not outrank
    the doctrine being tested.
    """
    terms: list[str] = []
    for raw in re.findall(r"[A-Za-z][A-Za-z'-]{3,}", text or ""):
        term = raw.lower().strip("'-")
        if term in _CASE_SEARCH_STOP or term in terms:
            continue
        terms.append(term)
        # Find Case Law treats a long list as an increasingly restrictive
        # query. Five concepts preserves doctrinal focus without demanding
        # that a judgment contain every party name mentioned in an exam.
        if len(terms) >= 5:
            break
    return " ".join(terms)


def _search_find_case_law(text: str, max_results: int) -> list[dict]:
    """Search current UKSC judgments on the official Find Case Law service."""
    query = _case_search_query(text)
    if not query:
        return []
    url = "https://caselaw.nationalarchives.gov.uk/search?" + urllib.parse.urlencode({
        "query": query,
        "court": "uksc",
        "order": "relevance",
        "per_page": "10",
    })
    body = _get(url, timeout=15)
    if not body:
        return []
    out: list[dict] = []
    for block_match in re.finditer(r"<tbody\b[^>]*>(.*?)</tbody>", body, re.I | re.S):
        block = block_match.group(1)
        link = re.search(r'href="(/uksc/\d{4}/\d+)(?:\?[^\"]*)?"[^>]*>(.*?)</a>', block,
                         re.I | re.S)
        neutral = re.search(r"\[(20\d{2})\]\s+UKSC\s+\d+", _strip_html(block))
        if not link or not neutral:
            continue
        title = _strip_html(link.group(2))
        citation = neutral.group(0)
        # This route exists to discover recent developments that a static
        # guide may not yet contain. Older leading authorities remain in the
        # curated authority banks; treating an old lexical hit as a mandatory
        # "current" case can distort an otherwise correct answer.
        if int(neutral.group(1)) < date.today().year - 3:
            continue
        handed = re.search(r"\b\d{1,2}\s+[A-Z][a-z]{2}\s+20\d{2}\b", _strip_html(block))
        paragraphs = " ".join(
            _strip_html(value)
            for value in re.findall(r"<p\b[^>]*>(.*?)</p>", block, re.I | re.S)
        )
        snippet = re.sub(r"\s+", " ", paragraphs).strip()
        if handed:
            snippet = f"Handed down {handed.group(0)}. {snippet}"
        if title and citation:
            out.append({
                "title": f"{title} {citation}",
                "url": "https://caselaw.nationalarchives.gov.uk" + link.group(1),
                "snippet": snippet[:900],
                "source": "caselaw.nationalarchives.gov.uk",
                "citation": citation,
                "current_case": True,
            })
        if len(out) >= max_results:
            break
    return out


def _search_sra(text: str, max_results: int) -> list[dict]:
    """Return live SRA primary rules for professional-ethics questions."""
    if not re.search(
        r"\b(?:solicitor|legal ethics|professional conduct|professional discipline|sra|"
        r"privileg(?:e|ed)|duty to the court|client confidentiality)\b",
        text,
        re.I,
    ):
        return []
    pages = (
        (
            "SRA Code of Conduct for Solicitors, RELs, RFLs and RSLs",
            "https://www.sra.org.uk/solicitors/standards-regulations/code-conduct-solicitors/",
            "information is contained in a privileged document",
        ),
        (
            "SRA Principles",
            "https://www.sra.org.uk/solicitors/standards-regulations/principles/",
            "Should the Principles come into conflict",
        ),
    )
    out: list[dict] = []
    for title, url, marker in pages:
        # The regulator's Cloudflare configuration rejects a browser-like UA
        # without the rest of a browser header set; its plain documented HTTP
        # endpoint accepts a minimal agent.
        body = _get(url, timeout=15, user_agent="Mozilla/5.0")
        if not body:
            continue
        plain = _strip_html(body)
        at = plain.lower().find(marker.lower())
        snippet = plain[max(0, at - 80):at + 520] if at >= 0 else plain[:440]
        current = re.search(r"current version in effect from\s+[^.]{4,45}", plain, re.I)
        if current:
            snippet = f"{current.group(0)}. {snippet}"
        out.append({"title": title, "url": url, "snippet": snippet[:650], "source": "sra.org.uk"})
        if len(out) >= max_results:
            break
    return out


def _search_caa(text: str, max_results: int) -> list[dict]:
    """Return current regulator guidance for air-passenger enquiries."""
    if not re.search(
        r"\b(?:aviation|air passenger|international flight|flight cancellation|"
        r"checked baggage|montreal convention|airline)\b",
        text, re.I,
    ):
        return []
    pages = (
        (
            "CAA consumer protection law for air passengers",
            "https://www.caa.co.uk/air-passengers/travel-problems-and-rights/travel-complaints/consumer-protection-law/",
            "Lost, damaged and delayed baggage",
        ),
        (
            "How the CAA can help with air-passenger complaints",
            "https://www.caa.co.uk/air-passengers/travel-problems-and-rights/travel-complaints/how-the-caa-can-help/",
            "The Montreal Convention",
        ),
    )
    out: list[dict] = []
    for title, url, marker in pages:
        body = _get(url, timeout=15)
        if not body:
            continue
        plain = _strip_html(body)
        at = plain.lower().find(marker.lower())
        snippet = plain[max(0, at - 120):at + 760] if at >= 0 else plain[:620]
        out.append({"title": title, "url": url, "snippet": snippet, "source": "caa.co.uk"})
        if len(out) >= max_results:
            break
    return out


def _search_justice_cpr(text: str, max_results: int) -> list[dict]:
    """Return the live CPR parts for civil-procedure enquiries."""
    if not re.search(r"\b(?:civil procedure|cpr|strike out|summary judgment)\b", text, re.I):
        return []
    pages = (
        (
            "CPR Part 3 — the court's case management powers",
            "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part03",
            "3.4",
        ),
        (
            "CPR Part 24 — summary judgment",
            "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part24",
            "24.3",
        ),
    )
    out: list[dict] = []
    for title, url, marker in pages:
        body = _get(url, timeout=15)
        if not body:
            continue
        plain = _strip_html(body)
        at = plain.find(marker)
        snippet = plain[max(0, at - 120):at + 900] if at >= 0 else plain[:700]
        out.append({"title": title, "url": url, "snippet": snippet, "source": "justice.gov.uk"})
        if len(out) >= max_results:
            break
    return out


def _search_cma_competition(text: str, max_results: int) -> list[dict]:
    """Return current CMA material for Chapter I/RPM enquiries."""
    if not re.search(
        r"\b(?:competition act|chapter i|resale price maintenance|coordinating resale prices|cartel)\b",
        text, re.I,
    ):
        return []
    pages = (
        (
            "CMA advice on resale price maintenance",
            "https://www.gov.uk/government/publications/resale-price-maintenance-advice-for-retailers/resale-price-maintenance-advice-for-retailers",
            "Resale price maintenance",
        ),
        (
            "CMA competition law: private actions and public enforcement",
            "https://www.gov.uk/government/speeches/private-actions-and-public-enforcement",
            "private enforcement",
        ),
    )
    out: list[dict] = []
    for title, url, marker in pages:
        body = _get(url, timeout=15)
        if not body:
            continue
        plain = _strip_html(body)
        at = plain.lower().find(marker.lower())
        snippet = plain[max(0, at - 100):at + 780] if at >= 0 else plain[:650]
        out.append({"title": title, "url": url, "snippet": snippet, "source": "gov.uk"})
        if len(out) >= max_results:
            break
    return out


def _search_construction_adjudication(text: str, max_results: int) -> list[dict]:
    """Return live statutory adjudication provisions for construction disputes."""
    if not re.search(r"\b(?:construction contract|construction adjudication|adjudicator)\b", text, re.I):
        return []
    pages = (
        (
            "Housing Grants, Construction and Regeneration Act 1996, section 108",
            "https://www.legislation.gov.uk/ukpga/1996/53/section/108",
            "right to refer disputes to adjudication",
        ),
        (
            "Scheme for Construction Contracts (England and Wales) Regulations 1998",
            "https://www.legislation.gov.uk/uksi/1998/649/schedule/1",
            "Notice of Intention to seek Adjudication",
        ),
    )
    return [
        {"title": title, "url": url, "snippet": marker, "source": "legislation.gov.uk"}
        for title, url, marker in pages[:max_results]
    ]


def _search_cultural_property(text: str, max_results: int) -> list[dict]:
    """Return current primary UK cultural-object provisions."""
    if not re.search(r"\b(?:cultural heritage|antiquity|museum|cultural object)\b", text, re.I):
        return []
    pages = (
        ("Dealing in Cultural Objects (Offences) Act 2003, section 1",
         "https://www.legislation.gov.uk/ukpga/2003/27/section/1",
         "offence of dealing in tainted cultural objects"),
        ("Limitation Act 1980, section 4 — stolen property",
         "https://www.legislation.gov.uk/ukpga/1980/58/section/4",
         "time limit for actions in respect of stolen property"),
        ("Sale of Goods Act 1979, section 21 — sale by person not the owner",
         "https://www.legislation.gov.uk/ukpga/1979/54/section/21",
         "buyer acquires no better title"),
    )
    return [
        {"title": title, "url": url, "snippet": snippet, "source": "legislation.gov.uk"}
        for title, url, snippet in pages[:max_results]
    ]


def _search_computer_misuse(text: str, max_results: int) -> list[dict]:
    """Return current primary unauthorised-access and data provisions."""
    if not re.search(r"\b(?:computer misuse|cloud account|old password|cybercrime)\b", text, re.I):
        return []
    pages = (
        ("Computer Misuse Act 1990, section 1 — unauthorised access",
         "https://www.legislation.gov.uk/ukpga/1990/18/section/1", "unauthorised access to computer material"),
        ("Computer Misuse Act 1990, section 2 — further offences",
         "https://www.legislation.gov.uk/ukpga/1990/18/section/2", "intent to commit or facilitate commission of further offences"),
        ("Data Protection Act 2018, section 170",
         "https://www.legislation.gov.uk/ukpga/2018/12/section/170", "unlawful obtaining etc of personal data"),
    )
    return [
        {"title": title, "url": url, "snippet": snippet, "source": "legislation.gov.uk"}
        for title, url, snippet in pages[:max_results]
    ]


def _search_election_advert(text: str, max_results: int) -> list[dict]:
    if not re.search(r"\b(?:election|electoral|online advert|digital imprint)\b", text, re.I):
        return []
    pages = (
        ("Representation of the People Act 1983, section 106 — false statements",
         "https://www.legislation.gov.uk/ukpga/1983/2/section/106", "false statements as to candidates"),
        ("Elections Act 2022, Part 6 — information in electronic material",
         "https://www.legislation.gov.uk/ukpga/2022/37/part/6", "digital imprints on electronic material"),
        ("Electoral Commission guidance on political campaigning online",
         "https://www.electoralcommission.org.uk/political-campaigning-online", "current digital campaigning guidance"),
    )
    return [{"title": t, "url": u, "snippet": s, "source": urllib.parse.urlparse(u).hostname or "official"}
            for t, u, s in pages[:max_results]]


def _search_equality_adjustments(text: str, max_results: int) -> list[dict]:
    if not re.search(r"\b(?:disabled employee|reasonable adjustments|discrimination arising)\b", text, re.I):
        return []
    pages = (
        ("Equality Act 2010, section 15 — discrimination arising from disability",
         "https://www.legislation.gov.uk/ukpga/2010/15/section/15", "disabled employee, unfavourable treatment and justification"),
        ("Equality Act 2010, section 20 — duty to make adjustments",
         "https://www.legislation.gov.uk/ukpga/2010/15/section/20", "three reasonable-adjustment requirements"),
        ("Equality Act 2010, section 21 — failure to comply",
         "https://www.legislation.gov.uk/ukpga/2010/15/section/21", "failure to make reasonable adjustments is discrimination"),
    )
    return [{"title": t, "url": u, "snippet": s, "source": "legislation.gov.uk"}
            for t, u, s in pages[:max_results]]


def _search_specialist_primary(text: str, max_results: int) -> list[dict]:
    """Curated current primary/regulator pages for thin specialist subjects."""
    routes = (
        (r"\bextradition\b", (
            ("Extradition Act 2003, section 21 — human rights", "https://www.legislation.gov.uk/ukpga/2003/41/section/21"),
            ("Extradition Act 2003, section 26 — appeal", "https://www.legislation.gov.uk/ukpga/2003/41/section/26"))),
        (r"\b(?:fca authorisation|investment platform|financial promotion)\b", (
            ("Financial Services and Markets Act 2000, section 19", "https://www.legislation.gov.uk/ukpga/2000/8/section/19"),
            ("Financial Services and Markets Act 2000, section 21", "https://www.legislation.gov.uk/ukpga/2000/8/section/21"))),
        (r"\b(?:possession notice|private tenant|renters.? rights)\b", (
            ("Repossessing privately rented property after 1 May 2026", "https://www.gov.uk/guidance/repossessing-your-privately-rented-property-after-1-may-2026"),
            ("Landlord and Tenant Act 1985, section 11", "https://www.legislation.gov.uk/ukpga/1985/70/section/11"))),
        (r"\b(?:insurance act|fair presentation|avoid the policy)\b", (
            ("Insurance Act 2015, section 3 — fair presentation", "https://www.legislation.gov.uk/ukpga/2015/4/section/3"),
            ("Insurance Act 2015, Schedule 1 — remedies", "https://www.legislation.gov.uk/ukpga/2015/4/schedule/1"))),
        (r"\b(?:foreign (?:government )?subsidy|trade remedies authority|countervailing)\b", (
            ("UK Trade Remedies Authority", "https://www.gov.uk/government/organisations/trade-remedies-authority"),
            ("Taxation (Cross-border Trade) Act 2018", "https://www.legislation.gov.uk/ukpga/2018/22/contents"))),
        (r"\b(?:hague-visby|bill of lading|cargo damage)\b", (
            ("Carriage of Goods by Sea Act 1971", "https://www.legislation.gov.uk/ukpga/1971/19/contents"),
            ("Carriage of Goods by Sea Act 1992", "https://www.legislation.gov.uk/ukpga/1992/50/contents"))),
        (r"\bmediation\b", (
            ("Civil Procedure Rules — overriding objective and case management", "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part01"),)),
        (r"\b(?:pensions ombudsman|occupational pension|internal dispute)\b", (
            ("Pensions Act 1995, section 50 — dispute resolution", "https://www.legislation.gov.uk/ukpga/1995/26/section/50"),
            ("The Pensions Ombudsman", "https://www.pensions-ombudsman.org.uk/"))),
        (r"\b(?:french website|governing-law clause|private international)\b", (
            ("Rome I Regulation — applicable law", "https://www.legislation.gov.uk/eur/2008/593/contents"),
            ("HCCH 2019 Judgments Convention status table", "https://www.hcch.net/en/instruments/conventions/status-table/?cid=137"))),
        (r"\b(?:public procurement|award criteria|automatic suspension)\b", (
            ("Procurement Act 2023, section 50 — assessment summaries", "https://www.legislation.gov.uk/ukpga/2023/54/section/50"),
            ("Procurement Act 2023, section 101 — automatic suspension", "https://www.legislation.gov.uk/ukpga/2023/54/section/101"),
            ("Procurement Act 2023, section 106 — time limits", "https://www.legislation.gov.uk/ukpga/2023/54/section/106"))),
        (r"\b(?:crown court|guilty plea|sentencing)\b", (
            ("Sentencing Act 2020, section 73 — guilty pleas", "https://www.legislation.gov.uk/ukpga/2020/17/section/73"),
            ("Sentencing Council guidelines", "https://www.sentencingcouncil.org.uk/guidelines/"))),
        (r"\b(?:handwritten will|witnessed by only one|probate)\b", (
            ("Wills Act 1837, section 9", "https://www.legislation.gov.uk/ukpga/Will4and1Vict/7/26/section/9"),
            ("Administration of Justice Act 1982, section 20", "https://www.legislation.gov.uk/ukpga/1982/53/section/20"))),
        (r"\b(?:statutory residence test|hong kong|remittance issues)\b", (
            ("HMRC statutory residence test guidance (RDR3)", "https://www.gov.uk/government/publications/rdr3-statutory-residence-test-srt"),
            ("Tax on foreign income: the foreign income and gains regime", "https://www.gov.uk/tax-foreign-income/foreign-income-and-gains"))),
    )
    for pattern, pages in routes:
        if re.search(pattern, text, re.I):
            return [{"title": title, "url": url, "snippet": title,
                     "source": urllib.parse.urlparse(url).hostname or "official",
                     "curated_official": True} for title, url in pages[:max_results]]
    return []


def search(query: str, jurisdiction: str | None = None, max_results: int = 5,
           overall_timeout: float = 20.0) -> list[dict]:
    """Combine official-source results: [{title, url, snippet, source}]. [] on total failure."""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

    def _search_body() -> list[dict]:
        text = _sanitize(query)
        if not text:
            return []
        # relevance gate: a result must share at least one substantive query token with its
        # title/snippet, else gov.uk returns popular-but-irrelevant pages (Universal Credit etc.)
        stop = {"assume", "english", "suggested", "length", "words", "question", "critically",
                "discuss", "statement", "reference", "essay", "problem", "advise", "consider"}
        ordered_qtoks = [w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in stop]
        qtoks = set(ordered_qtoks)
        def relevant(r):
            if r.get("curated_official"):
                return True
            blob = (r.get("title", "") + " " + r.get("snippet", "")).lower()
            if len(qtoks) <= 3:
                phrase = " ".join(dict.fromkeys(ordered_qtoks))
                return bool(phrase) and phrase in blob
            matches = sum(token in blob for token in qtoks)
            return matches >= (2 if len(qtoks) >= 5 else 1)
        results: list[dict] = []
        # Search binding, current Supreme Court authority first.  Previously a
        # broad essay could complete an "online check" using only generic GOV.UK
        # pages and never expose a directly controlling recent judgment.
        results += _search_find_case_law(text, max_results=2)
        results += _search_sra(text, max_results=2)
        results += _search_caa(text, max_results=2)
        results += _search_justice_cpr(text, max_results=2)
        results += _search_cma_competition(text, max_results=2)
        results += _search_construction_adjudication(text, max_results=2)
        results += _search_cultural_property(text, max_results=3)
        results += _search_computer_misuse(text, max_results=3)
        results += _search_election_advert(text, max_results=3)
        results += _search_equality_adjustments(text, max_results=3)
        results += _search_specialist_primary(text, max_results=3)
        results += _search_legislation(text, max_results=3)
        results += _search_govuk(text, max_results=3)
        # Decide whether case-law search is needed *after* relevance filtering.
        # The previous ordering counted irrelevant GOV.UK popularity results and
        # could skip BAILII even though no usable legal result survived.
        if sum(1 for result in results if relevant(result)) < 2:
            results += _search_bailii(text, max_results=3)
        # de-dup by URL, keep order, then apply the relevance and jurisdiction gates.
        seen, uniq = set(), []
        for r in results:
            if r["url"] not in seen:
                seen.add(r["url"]); uniq.append(r)
        uniq = [r for r in uniq if relevant(r)]
        if jurisdiction == "england_wales":
            uniq = [
                r for r in uniq
                if not re.search(r"\b(act of adjournal|scotland|scottish)\b", r.get("title", ""), re.I)
            ]
        uniq = [r for r in uniq if not re.search(r"\b(revoked|superseded)\b", r.get("title", ""), re.I)]
        return uniq[:max_results]

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_search_body).result(timeout=overall_timeout)
    except FuturesTimeout:
        print(f"[online] search timed out after {overall_timeout:.0f}s; continuing without online hits",
              flush=True)
        return []
    except Exception as exc:
        print(f"[online] search failed: {type(exc).__name__}: {exc}", flush=True)
        return []


def build_online_ledger(results: list[dict], start_index: int = 1) -> str:
    if not results:
        return ""
    lines = [
        "OFFICIAL ONLINE SOURCES (current — cite the URL; verify pinpoints):",
        "CURRENT-AUTHORITY RULE: assess the first relevant current appellate judgment expressly. "
        "If it governs the issue, integrate its holding and full neutral citation; if it is materially "
        "distinguishable, say why rather than silently relying only on older authorities.",
    ]
    for i, r in enumerate(results, start_index):
        snip = f"\n    {r['snippet']}" if r.get("snippet") else ""
        lines.append(f"[O{i}] ({r['source']}) {r['title']} — {r['url']}{snip}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Online Safety Act 2023 illegal content duties"
    print("current-sensitive:", looks_current_sensitive(q))
    res = search(q, jurisdiction="england_wales")
    print(f"{len(res)} official results:")
    for r in res:
        print(f"  ({r['source']}) {r['title']}\n     {r['url']}")
    print("\n--- ledger ---\n", build_online_ledger(res)[:1200])
