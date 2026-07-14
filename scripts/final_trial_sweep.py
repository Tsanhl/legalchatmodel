#!/usr/bin/env python3
"""Final trial sweep: 20 questions across 20 subjects, 1,000..20,000 words, asked through
the running chat UI exactly like a user (POST /api/chat, SSE consumed). Every exchange lands
in the improvement-record folder; a summary line per question is appended to sweep_log.jsonl."""
import json, re, time, urllib.request, pathlib, sys

BASE = "http://127.0.0.1:8765"
OUT = pathlib.Path(__file__).resolve().parents[1] / "user's request record for improvements" / "final_trial_sweep_log.jsonl"

QUESTIONS = [
 (1000,  "contract_law", "essay", "The doctrine of consideration is an outdated technical requirement that English contract law should abandon. Critically discuss with reference to Williams v Roffey, Foakes v Beer and promissory estoppel."),
 (2000,  "tort_law", "problem", "Priya, a junior doctor, misreads a scan and discharges Tom, who suffers a stroke hours later; his wife Una witnesses the collapse and develops PTSD; Tom's employer loses a major contract because Tom cannot work. Advise on negligence liability including duty, breach (Bolam/Bolitho/Montgomery), causation, psychiatric injury and pure economic loss."),
 (3000,  "criminal_law", "problem", "Gus, drunk, throws a bottle in a crowded bar which strikes Hana, who falls and later dies after a mismanaged operation; his friend Ivo had shouted 'go on, throw it'. Meanwhile Hana's brother Jack, hearing of her death and after years of bullying by Gus, waits two hours then attacks Gus with a bat, killing him. Advise all parties on homicide liability, causation, intoxication, accessorial liability, loss of control and diminished responsibility."),
 (4000,  "land_law", "problem", "Kay and Lee bought a house as joint tenants; Kay later wrote to Lee 'I want my half share sorted when we sell'. Lee secretly mortgaged the house, then died leaving his estate to Mia. Meanwhile neighbour Ned has used a track across the garden for 22 years, and tenant Ola claims an overriding interest through actual occupation. Advise on severance, the mortgage's effect, the easement claim and Schedule 3 LRA 2002."),
 (5000,  "trusts_law", "problem", "Asher's will leaves £500,000 'to my executor Ben to distribute among such of my loyal friends as he considers deserving', £100,000 'for the advancement of cricket in Durham', and his house to Ben 'trusting he will let my sister live there for life'. Ben has spent £50,000 of the estate on his own debts, traceable into shares now worth £80,000. Advise on validity of each disposition, certainty of objects, purpose trusts, secret/half-secret trust analysis, breach and tracing."),
 (6000,  "public_law", "essay", "Parliamentary sovereignty remains the fundamental rule of the UK constitution, but it is now qualified by the rule of law. Critically discuss with reference to the Human Rights Act 1998, EU membership and withdrawal, Jackson, Miller (No 1) and (No 2), Privacy International and devolution."),
 (7000,  "human_rights_law", "essay", "The Human Rights Act 1998 strikes a defensible balance between parliamentary sovereignty and effective rights protection. Critically discuss with reference to ss 2, 3, 4 and 6 HRA, the margin of appreciation, proportionality, and proposals for reform."),
 (8000,  "eu_law", "essay", "Direct effect and supremacy transformed the EU legal order, and their loss transforms UK law after withdrawal. Critically discuss with reference to Van Gend en Loos, Costa v ENEL, Factortame, and the status of assimilated law in the UK."),
 (9000,  "business_law", "problem", "Dana is a director of Retail plc who diverted a corporate opportunity to her own company, approved accounts hiding losses, and continued trading while the company was hopelessly insolvent; the company is now in liquidation. Advise the liquidator on directors' duties under ss 171-177 CA 2006, remedies, wrongful trading under s 214 IA 1986, and lifting the corporate veil."),
 (10000, "employment_law", "problem", "Ravi worked via an app for five years under a contract labelling him self-employed; the app sets prices, uniforms and routes. He was 'deactivated' after raising safety complaints and organising a drivers' association. Advise on worker/employee status (Autoclenz, Uber), unfair dismissal, whistleblowing protection, and trade-union detriment."),
 (11000, "family_law", "problem", "After a 20-year marriage, Sam (a surgeon) and Tia (who gave up her career for childcare) divorce. Assets: £2m house, £1.5m pension, £800k business Sam built, and a £300k inheritance Tia received last year. Sam hid a £200k bonus during negotiations. Advise on financial remedies: needs/sharing/compensation, matrimonial vs non-matrimonial property, pension sharing, conduct and non-disclosure."),
 (12000, "evidence_law", "essay", "The rules governing hearsay and bad character evidence under the Criminal Justice Act 2003 achieve a workable balance between probative value and fairness. Critically discuss with reference to the statutory gateways, judicial discretion, and Article 6 ECHR."),
 (13000, "commercial_law", "problem", "Seller ships 10,000 units of electronics to Buyer under CIF terms; the goods are damaged at sea, the bill of lading was altered, and Seller had already sold the same consignment to another buyer. A finance house holds a Romalpa clause over Buyer's stock. Advise on passing of property and risk, nemo dat exceptions, documentary duties under CIF, and retention of title."),
 (14000, "intellectual_property_law", "essay", "Copyright law's response to generative AI is inadequate: neither training on protected works nor AI outputs are satisfactorily governed. Critically discuss under UK law, including originality, the TDM exception debate, authorship of computer-generated works (s 9(3) CDPA), and infringement."),
 (15000, "competition_law", "problem", "MegaCorp holds 55% of the UK cloud services market, prices below cost when rivals enter, bundles storage with mandatory analytics, and has agreed with its two largest competitors to align list prices. Advise under Chapters I and II CA 1998 / Articles 101 and 102 TFEU analogues, including market definition, abuse, objective justification, and penalties."),
 (16000, "environmental_law", "essay", "UK environmental law after Brexit relies too heavily on aspirational targets and too little on enforceable duties. Critically discuss with reference to the Environment Act 2021, the Office for Environmental Protection, judicial review of environmental decisions, and private-law actions in nuisance."),
 (17000, "immigration_refugee_law", "essay", "The UK's asylum framework increasingly prioritises deterrence over protection. Critically discuss with reference to the Refugee Convention, the Nationality and Borders Act 2022, the Illegal Migration Act 2023, safe-third-country policies and Article 3 ECHR limits."),
 (18000, "tax_law", "essay", "The UK's approach to tax avoidance — from the Ramsay principle to the GAAR — shows the limits of judicial and statutory anti-avoidance. Critically discuss."),
 (19000, "jurisprudence_law", "essay", "Hart's rule of recognition explains legal validity better than Dworkin's interpretivism. Critically discuss, using UK constitutional practice as your testing ground."),
 (20000, "restitution_law", "essay", "The law of unjust enrichment is now a coherent, independent branch of English private law. Critically discuss with reference to the unjust factors, defences including change of position, and proprietary restitution."),
]

def post_chat(conv_id, message):
    payload = json.dumps({"conversation_id": conv_id, "message": message,
                          "jurisdiction": "england_wales", "online_mode": "auto"}).encode()
    req = urllib.request.Request(BASE + "/api/chat", data=payload,
                                 headers={"Content-Type": "application/json"})
    final_len = 0
    with urllib.request.urlopen(req, timeout=14400) as resp:  # up to 4h per question
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if line.startswith("data: "):
                try:
                    obj = json.loads(line[6:])
                except Exception:
                    continue
                if "delta" in obj:
                    final_len += len(obj["delta"])
                if obj.get("done"):
                    break
    return final_len

def new_conv():
    req = urllib.request.Request(BASE + "/api/conversations", data=b"{}",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["id"]

def main():
    start_at = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    for i, (words, subject, register, stem) in enumerate(QUESTIONS):
        if i < start_at:
            continue
        q = (f"Assume English law. {'Essay question' if register=='essay' else 'Problem question'} "
             f"({subject.replace('_',' ').title()}), suggested length: {words:,} words. {stem}")
        t0 = time.time()
        status = "ok"
        try:
            conv = new_conv()
            chars = post_chat(conv, q)
        except Exception as exc:
            status, chars = f"error: {exc}", 0
        rec = {"i": i, "words_asked": words, "subject": subject, "register": register,
               "chars_streamed": chars, "minutes": round((time.time()-t0)/60, 1), "status": status}
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
        time.sleep(10)

if __name__ == "__main__":
    main()
