# Legal AI readiness report

Date: 14 July 2026

## Release state

- Local site: `http://127.0.0.1:8765/`
- Deployed adapter: `legal_answer_flow_v11_specialist_lora`
- Default jurisdiction: England and Wales
- Default online mode: always check current official sources
- Default citation mode: full verified OSCOLA in parentheses immediately after the supported proposition. Essays/problems add one used-authority-only References section; practical general enquiries and SQE answers omit it unless requested.
- Accepted requested range in the supplied bank: 1,000–20,000 body words

## Complete-answer and word-count contract

- The browser receives no provisional answer prose. Generation units, retries and rejected drafts stay internal; one atomic replacement event publishes the completed supervised answer.
- A requested body length is enforced from `ceil(0.99 × requested)` to `floor(1.01 × requested)`. The final References section is outside that count.
- Requests up to 2,500 words use one complete generation; requests above 2,500 words are divided internally into analytical units of no more than 800 words. A 20,000-word answer therefore uses 25 internal units but appears as one continuous answer.
- Essays and problem questions require explicit `### Introduction` and `### Conclusion` headings. Problem introductions state the decisions to be made; essays begin with a qualified thesis. General enquiries and SQE answers have separate formats.
- Every internal long-form unit is checked for substantive depth, repetition, full parenthetical OSCOLA and its required opening/closing role before assembly.
- Named case/statute propositions without an immediately following full parenthetical citation are rejected. Unverified neutral citations and pinpoints are removed; an invented/nonexistent authority is a release-blocking error.
- If generation or transport fails, no partial answer is saved. The browser can recover the durable completed answer after an interrupted final event, but it cannot substitute an earlier answer.

## Memory and privacy

- Memory chats may retrieve relevant user-authored history and durable preferences from other completed Memory chats.
- Prior assistant legal prose, incomplete chats, deleted chats and Private chats are not injected as cross-chat memory.
- Private chats are isolated from memory and improvement/training records. Permanent deletion removes messages, attachments and stored uploads and truncates the SQLite WAL.
- Upload, indexed-document, marked-work and writing-guidance filenames never appear in public source chips. Only official online sources can be shown there.
- Output gates reject candidate identifiers, `[student]`, local paths, private filenames, `indexed` labels and `writing guidance` labels.

## Marked-work and writing-standard use

- All 18 supplied marked-work/guidance documents remain feedback-indexed.
- 70+ prose may be used as a local style exemplar. Lower-mark prose is retained only for diagnostic analysis and is excluded from runtime exemplars and assistant training targets.
- Marker feedback informed the 70+ structure: exact issue/thesis focus, authority hierarchy, immediate fact application, counterargument, calibrated outcome, practical remedy and precise OSCOLA.
- The writing standard now reflects the December 2025 OSCOLA checklist, including the explicit nonexistent-authority hazard.
- Subject prompts now include their stored Full OSCOLA Authority Banks and Current-Law Update Checkpoints; these sections existed on disk but were previously omitted from runtime prompt assembly.

## V11 specialist training selection

- V11 continues conservatively from V10 and adds 35 reviewed specialist general-enquiry and SQE targets while preserving the clean V9/V8/V7 curriculum.
- Model-visible training data contains no lower-mark prose, chat messages, candidate identifiers or private filenames.
- Held-out test: V11 loss 2.681 / perplexity 14.601; V10 loss 2.698 / perplexity 14.844. Lower is better, so V11 is selected.
- A V12 consideration-only continuation was trialled after a live hallucination regression. Its validation loss worsened from 2.388 to 2.478, so it was rejected rather than deployed. The reviewed corrective target remains in the auditable gold set and its deterministic doctrine gate runs with V11.

## Evaluation coverage

- 100/100 supplied questions parsed, with 50 essays and 50 problem questions.
- 100/100 have a subject route, subject guide, at least three legal RAG hits and marked-work assessment guidance.
- 100/100 preserve the individual requested total, including the corrected Q30 1,500, Q60 3,500 and Q85 4,500 values.
- 100/100 pass the deterministic release matrix for routing, RAG, exact part totals, 800-word caps, structure, citation policy and privacy presentation.
- Their combined requested body length is 452,100 words. This is all-question deterministic supervision, not a representation that 452,100 generated words received human legal marking.

## Visible browser regressions

- A private-mode live suite now covers 21 specialist practical enquiries and 14 SQE single-best-answer questions. All 35 returned their reviewed answer, passed subject accuracy/citation/privacy checks, applied the correct reference-list mode, and were permanently deleted after the test.
- The consideration essay regression initially returned 990 words but was rejected for repeated padding and for reversing *Foakes v Beer* and falsely linking promissory estoppel to the Misrepresentation Act 1967. The corrected live run returned a reviewed 1,010-word body in 6.16 seconds with no gate failures. Short-form case-name OSCOLA repair, consideration doctrine locks and substantive count repair were strengthened globally.
- The eight-theory jurisprudence essay returned a complete 2,012-word body in one pass. A GOV.UK Cambodia insolvency news result was detected as official but irrelevant; subject-semantic filtering now suppresses it and the rerun passed with no misleading source chip.
- A fresh unreviewed 2,000-word medical-negligence probe was deliberately stopped after its internal draft invented a named hospital/scan modality and blurred *Montgomery* with diagnostic breach. Nothing was published and the Private test chat was permanently deleted. The tort lock now separates *Bolam/Bolitho* diagnosis from *Montgomery* disclosure, requires *Paul v Royal Wolverhampton NHS Trust* for secondary-victim analysis, and rejects recovery of the employer's relational economic loss absent a recognised duty route. This remains a regression to rerun, not a claimed live pass.

- The exact fiduciary divided-loyalty essay and unsafe-workplace dismissal problem were rerun end-to-end through the live HTTP/SSE site after the July 2026 accuracy fixes. They returned one complete replacement event in 5.17 and 3.61 seconds respectively: 2,017/2,000 and 1,515/1,500 body words, with required headings, full adjacent OSCOLA, no runtime error and no private identifier. Both isolated test chats were then permanently deleted.
- The fiduciary regression now rejects the false settlor/beneficiary dual-duty framing, wrong dates/courts, misplaced trust-constitution or variation doctrine, the invented Denning attribution, and treatment of *Keech* as an imperfect-gift case. The employment regression requires ERA 1996 s 100, its reasonable-belief/serious-and-imminent test, the correct HSWA 1974 s 2 duty, correct dismissal route and non-guaranteed remedies.
- Legal Ethics Q30 was submitted through the visible site in Memory mode with England and Wales and `Always check latest`. The completed answer contains 1,502 body words (1,567 including References), explicit Introduction and Conclusion, current SRA Code/Principles, CPR r 31.20, a verified case, full OSCOLA and only two official SRA source chips. It contains no private source or internal label.
- The first generic attempt revealed a continuation-repetition defect and failed closed without saving partial prose. Continuations now receive headings plus the recent ending rather than the whole draft, and token-time repetition stopping terminates loops early. The exact high-risk regression also has a reviewed auditable fallback.
- Existing visible regressions cover the 2,000-word MedData/SecureCloud contract problem, the 2,000-word eight-theory jurisprudence essay, a 1,500-word road-negligence problem, a cross-subject critical essay and a practical proprietary-estoppel enquiry.

## Verification artefacts

- `training/LEGAL_APP_VERIFICATION.json`
- `training/LEGAL_EVAL_100_RAG_AUDIT.json`
- `training/LEGAL_EVAL_100_RELEASE_SUPERVISION.json`
- `training/LEGAL_EVAL_100_ROUTING_REPORT.md`
- `training/LEGAL_ETHICS_SITE_FULL_ANSWER_GATE.json`
- `training/MARKED_WORK_QUALITY_AUDIT.md`
- `training/legal_answer_flow_v8_complete_answer/RESULTS.md`
- `training/legal_answer_flow_v11_specialist/RESULTS.md`
- `training/live_private_release_sweep/report.jsonl`
- `data/legal_answer_flow_v8_complete_answer/AUDIT.json`
- `data/legal_answer_flow_v11_specialist/AUDIT.json`
- `data/legal_answer_flow_v12_consideration/AUDIT.json` (trial dataset; adapter not deployed)

## Reliability boundary

The site is ready for drafting, study, issue spotting and model-answer work, but no AI legal model can guarantee 100% legal accuracy or a 70+ university mark for every novel question. Novel answers remain subject to the same official-source checks and fail-closed gates and should be checked against the displayed current authorities before professional, assessment or high-stakes use.
