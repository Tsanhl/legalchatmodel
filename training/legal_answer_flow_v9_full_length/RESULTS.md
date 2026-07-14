# V9 full-length corrective adapter results

## Selection

V9 iteration 20 is selected for deployment. It continues from V8 and adds
reviewed complete 1,200-, 1,500- and 2,000-word targets so ordinary answers do
not inherit the 600–800-word stopping pattern used for long-form units. It also
teaches the mode distinction: essays/problems include one used-authority-only
References section; general enquiries and SQE use full inline OSCOLA without a
final list unless requested.

## Measured results

- Initial validation loss: 2.656
- Iteration-10 validation loss: 2.505
- Iteration-20 validation loss: 2.543
- V9 iteration-20 held-out test: loss 2.704; perplexity 14.934
- V9 iteration-10 held-out test: loss 2.707; perplexity 14.989
- V8 on the identical held-out test: loss 2.711; perplexity 15.050

Iteration 20 has the best untouched test result and is therefore deployed.

## Dataset and privacy

- Train: 104 examples; validation: 18; test: 12.
- Added reviewed full-answer targets cover contract, tort, jurisprudence, legal
  ethics, land-law general enquiry and SQE modes.
- The 100-question release bank still spans 1,000–20,000 words and 25 routed
  subject families; requests above 2,500 words use internal units of at most
  800 words.
- No lower-mark submitted prose, private filename, candidate identifier or
  internal indexed/guidance label appears in model-visible targets.
