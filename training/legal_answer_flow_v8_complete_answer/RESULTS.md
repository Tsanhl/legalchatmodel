# V8 complete-answer corrective adapter results

## Selection

V8 is selected for deployment. It continues from the clean V7 70+ adapter and
adds reviewed complete-answer targets for essay, problem-question and general-
enquiry modes. The correction focuses on explicit Introduction/Conclusion
structure, proposition-level full parenthetical OSCOLA, one used-authority-only
References section, and suppression of plans, internal labels and private names.

## Measured results

- Initial validation loss at iteration 1: 2.447
- Iteration-10 validation loss: 2.449
- Final iteration-20 validation loss: 2.377
- Held-out test loss: 2.711; perplexity 15.050
- Same held-out test with V7: loss 2.716; perplexity 15.126

Lower is better. V8 improves both final validation loss and the untouched
held-out test result, while application-level release gates remain mandatory.

## Dataset and privacy boundaries

- Train: 52 examples (24 clean 70+ V7 examples plus 28 reviewed corrective examples)
- Validation: 18 held-out 75-mark examples
- Test: 12 held-out 72-mark examples
- No lower-mark submitted prose, chat messages, candidate identifiers, private
  filenames, `indexed`/`writing guidance` labels, or raw marker comments appear
  in model-visible targets.
- Private source metadata is neutralised even though metadata is not passed to
  the model.
