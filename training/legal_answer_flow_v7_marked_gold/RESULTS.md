# V7 marked-gold adapter results

## Selection

V7 is the deployed adapter. It was trained from the fresh Qwen 2.5 7B
4-bit base using only answer prose from work marked 70 or above. Lower-mark
scripts remain available to the runtime only as diagnostic marker feedback and
are not training targets or answer exemplars.

Training was stopped after iteration 60 because validation loss had flattened
and the following checkpoint took substantially longer than the prior cadence.
The iteration-60 checkpoint was saved normally and is the final
`adapters.safetensors` file.

## Observed training metrics

- Initial validation loss: 2.466
- Iteration 20 validation loss: 2.507
- Iteration 40 validation loss: 2.480
- Held-out test loss (V7 iteration 60): 2.716; perplexity 15.126
- Same held-out test set with the former V6 adapter: 2.835; perplexity 17.033

The held-out set contains separate 72-mark prose that was not used for V7
training. Lower loss is better. V7 is therefore selected over V6, subject to
the application-level contract-law full-answer gate.

## Dataset boundaries

- Train: 24 examples derived from 71-75-mark submitted answer prose
- Validation: 18 examples derived from held-out 75-mark prose
- Test: 12 examples derived from held-out 72-mark prose
- No private chats, student identifiers, marker comments, prompt drills, or
  synthetic meta-instructions appear in assistant targets
