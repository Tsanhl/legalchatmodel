# V10 full-length stability adapter results

V10 continues from the deployed V9 full-length adapter for ten conservative
iterations at a 5e-7 learning rate. The training data remains the privacy-clean
V9 corpus: reviewed complete 1,200-2,000-word answers, 600-800-word long-form
units, separate essay/problem/general/SQE citation modes, and no lower-mark
submitted prose or private identifiers.

On the same untouched 12-example held-out test used for V9, V10 achieved loss
2.698 and perplexity 14.844. V9 achieved loss 2.704 and perplexity 14.934.
Because lower is better, V10 is deployed. Runtime supervision still enforces
the requested +/-1% body count, structure, privacy, source integrity, current
official-source checking, and answer-mode-specific OSCOLA handling.
