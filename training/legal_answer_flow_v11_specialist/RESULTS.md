# V11 specialist adapter results

V11 continues from the deployed V10 adapter for 30 conservative iterations at
a 5e-7 learning rate. It adds 35 reviewed, release-gated prompts: 21 practical
general enquiries and 14 SQE single-best-answer questions. The dataset contains
no private filenames, paths, candidate identifiers or lower-mark prose targets.

- Initial validation loss: 2.625
- Final validation loss: 2.318
- Untouched 12-example test loss: 2.681
- Untouched test perplexity: 14.601
- V10 comparison: loss 2.698, perplexity 14.844

Lower is better, so V11 is deployed. Runtime RAG, official-source checking,
subject gates, word-count supervision and privacy controls remain active.
