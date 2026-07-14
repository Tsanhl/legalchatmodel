# V11 specialist LoRA adapter

This is the deployed LegalChatModel adapter for
`mlx-community/Qwen2.5-7B-Instruct-Uncensored-4bit`.

- Fine-tuning type: LoRA, rank 8, 16 layers
- Context used for training: 4,096 tokens
- Selected validation: V11 outperformed the preceding V10 adapter on the
  untouched test set (loss 2.681 vs 2.698; perplexity 14.601 vs 14.844)
- Privacy audit: the model-visible corrective datasets contain no local paths,
  candidate identifiers, private filenames, chat transcripts, or lower-mark
  submitted prose

Only the final `adapters.safetensors` and its configuration are published.
Intermediate checkpoints and the rejected V12 experiment are intentionally
excluded.

An adapter is not a legal authority or an accuracy guarantee. Runtime official
source checking, authority repair, privacy filtering, and release supervision
remain necessary.
