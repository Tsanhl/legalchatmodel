# Legal Answer Flow LoRA

Purpose: fine-tune the local model so users can get API-like legal-answer behaviour from the local model server.

This adapter should learn:
- essay/problem-question structure,
- OSCOLA default and explicit citation-style override,
- source-ledger discipline,
- no fake citations/pinpoints/quotes,
- selected upload + indexed database first,
- official online fallback when source support is thin/current,
- supervisor-style final answer correction,
- Chinese/English explanation style where requested.

It should not memorize confidential facts or replace RAG. Indexed legal documents remain the model database at runtime.

Run from the project root (`mlx-lm-main`):

```bash
./scripts/train_legal_answer_lora.sh
./scripts/chat_legal_answer_lora.sh
./scripts/server_legal_answer_lora.sh
```

Server endpoint for the app later:

```text
http://127.0.0.1:8000/v1/chat/completions
```
