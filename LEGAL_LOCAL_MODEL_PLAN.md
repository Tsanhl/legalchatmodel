# Legal Local Model Plan

Goal: make the local MLX model behave like an API provider while still using the legal app database/RAG and supervisor flow.

## Concept

The local model is the generation engine. The legal database remains outside the model as indexed RAG plus selected uploads. The model is trained to obey source-ledger discipline and answer-shape rules.

## Flow

```text
User question
-> selected chat uploads
-> indexed model database / RAG
-> legal answering guides
-> if thin/current: official online source fallback
-> source ledger
-> local fine-tuned model
-> citation guard / supervisor
-> final answer
```

## What Training Teaches

- legal answer structure
- essay/problem-question routing
- OSCOLA by default
- explicit citation-style override
- no fake sources, quotes or pinpoints
- use exact page/paragraph only when present in the source ledger
- insufficient-source handling
- bilingual Chinese/English legal explanation style
- supervised rewrite from weak draft to final answer

## What Runtime RAG Provides

- indexed documents
- uploaded chat documents selected by tick box
- current official online sources when needed
- source ledger entries

## First Commands

```bash
cd /path/to/mlx-lm-main
./scripts/train_legal_answer_lora.sh
./scripts/chat_legal_answer_lora.sh
./scripts/server_legal_answer_lora.sh
```

For a very small smoke run before real training:

```bash
./scripts/train_legal_answer_lora.sh --iters 2
```
