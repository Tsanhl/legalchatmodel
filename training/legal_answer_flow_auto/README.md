# Legal Answer Flow Auto LoRA

This config trains the local MLX model on generated source-ledger examples from:

- `model_database/snapshot/chroma_db`
- `model_database/snapshot/law_guides`
- `model_database/snapshot/gold_standard_shapes`
- `model_database/snapshot/project_guides`

Build data first:

```bash
./scripts/build_legal_dataset_from_snapshot.sh
```

Run a short smoke training:

```bash
./scripts/train_legal_answer_auto_lora.sh --iters 2 --steps-per-report 1 --steps-per-eval 1 --save-every 2
```

Run a longer first pass:

```bash
./scripts/train_legal_answer_auto_lora.sh
```

The model is not the source of truth. At runtime the app should still do:

```text
user question
-> selected uploads
-> indexed RAG/BM25/vector
-> official online fallback if thin/current
-> source ledger
-> local model generation
-> citation guard/supervisor
-> final answer
```
