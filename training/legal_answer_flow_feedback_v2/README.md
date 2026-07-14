# Legal Answer Flow Feedback v2

This dataset is generated from captured app feedback.

Default feedback source (set `LEGAL_FEEDBACK_SOURCE` to your app's records folder):

`$LEGAL_FEEDBACK_SOURCE/user's request record for improvements/<dd-mm-yyyy>/corrections/*.json`

Promotion command:

```bash
python3 scripts/promote_feedback_to_lora_data.py
```

The promoter is idempotent. It tracks promoted feedback IDs in
`training/legal_answer_flow_feedback_v2/promotion_state.json` and only adds new
records once. Short comments are written to `data/legal_answer_flow_feedback_v2/review_needed.jsonl`
unless `--include-comments` is used.

Auto-promotion watcher:

```bash
./scripts/watch_feedback_v2_dataset.sh
```

The watcher polls the dated feedback folders and runs the promoter when new or
changed correction JSON files appear. It updates the v2 JSONL dataset only; it
does not start GPU training by default.

Optional batch auto-training after enough new corrections:

```bash
./scripts/watch_feedback_v2_dataset.sh \
  --train-command './scripts/train_feedback_v2_lora.sh' \
  --train-threshold 10
```

Training command:

```bash
./scripts/train_feedback_v2_lora.sh
```

The training script refuses to start if there are zero promoted feedback
examples. Do not run GPU training after every tiny comment. Promote feedback
often, then train v2 when there are enough real corrections to justify another
adapter.
