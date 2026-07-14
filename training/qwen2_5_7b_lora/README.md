# Qwen2.5 7B QLoRA Training

This workspace is set up to fine-tune `mlx-community/Qwen2.5-7B-Instruct-Uncensored-4bit` with MLX LoRA.

## Files

- `training/qwen2_5_7b_lora/config.yaml`: training config
- `data/qwen2_5_7b_chat_lora/train.jsonl`: training examples
- `data/qwen2_5_7b_chat_lora/valid.jsonl`: validation examples
- `data/qwen2_5_7b_chat_lora/test.jsonl`: optional test examples
- `adapters/qwen2_5_7b_lora/`: created after training

## Data Format

Use one JSON object per line:

```jsonl
{"messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"Your prompt"},{"role":"assistant","content":"Your desired answer"}]}
```

For a real run, replace the sample records with your own examples. Keep `valid.jsonl` separate so you can see whether validation loss improves instead of only memorizing the training examples.

## Commands

From this repo:

```bash
./scripts/train_qwen_lora.sh
```

Chat with the base model:

```bash
./scripts/chat_qwen_base.sh
```

Chat with the trained adapter after training:

```bash
./scripts/chat_qwen_lora.sh
```

Run an OpenAI-compatible server with the trained adapter:

```bash
./scripts/server_qwen_lora.sh
```

Then test it:

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data '{
    "model": "mlx-community/Qwen2.5-7B-Instruct-Uncensored-4bit",
    "messages": [
      {"role": "user", "content": "Hello"}
    ]
  }'
```

Fuse the adapter into a standalone model:

```bash
./scripts/fuse_qwen_lora.sh
```

