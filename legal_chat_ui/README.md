# Legal Chat UI (local model only)

A self-contained chat front-end for the fine-tuned legal model. It talks **only**
to the local model, uses `legal_answer_flow_v11_specialist_lora` by default, and
offers two explicit history modes.

- **Memory** — saved in SQLite, available as context to later Memory chats, and
  mirrored into the improvement/training-review records.
- **Private** — saved only for reopening that chat, isolated from cross-chat memory
  and training records, with permanent deletion for its messages and uploads.

## Run

```bash
cd /path/to/mlx-lm-main
./scripts/chat_ui.sh
```

Then open <http://127.0.0.1:8765> in your browser. The model loads in the background
(the status pill flips to **"Local model ready"** when it's loaded); chat once it's ready.

Options:

```bash
./scripts/chat_ui.sh --port 9000     # different port
./scripts/chat_ui.sh --no-adapter    # base model only (no fine-tune)
./scripts/chat_ui.sh --temp 0.2      # lower temperature
```

## Where chats are saved

- `legal_chat_ui/chat.sqlite3` — `conversations` + `messages` tables.
- Conversations appear in the left sidebar (newest first). Click one to reload it.
- Existing chats are migrated to Memory mode. Private chats have a trash action
  that securely removes their database rows and private uploads.

```bash
# inspect saved chats from the terminal
sqlite3 legal_chat_ui/chat.sqlite3 \
  "select c.title, m.role, substr(m.content,1,80) from messages m join conversations c on c.id=m.conversation_id order by m.id;"
```

## What this is / isn't

- ✅ Local fine-tuned `Qwen2.5-7B` + the deployed V11 specialist adapter.
- ✅ Cross-chat Memory mode plus isolated, fully deletable Private mode.
- ✅ Bundled anonymized subject guides plus optional local database RAG.
- ✅ Mandatory relevant official-source checking, adjacent OSCOLA and complete-answer supervision.
- ✅ Jurisdiction selector (passed into the system prompt).
- ✅ All chats persisted and re-openable.
- ⛔ No external provider keys. The local model is the only engine.

## Architecture

```
browser (static/index.html, app.js, styles.css)
  -> POST /api/chat (Server-Sent Events)
       -> upload context + local/bundled RAG + official-source check
       -> subject guide + V11 model + deterministic release supervision
       -> one complete replacement event
       -> chat.sqlite3 according to Memory or Private mode
```
