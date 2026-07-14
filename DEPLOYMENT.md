# Deployment and private-data architecture

The checked release is a local Apple-silicon application. Its MLX inference
backend cannot be moved unchanged to a normal Linux CPU web host. Publishing the
source repository and publishing a working multi-user inference service are two
different tasks.

## Recommended no-GPU-cost pilot

Run the approved app on the Mac and place a production
[Cloudflare Tunnel](https://developers.cloudflare.com/tunnel/setup/) in
front of `http://127.0.0.1:8765`. This keeps the MLX model, private RAG database
and feedback records on the Mac while providing an HTTPS hostname. A Quick
Tunnel is suitable only for a short demonstration: Cloudflare documents that it
has no uptime guarantee and does not support Server-Sent Events, which this app
uses for progress and final-answer events.

Do not expose the current development server to unknown public users until it
has authentication, per-user authorization, request/upload limits, rate
limiting, abuse controls, a privacy notice and backups. The tunnel supplies a
network route; it does not add application-level access control.

## Configurable private storage

The app keeps safe local defaults, but a deployment may point each private store
at an encrypted local disk or mounted private volume:

```bash
export LEGAL_RAG_DB=/secure/legal/chroma.sqlite3
export LEGAL_GUIDANCE_DB=/secure/legal/feedback_index.sqlite3
export LEGAL_CHAT_DB=/secure/legal/chat.sqlite3
export LEGAL_FEEDBACK_ROOT=/secure/legal/training_feedback
export LEGAL_PRIVATE_UPLOAD_ROOT=/secure/legal/private_uploads

./scripts/chat_ui.sh
```

- `LEGAL_RAG_DB` is the read-only legal retrieval database.
- `LEGAL_GUIDANCE_DB` contains anonymized writing-guidance chunks.
- `LEGAL_CHAT_DB` stores conversations and their Memory/Private mode.
- `LEGAL_FEEDBACK_ROOT` receives Memory-mode exchanges and corrections for
  human review and later training promotion.
- `LEGAL_PRIVATE_UPLOAD_ROOT` stores Private uploads until their conversation is
  permanently deleted.

Private chats are rejected by the feedback endpoint and never enter the
training-feedback folder. A cloud host saves feedback on the cloud host, not on
the owner's Mac, unless that folder is a mounted/synchronised private volume.

## Where the 5.8 GB database may live

The SQLite/Chroma file can be held in private object storage as an encrypted
backup, but the running retrieval process needs a local or mounted copy; it
cannot execute SQLite queries directly against an HTTP object. Cloudflare R2's
documented [Standard-storage free tier](https://developers.cloudflare.com/r2/pricing/)
currently includes 10 GB-month, which is
large enough for one 5.8 GB object, subject to account and usage limits. Keep the
bucket private and verify that every source may lawfully be stored there.

Do not put the database in GitHub: [GitHub Free limits an individual Git LFS
file to 2 GB](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage),
and the database may preserve private or copyrighted source text.

## Fully cloud-hosted alternative

A genuine always-on public service requires porting inference from MLX to a
Linux-compatible engine and using GPU compute. [Hugging Face Spaces offers free
CPU hardware](https://huggingface.co/docs/hub/main/spaces-overview), but GPU
hardware is a paid upgrade unless a community grant is approved; free Spaces
also sleep when unused. The base model, adapter conversion, licence, private
storage and inference quality must all be revalidated after a port. This is not
equivalent to uploading the current repository.

## Feedback-to-training lifecycle

1. A user chooses Memory mode and submits an answer correction through the
   correction button.
2. The server writes a Markdown review record and JSON sidecar under
   `LEGAL_FEEDBACK_ROOT`.
3. Short comments remain human-review items. A substantial replacement answer
   may be considered as a candidate, never trusted automatically.
4. The owner verifies privacy, legal accuracy, citations, provenance and consent.
5. Approved examples are promoted with the existing feedback-promotion script,
   evaluated on an untouched test set, and deployed only if they beat the
   approved adapter and pass every release gate.

Never train automatically on unreviewed public feedback. That would allow prompt
injection, fabricated authorities, personal data and deliberately poisoned legal
answers to contaminate the model.
