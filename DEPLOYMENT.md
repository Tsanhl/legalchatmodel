# Deployment and private-data architecture

The checked release is a local Apple-silicon application. Its MLX inference
backend cannot be moved unchanged to a normal Linux CPU web host. Publishing the
source repository and publishing a working multi-user inference service are two
different tasks.

## Recommended no-GPU-cost public pilot

Run the approved app on the Mac and place a production
[Cloudflare Tunnel](https://developers.cloudflare.com/tunnel/setup/) in
front of `http://127.0.0.1:8765`. This keeps the MLX model, private RAG database
and feedback records on the Mac while providing an HTTPS hostname. A Quick
Tunnel is suitable only for a short demonstration: Cloudflare documents that it
has no uptime guarantee and does not support Server-Sent Events, which this app
uses for progress and final-answer events.

The application now has an opt-in public mode with Cloudflare Access JWT
validation, just-in-time user records, per-user authorization, persistent
generation limits, upload quotas, consent-aware feedback and account deletion.
The ordinary local launcher does not enable any of this public surface.

Cloudflare Access's current free plan is limited to 50 users. It is appropriate
for a public pilot, not an unlimited free GPT-scale service. The local 7B model
also serializes generation, so one complete answer runs at a time.

### What a website visitor downloads

Nothing beyond the normal HTML, CSS and JavaScript page. The Mac downloads and
stores the Qwen base model once, loads V11 and performs inference. A visitor does
not clone GitHub, install Python or download model weights.

### First public launch

1. Install the checked application and adapter on the host Mac:

   ```bash
   git pull
   git lfs pull
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

2. Put a domain on Cloudflare, create a **named** tunnel, and route its public
   hostname to `http://127.0.0.1:8765`. A sample ingress file is at
   `deploy/cloudflared-config.example.yml`.

3. Create a self-hosted Cloudflare Access application for the same hostname.
   Configure an identity provider and an Allow policy. For a deliberately open
   email pilot, Cloudflare documents that an Allow policy using the One-time PIN
   login method permits anyone with a valid email; use that only with the quotas
   and abuse controls enabled. Never use a Bypass policy.

4. Copy the Access **team domain** and **Application Audience (AUD) tag**, then
   start the origin. For an interactive foreground run:

   ```bash
   export CF_ACCESS_TEAM_DOMAIN="https://your-team.cloudflareaccess.com"
   export CF_ACCESS_AUD="your-application-audience-tag"
   export LEGAL_PUBLIC_DATA_DIR="$HOME/Library/Application Support/LegalAI-public"

   ./scripts/public_chat_ui.sh
   ```

   On macOS, the recommended persistent setup stores the tunnel token outside
   the repository with mode `0600` and installs separate origin/tunnel user
   launch agents. Download `cloudflared`, then run:

   ```bash
   python3 scripts/configure_public_macos.py install \
     --team-domain "https://your-team.cloudflareaccess.com" \
     --aud "your-application-audience-tag"
   ```

   Paste the tunnel token only at the hidden prompt. It is never printed or
   added to Git. Check the services with
   `python3 scripts/configure_public_macos.py status`. The Mac must remain
   powered on, connected to the internet and logged in for this user-level
   service to answer visitors.

5. If you used the foreground launch, start the named tunnel with
   `cloudflared tunnel run YOUR_TUNNEL_NAME`. The macOS installer in step 4
   starts it automatically. Visit the public hostname: Access performs the
   login, and the origin separately validates the JWT signature, issuer and
   audience.

The public launcher binds only to loopback and refuses to start without the
Access configuration and separate database/feedback/upload paths. This prevents
the existing personal chat database being selected accidentally. It also points
retrieval at separate public-index paths. Unless you deliberately build a
redistribution-safe public index there, the app uses its bundled anonymised
guides and cannot query the private `model_database` directory.

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
training-feedback folder. In public mode, Memory chats are not automatically
mirrored into training records. A correction is stored in the structured
`feedback` SQL table, and a training-candidate JSON/Markdown pair is created
only when that user explicitly opts in. A cloud host saves these records on the
cloud host, not on the owner's Mac, unless the folder is a mounted or
synchronised private volume.

## Public SQL schema and deletion

The server migrates its SQLite database automatically. It contains:

- `users`: opaque internal ID plus verified external identity subject;
- `conversations`: every row has a `user_id` owner;
- `messages` and `attachments`: accessible only through an owned conversation;
- `feedback`: correction, consent flag and human-review status;
- `usage_events`: persistent hourly/daily generation limits.

Every read, upload, correction, generation and deletion is scoped to the
authenticated user. Public conversation deletion is permanent. Account deletion
removes that user's identity, chats, messages, attachments, feedback, usage
events and stored upload files, then truncates the SQLite WAL.

Default public limits can be changed before launch:

```bash
export LEGAL_REQUESTS_PER_HOUR=20
export LEGAL_REQUESTS_PER_DAY=50
export LEGAL_MAX_USER_CONVERSATIONS=100
export LEGAL_MAX_UPLOAD_BYTES=8388608
export LEGAL_MAX_USER_STORAGE_BYTES=52428800
export LEGAL_MAX_QUESTION_CHARS=60000
export LEGAL_RETENTION_DAYS=90
```

Public mode applies the retention period at startup. Set it to the period stated
in the published privacy notice; `0` disables automatic expiry.

### Encrypted backups

Install `age`, create an age key pair, keep the private identity offline, and
back up to its public recipient:

```bash
export LEGAL_BACKUP_AGE_RECIPIENT="age1..."
export LEGAL_BACKUP_DIR="/path/to/encrypted/offsite/backups"
./scripts/backup_public_data.sh
```

The script uses SQLite's online backup API, packages the SQL snapshot plus
feedback and uploads, and writes only an encrypted `.tar.gz.age` file. Schedule
it with `launchd` after testing a restore. Backups must follow the same retention
and deletion policy as the live service.

Before giving the hostname to unknown users, publish a privacy notice and terms
covering the operator's identity, legal-information disclaimer, purposes of
processing, Memory versus Private behaviour, feedback consent, retention,
account deletion, international transfers and a contact route. Enable FileVault
or equivalent disk encryption, encrypted off-device backups, monitoring and an
incident-response process. These operational/legal documents are specific to
the operator and jurisdiction and cannot safely be inferred by the codebase.

## Where the private database may live

The SQLite/Chroma file can be held in private object storage as an encrypted
backup, but the running retrieval process needs a local or mounted copy; it
cannot execute SQLite queries directly against an HTTP object. Cloudflare R2's
documented [Standard-storage free tier](https://developers.cloudflare.com/r2/pricing/)
currently includes 10 GB-month. The current primary Chroma SQLite file is about
5.5 GB, while the complete local `model_database` directory is about 8.8 GB.
That leaves little free-tier backup headroom. Keep the bucket private and verify
that every source may lawfully be stored there.

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
2. The server always records structured feedback in SQLite. It writes a
   Markdown review record and JSON sidecar under `LEGAL_FEEDBACK_ROOT` only when
   the user selects the training-consent checkbox.
3. Short comments remain human-review items. A substantial replacement answer
   may be considered as a candidate, never trusted automatically.
4. The owner verifies privacy, legal accuracy, citations, provenance and consent.
5. Approved examples are promoted with the existing feedback-promotion script,
   evaluated on an untouched test set, and deployed only if they beat the
   approved adapter and pass every release gate.

For the public folder layout, pass the configured root explicitly. The scanner
recurses through opaque user/date folders and sees only consented JSON files:

```bash
python scripts/promote_feedback_to_lora_data.py \
  --feedback-dir "$LEGAL_FEEDBACK_ROOT"
```

Never train automatically on unreviewed public feedback. That would allow prompt
injection, fabricated authorities, personal data and deliberately poisoned legal
answers to contaminate the model.
