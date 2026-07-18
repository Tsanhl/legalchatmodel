# AGENTS.md

## Cursor Cloud specific instructions

This repo is Apple's `mlx-lm` (Python package under `mlx_lm/`, upstream inference/
fine-tuning library) forked to host a custom primary product: the **Legal Chat**
web app in `legal_chat_ui/`. The Legal Chat server is a Python-stdlib
`ThreadingHTTPServer` (no Flask/FastAPI/Node build); the browser UI is static
`legal_chat_ui/static/{index.html,app.js,styles.css}`.

### Setup / dependencies
- Everything installs into `.venv` via the update script (`pip install -e .`).
  Activate with `source .venv/bin/activate`. `mlx` is intentionally **not**
  installed on Linux (it is `platform_system == 'Darwin'` only in `setup.py`), so
  the in-process MLX backend is unavailable here.
- The V11 LoRA adapter (`adapters/legal_answer_flow_v11_specialist_lora/adapters.safetensors`)
  is a Git LFS file. Run `git lfs pull` to materialize it. It is only needed by the
  Apple-silicon MLX backend; on Linux the app runs fine with the LFS pointer left
  in place (`verify_legal_app.py` accepts the pointer).

### Running the Legal Chat app on Linux (non-Apple)
- The app auto-selects the `llama-server` backend on non-Apple hosts and expects an
  OpenAI-compatible server at `http://127.0.0.1:8080/v1`. The `/api/chat` endpoint
  returns HTTP 503 ("Model is still loading") until `MODEL.ready` is true — i.e.
  until a backend answers `GET /v1/models`. So a model backend MUST be running to
  send any chat, even though many answers are served from deterministic reviewed
  fixtures without calling the model (`pipeline.curated_regression_answer`).
- Lightweight local backend for dev/testing (not part of repo deps):
  `pip install "llama-cpp-python[server]"`, download a small GGUF (e.g.
  `Qwen/Qwen2.5-0.5B-Instruct-GGUF`), then
  `python -m llama_cpp.server --model <file.gguf> --host 127.0.0.1 --port 8080 --n_ctx 4096`.
- Start the app: `python legal_chat_ui/server.py` (or `./scripts/chat_ui.sh`, which
  activates `.venv` first). Serves `http://127.0.0.1:8765`.
- Set `LEGAL_ONLINE_MODE=off` to skip live official-source lookups
  (legislation.gov.uk, bailii, etc.); they degrade gracefully but can be slow or
  hang in a sandbox. Good curated hello-world question: "Explain proprietary
  estoppel in practical terms: what must a claimant prove, what remedies can the
  court award, and what evidence should be preserved?"

### Lint / test / verify
- Lint: `pre-commit run --all-files` (black + isort). NOTE: this currently reports
  formatting drift on many committed files — this fork does not run pre-commit in
  CI (`pull_request.yml` is gated to the upstream repo), so that drift is
  pre-existing, not your regression.
- The `tests/` unittest suite requires `mlx` (Apple silicon) and cannot run on
  Linux. The Linux verification gate is `python scripts/verify_legal_app.py`
  (deterministic, no GPU/model; ~90 checks; also runs in `legal-public-checks.yml`).
