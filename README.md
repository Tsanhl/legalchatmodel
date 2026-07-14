# LegalChatModel

LegalChatModel is a local, ChatGPT-style legal drafting and study application
tailored to the law of England and Wales. It combines a fine-tuned MLX model,
an anonymized legal knowledge base, official current-law checking, structured
answer modes, OSCOLA output controls, cross-chat Memory, and isolated Private
conversations.

The target is first-class/70+ law-school technique: precise issue selection,
authority-led analysis, counterargument, fact-specific application, calibrated
conclusions, and practical remedies. That is an engineering and writing
standard—not a promise of a particular mark or error-free legal advice.

## Core behaviour

- **Essay:** explicit Introduction with a qualified thesis; issue-led critical
  analysis; competing views; reasoned Conclusion; used-authority References.
- **Problem question:** chronology and issues; exact rule; immediate application;
  alternative factual branches; causation, defences, remedies and final advice.
- **General enquiry:** direct plain-language answer, governing law, limitations,
  evidence and next steps; no separate reference list unless requested.
- **SQE:** single best answer first, concise IRAC explanation, and rejection of
  the material distractors; no separate reference list unless requested.
- **OSCOLA:** full verified citations appear in parentheses immediately after
  the supported proposition. Essays and problems also receive a deterministic
  used-authority-only References section.
- **Word count:** requested body length is accepted only within −1%/+1%.
  Requests up to 2,500 words use one supervised answer; larger answers are
  assembled internally from focused units of no more than 800 words. A
  20,000-word request therefore plans 25 units but publishes one continuous
  final answer.

## Answer flow

```text
user question
  -> jurisdiction + answer-mode + requested-word-count detection
  -> uploaded documents (current chat only)
  -> lexical RAG from the local database or bundled anonymized guides
  -> mandatory relevant official-source current-law check
  -> subject structure and authority bank
  -> V11 local model draft
  -> doctrine, citation, privacy, repetition, structure and word-count gates
  -> one atomic complete answer
```

An official domain is not sufficient by itself: online results are filtered for
subject relevance before appearing as source chips. If no sufficiently relevant
official result is found, the application says so instead of presenting an
unrelated page as support.

## Bundled anonymized legal knowledge

The public repository includes 51 subject/writing guides in
`legal_chat_ui/law_guides`. They contain the distilled method, issue maps,
common mark-loss warnings, accuracy checkpoints and verified OSCOLA authority
banks needed by the runtime. A clone does not need a private external folder.

Coverage includes contract, tort, criminal law and procedure, evidence, land,
equity and trusts, company, commercial, insolvency, employment, family, public
law, human rights, EU law, international law, jurisprudence, medical law,
privacy/media, AI/data protection, consumer, intellectual property, competition,
environmental, immigration/refugee, tax, pensions, housing, civil procedure,
sentencing, restitution, remedies, aviation, construction, cybercrime, election,
extradition, insurance, maritime, international trade, financial regulation,
public procurement, cultural heritage, mediation and succession.

The distilled quality rules are:

1. Answer the exact proposition or advisee, not the general topic.
2. Separate doctrinal gateways before remedies or policy evaluation.
3. State a verified rule and citation before applying it.
4. Apply every major conclusion to a stated fact; label missing facts and show
   how each alternative changes the outcome.
5. Give the strongest argument and strongest objection, then resolve them.
6. Distinguish causation, remoteness, scope of duty, proof and quantum.
7. Use calibrated outcomes such as “likely” or “strong argument” where facts or
   doctrine remain contestable.
8. Never invent a case, statute, court, year, quotation, pinpoint, party fact,
   clause number or source.
9. Use current official law for unstable statutes, procedure and regulation.
10. Conclude by answering the question, not by repeating the introduction.

## Privacy model

- **Memory:** completed user-authored Memory chats may supply relevant history
  to later Memory chats and may join local improvement records.
- **Private:** isolated from cross-chat memory and training/improvement records;
  the conversation and its uploads can be permanently deleted.
- Prior assistant prose, incomplete chats and Private chats are excluded from
  cross-chat retrieval.
- Local filenames, paths, candidate-number-shaped strings, internal source
  labels and marked-work provenance are removed or blocked from output.
- Chat SQLite databases, uploads, raw training datasets, local improvement logs,
  private source documents and the large vector database are ignored by Git.

The Git history contains a neutral noreply author. The public tracked tree is
audited for local usernames, personal candidate identifiers, private source
folder names, attachment paths and personal email addresses.

## Database and adapter policy

The deployed V11 adapter is included through Git LFS. Intermediate checkpoints,
failed experiments and all other local adapters are excluded. The production
weights are pinned by SHA-256 so a missing Git LFS download, corrupted file or
inferior experiment cannot silently replace V11.

The local 5.8 GB vector database is deliberately **not** published: normal
GitHub storage is unsuitable, and a source index can preserve copyrighted or
private text. When that database is absent, the application automatically uses
the bundled anonymized guides and reviewed public answer corpus as its lexical
RAG knowledge base. Users may build their own local index without committing it.

For the free-pilot, private-storage and feedback-to-training architecture, see
[DEPLOYMENT.md](DEPLOYMENT.md).

## Quick start on Apple silicon

Requirements: macOS, Python 3.10+ and enough memory for the 4-bit 7B base model.

```bash
git clone https://github.com/Tsanhl/legalchatmodel.git
cd legalchatmodel
git lfs pull

python3 -m venv .venv
source .venv/bin/activate
pip install -e .

./scripts/chat_ui.sh
```

Open <http://127.0.0.1:8765/> and wait for **Local model ready**. The base model
is downloaded from its model host on first use; its own model card and licence
apply separately.

### Public pilot

Website visitors do **not** download Qwen or this repository. The operator's
Apple-silicon Mac downloads the base model once and serves browser users through
an authenticated named tunnel. The public launcher adds verified Cloudflare
Access identities, per-user chat/Memory isolation, SQL feedback, explicit
training consent, persistent rate limits, upload quotas and complete account
deletion:

```bash
export CF_ACCESS_TEAM_DOMAIN="https://your-team.cloudflareaccess.com"
export CF_ACCESS_AUD="your-application-audience-tag"
./scripts/public_chat_ui.sh
```

This is not deployable through GitHub Pages: Pages cannot run the Python/MLX
model process. Complete tunnel, Access policy and private-storage instructions
are in [DEPLOYMENT.md](DEPLOYMENT.md).

Useful options:

```bash
./scripts/chat_ui.sh --port 9000
./scripts/chat_ui.sh --no-adapter
./scripts/chat_ui.sh --temp 0.1
```

## Verification

```bash
python3 scripts/verify_legal_app.py
python3 scripts/live_private_release_sweep.py --general --fresh
python3 scripts/live_private_release_sweep.py --sqe --fresh
```

The deterministic suite covers routing, bundled/public RAG fallback, current-law
search, exact 1,000–20,000-word planning, structure, adjacent OSCOLA, privacy,
Memory/Private separation and hard deletion. Live test conversations use Private
mode and are permanently deleted after inspection.

## Important limitations

No language model can guarantee 100% legal accuracy, a 70+ mark, or suitability
for professional advice. Current law can change, official results can be
incomplete, and novel fact patterns can expose model errors. Check cited primary
authorities and obtain qualified advice for professional, assessment or
high-stakes use.

---

## Upstream MLX LM

MLX LM is a Python package for generating text and fine-tuning large language
models on Apple silicon with MLX.

Some key features include:

* Integration with the Hugging Face Hub to easily use thousands of LLMs with a
  single command. 
* Support for quantizing and uploading models to the Hugging Face Hub.
* [Low-rank and full model
  fine-tuning](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
  with support for quantized models.
* Distributed inference and fine-tuning with `mx.distributed`

The easiest way to get started is to install the `mlx-lm` package:

**With `pip`**:

```sh
pip install mlx-lm
```

**With `conda`**:

```sh
conda install -c conda-forge mlx-lm
```

### Quick Start

To generate text with an LLM use:

```bash
mlx_lm.generate --prompt "How tall is Mt Everest?"
```

To chat with an LLM use:

```bash
mlx_lm.chat
```

This will give you a chat REPL that you can use to interact with the LLM. The
chat context is preserved during the lifetime of the REPL.

Commands in `mlx-lm` typically take command line options which let you specify
the model, sampling parameters, and more. Use `-h` to see a list of available
options for a command, e.g.:

```bash
mlx_lm.generate -h
```

The default model for generation and chat is
`mlx-community/Llama-3.2-3B-Instruct-4bit`.  You can specify any MLX-compatible
model with the `--model` flag. Thousands are available in the
[MLX Community](https://huggingface.co/mlx-community) Hugging Face
organization.

### Python API

You can use `mlx-lm` as a module:

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")

prompt = "Write a story about Einstein"

messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True,
)

text = generate(model, tokenizer, prompt=prompt, verbose=True)
```

To see a description of all the arguments you can do:

```
>>> help(generate)
```

Check out the [generation
example](https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/examples/generate_response.py)
to see how to use the API in more detail. Check out the [batch generation
example](https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/examples/batch_generate_response.py)
to see how to efficiently generate continuations for a batch of prompts.

The `mlx-lm` package also comes with functionality to quantize and optionally
upload models to the Hugging Face Hub.

You can convert models using the Python API:

```python
from mlx_lm import convert

repo = "mistralai/Mistral-7B-Instruct-v0.3"
upload_repo = "mlx-community/My-Mistral-7B-Instruct-v0.3-4bit"

convert(repo, quantize=True, upload_repo=upload_repo)
```

This will generate a 4-bit quantized Mistral 7B and upload it to the repo
`mlx-community/My-Mistral-7B-Instruct-v0.3-4bit`. It will also save the
converted model in the path `mlx_model` by default.

To see a description of all the arguments you can do:

```
>>> help(convert)
```

#### Streaming

For streaming generation, use the `stream_generate` function. This yields
a generation response object.

For example,

```python
from mlx_lm import load, stream_generate

repo = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
model, tokenizer = load(repo)

prompt = "Write a story about Einstein"

messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True,
)

for response in stream_generate(model, tokenizer, prompt, max_tokens=512):
    print(response.text, end="", flush=True)
print()
```

#### Sampling

The `generate` and `stream_generate` functions accept `sampler` and
`logits_processors` keyword arguments. A sampler is any callable which accepts
a possibly batched logits array and returns an array of sampled tokens.  The
`logits_processors` must be a list of callables which take the token history
and current logits as input and return the processed logits. The logits
processors are applied in order.

Some standard sampling functions and logits processors are provided in
`mlx_lm.sample_utils`.

### Command Line

You can also use `mlx-lm` from the command line with:

```
mlx_lm.generate --model mistralai/Mistral-7B-Instruct-v0.3 --prompt "hello"
```

This will download a Mistral 7B model from the Hugging Face Hub and generate
text using the given prompt.

For a full list of options run:

```
mlx_lm.generate --help
```

To quantize a model from the command line run:

```
mlx_lm.convert --model mistralai/Mistral-7B-Instruct-v0.3 -q
```

For more options run:

```
mlx_lm.convert --help
```

You can upload new models to Hugging Face by specifying `--upload-repo` to
`convert`. For example, to upload a quantized Mistral-7B model to the
[MLX Hugging Face community](https://huggingface.co/mlx-community) you can do:

```
mlx_lm.convert \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    -q \
    --upload-repo mlx-community/my-4bit-mistral
```

Models can also be converted and quantized directly in the
[mlx-my-repo](https://huggingface.co/spaces/mlx-community/mlx-my-repo) Hugging
Face Space.

### Long Prompts and Generations 

`mlx-lm` has some tools to scale efficiently to long prompts and generations:

- A rotating fixed-size key-value cache.
- Prompt caching

To use the rotating key-value cache pass the argument `--max-kv-size n` where
`n` can be any integer. Smaller values like `512` will use very little RAM but
result in worse quality. Larger values like `4096` or higher will use more RAM
but have better quality.

Caching prompts can substantially speedup reusing the same long context with
different queries. To cache a prompt use `mlx_lm.cache_prompt`. For example:

```bash
cat prompt.txt | mlx_lm.cache_prompt \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --prompt - \
  --prompt-cache-file mistral_prompt.safetensors
``` 

Then use the cached prompt with `mlx_lm.generate`:

```
mlx_lm.generate \
    --prompt-cache-file mistral_prompt.safetensors \
    --prompt "\nSummarize the above text."
```

The cached prompt is treated as a prefix to the supplied prompt. Also notice
when using a cached prompt, the model to use is read from the cache and need
not be supplied explicitly.

Prompt caching can also be used in the Python API in order to avoid
recomputing the prompt. This is useful in multi-turn dialogues or across
requests that use the same context. See the
[example](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/examples/chat.py)
for more usage details.

### Supported Models

`mlx-lm` supports thousands of LLMs available on the Hugging Face Hub. If the
model you want to run is not supported, file an
[issue](https://github.com/ml-explore/mlx-lm/issues/new) or better yet, submit
a pull request. Many supported models are available in various quantization
formats in the [MLX Community](https://huggingface.co/mlx-community) Hugging
Face organization.

For some models the tokenizer may require you to enable the `trust_remote_code`
option. You can do this by passing `--trust-remote-code` in the command line.
If you don't specify the flag explicitly, you will be prompted to trust remote
code in the terminal when running the model. 

Tokenizer options can also be set in the Python API. For example:

```python
model, tokenizer = load(
    "qwen/Qwen-7B",
    tokenizer_config={"eos_token": "<|endoftext|>", "trust_remote_code": True},
)
```

### Large Models

> [!NOTE]
    This requires macOS 15.0 or higher to work.

Models which are large relative to the total RAM available on the machine can
be slow. `mlx-lm` will attempt to make them faster by wiring the memory
occupied by the model and cache. This requires macOS 15 or higher to
work.

If you see the following warning message:

> [WARNING] Generating with a model that requires ...

then the model will likely be slow on the given machine. If the model fits in
RAM then it can often be sped up by increasing the system wired memory limit.
To increase the limit, set the following `sysctl`:

```bash
sudo sysctl iogpu.wired_limit_mb=N
```

The value `N` should be larger than the size of the model in megabytes but
smaller than the memory size of the machine.
