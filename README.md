# Heimdal Engine

Heimdal Engine is a **host-agnostic local agent engine**. Externally it appears
as one agent; internally it runs a quality-controlled orchestration runtime so
that smaller local models can produce frontier-like agent outcomes through
architecture — context discipline, retrieval, verification, evals, patching, and
hardware-aware scheduling — rather than raw model size.

The full specification lives in [`docs/builder_pack/`](docs/builder_pack/);
start with `docs/builder_pack/INDEX.md`.

## Install

Python 3.11+ is required. The only runtime dependency is `pyyaml`.

```bash
bash scripts/dev_setup.sh        # venv + install + doctor (Ubuntu native / WSL2)
# or, in an existing environment:
pip install -e .
```

Heimdal runs fully offline. If [Ollama](https://ollama.com) is reachable it is
used automatically; otherwise a deterministic offline backend keeps the whole
pipeline runnable.

## CLI

```bash
heimdal doctor [--json] [--model <name>]      # profile hardware + model backend
heimdal run demo                              # run the built-in demo task
heimdal run --input examples/tasks/simple_task.json
heimdal run --instruction "Explain what a queue is."
heimdal eval run                              # run the eval suite + write a summary
heimdal verify --task task.json --answer answer.json   # verify a host-supplied answer
heimdal openclaw run --input task.json        # run an OpenClaw payload through Heimdal
heimdal hermes run --input task.json          # run a Hermes payload through Heimdal
heimdal hermes capabilities --json            # report host-integration capabilities
heimdal bridge init                           # create the local file bridge dirs
heimdal bridge once --backend offline         # process inbox jobs and exit
heimdal patch validate examples/patches/good.json
heimdal truth list                            # list local Truth Vault sources
heimdal truth add notes.md                    # add a .md/.txt file to the vault
heimdal truth search "refund policy"          # BM25 search over the vault
heimdal logs latest                           # inspect the most recent run
```

Without an install, use `python -m heimdal <command>`.

`heimdal verify` runs only the verifier on an answer a host drafted itself —
useful when a host wants Heimdal to grade a candidate rather than produce one.

### Host integrations (OpenClaw, Hermes)

A host framework drives Heimdal as a single agent — via a CLI command or
in-process:

```python
from heimdal.adapters.openclaw_host import handle as openclaw_handle
from heimdal.adapters.hermes_host import handle as hermes_handle

result = openclaw_handle(openclaw_payload)   # -> OpenClaw-style result dict
result = hermes_handle(hermes_payload)       # -> Hermes-style result dict
```

`handle()` translates the host payload, runs the full Quality Factory, and
translates the result back. It accepts `backend` / `model` / `verifier`
overrides, and when the payload carries `callback.file` the result is also
written under `storage/workspace/`. The adapter classes (`OpenClawAdapter`,
`HermesAdapter`) only translate; orchestration lives in the `*_host` modules.
CLI equivalents: `heimdal openclaw run` and `heimdal hermes run`.

The Hermes result is schema-formalized (`schemas/hermes_result.schema.json`):
it carries a machine-readable `code` (e.g. `SOURCE_MISSING`,
`VERIFIER_SEMANTIC_FAIL`), structured `needed_inputs` on `need_input`, and only
host-safe relative refs — never absolute paths, raw prompts, the full Context
Packet, or the internal sub-agent graph. `heimdal hermes capabilities` reports
what the host integration supports.

### Backend and model selection

`run` and `eval` accept `--backend ollama|offline` and `--model <name>` to force
a backend or override the worker model. `doctor --model <name>` picks the model
for capability tests. When no override is given, Heimdal prefers an installed
manifest candidate and skips embedding models; if no candidate is installed it
falls back to an installed generative model or fails with an actionable
`ollama pull` hint.

### Verifier

Verification is **rule-based** by default. A **hybrid** mode adds an optional
model-based *semantic verifier* for B2-B4 tasks — set `verifier.mode: hybrid`
in `config/heimdal.manifest.yml`, or pass `--verifier hybrid` to `run` / `eval`.

The gate order is fixed: (1) deterministic rule-based verifier, (2) optional
model-based semantic verifier, (3) final deterministic check. A deterministic
hard fail short-circuits before the model verifier runs — the semantic verifier
is an extra quality layer and can never override a deterministic fail. The
offline backend mocks the semantic verifier deterministically.

## How a task flows

```
Host input -> Adapter -> Host Task Envelope -> Core Runtime
  Intake -> Role Binding -> Task Contract -> Scheduler (Work mode)
  Quality Factory: Context Packet -> Router -> Worker -> Verifier -> Repair loop
  -> Result Envelope + Repro Pack + Trace Pack
```

- **Task Contract** is built for every task before any model call.
- **Context Packet** is the exact, token-budgeted context for every model call.
- **Verifier** returns a structured PASS/FAIL result and never rewrites the answer.
- **No-Guess Gate** returns `need_input` instead of inventing unsourced facts.
- **Repro Pack** and **Trace Pack** are written for every run.
- **Patch / Eval gate**: no patch reaches the `stable` channel without a passing
  eval run.
- Heimdal core is host-agnostic; **OpenClaw is only an adapter**. Mirror Mode is
  disabled by default; Dream Mode is feature-flagged.

## Layout

```
heimdal/        engine package (core runtime, models, adapters, hardware, sandbox)
schemas/        JSON schemas for envelopes, contracts, packets, packs, patches
config/         heimdal.manifest.yml + sandbox_policy.yml
examples/       example tasks, truth sources, skills, patches
eval/           the eval suite (smoke / must-pass / schema / no-guess)
docs/builder_pack/  the Heimdal Builder Pack specification
tests/          stdlib unittest suite
```

## Tests

```bash
python -m unittest discover -s tests -t . -v
```

## License

MIT — see [LICENSE](LICENSE).
