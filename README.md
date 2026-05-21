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
heimdal doctor [--json]                       # profile hardware + model backend
heimdal run demo                              # run the built-in demo task
heimdal run --input examples/tasks/simple_task.json
heimdal run --instruction "Explain what a queue is."
heimdal eval run                              # run the eval suite + write a summary
heimdal patch validate examples/patches/good.json
heimdal logs latest                           # inspect the most recent run
```

Without an install, use `python -m heimdal <command>`.

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
