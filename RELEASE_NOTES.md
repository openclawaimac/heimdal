# Heimdal v0.2.1 — Ollama Runtime Hardened Beta

First hardened beta of Heimdal Engine — a host-agnostic local agent engine that
makes smaller local models behave like frontier-class agents through
architecture: context discipline, retrieval, verification, and evals.

## Validated

Real-Ollama Runtime Validation passed on:

- Ubuntu 24.04 (WSL2)
- Ollama 0.15.6
- Model `qwen2.5:7b`
- `heimdal doctor`, `run demo`, simple task, no-guess task, the eval suite, and
  the unit test suite all pass

## Key features

- Host-agnostic engine core
- CLI adapter + OpenClaw adapter stub
- Task Contract + Context Packet built for every task
- Quality Factory (router → worker → verifier → repair loop)
- Repro Packs and Trace Packs written for every run
- No-Guess Gate (returns `need_input` instead of inventing facts)
- Deterministic rule-based verifier
- Ollama runtime hardening: typed errors, retries/backoff, request tracing
- Doctor model capability tests (generative-model aware)
- Patch system with an eval gate

## Known caveats

- The model-based semantic verifier is opt-in and not yet fully validated.
- Dream Mode and Mirror Mode exist only as gated stubs — not implemented.
- Hermes, MCP, and REST adapters are not implemented.
- Retrieval is keyword-based (no vector/semantic retrieval yet).
- Multi-GPU scheduling is not implemented.

## Install

```bash
pip install -e .
heimdal doctor
heimdal run demo
```

See [README.md](README.md) for full usage.
