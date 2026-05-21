# Heimdal Beta Definition of Done

## Functional

- CLI adapter works.
- OpenClaw adapter stub works.
- `heimdal doctor` works without Ollama and with Ollama.
- `heimdal run demo` works.
- `heimdal run --input <task.json>` works.
- Task Contract created for every task.
- Context Packet created for every model call.
- Repro Pack written for every run.
- Trace Pack written for every run.
- Verifier PASS/FAIL works.
- No-Guess Gate works.
- Patch validation works.
- Eval runner works.
- Sandbox policy exists and is enforced minimally.
- Hardware profile is written.

## Architecture

- Heimdal core is host-agnostic.
- OpenClaw is an adapter, not the core.
- Hermes adapter is spec'd, not required.
- Model profiles are configurable.
- Ollama backend is swappable in principle.
- Mirror Mode disabled by default.
- Dream Mode feature-flagged or stubbed.

## Out of scope for beta

Full UI dashboard, full MCP server, full Hermes implementation, advanced vector database, LoRA/finetuning, multi-node cluster, fully autonomous patch merge.
