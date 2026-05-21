# Heimdal Engine Builder Pack v0.2

This folder is intended to be dropped into a new repo and given to Codex, Claude Code, or another coding agent.

## Goal

Build Heimdal Engine: a host-agnostic local agent engine that appears externally as one agent but internally runs a quality-controlled orchestration engine with Context OS, Role Packs, Task Contracts, Repro Packs, Trace Packs, Quality Factory, Truth-first retrieval, Patch system with eval gate, Work/Dream/Mirror scheduling, Model/hardware auto-tuning, and adapters for OpenClaw first but not only OpenClaw.

## Primary target

- Ubuntu native or WSL2
- Ollama backend initially
- Python 3.11+
- SSD minimum, NVMe recommended
- Must work from 1 GPU and scale up to 8 GPUs

## How to use this pack

1. Put this folder in the root of a new repo.
2. Give the coding agent `10_codex_claude_tasks/MASTER_BUILD_PROMPT.md`.
3. Tell it to build in vertical slices, one PR/task at a time.
4. Each slice must pass the acceptance criteria in `11_release_acceptance/BETA_DEFINITION_OF_DONE.md`.

## Important principle

Heimdal is not just an OpenClaw module. Heimdal is a host-agnostic agent engine. OpenClaw is only the first adapter.
