# Master Build Prompt for Codex / Claude Code

You are building Heimdal Engine from scratch. Read this builder pack first. Treat it as the source of truth.

## Core goal

Build a host-agnostic local agent engine that appears externally as one agent but internally runs a quality-controlled orchestration runtime.

Initial target: Python 3.11+, Ubuntu native + WSL2, Ollama backend, CLI adapter + OpenClaw adapter stub, SSD minimum, 1 GPU useful minimum, scales to 8 GPUs.

## Non-negotiable architecture

Host Adapter Contract, Task Contract, Context Packet, Repro Pack, Trace Pack, Quality Factory, Verifier PASS/FAIL, No-Guess Gate, Hardware Profiler, Model Profiles, Patch System, Eval Gate, Sandbox Policy.

## Build method

Use vertical slices. Do not build everything in one huge commit. For each task: implement minimal functionality, add/update tests, add smoke command, ensure CLI still works, avoid unnecessary dependencies.

## Definition of Done

See `11_release_acceptance/BETA_DEFINITION_OF_DONE.md`.

## Constraints

Do not tightly couple Heimdal to OpenClaw. OpenClaw is only an adapter. Mirror Mode disabled by default. Dream Mode can be feature-flagged. All outputs that claim PASS must have Repro + Trace logs.
