# Heimdal Engine Overview

Heimdal Engine is a host-agnostic local agent engine. Externally it appears as a
single agent, while internally it runs a quality-controlled orchestration
runtime. It surrounds smaller local language models with context discipline,
retrieval, verification, evaluation, and hardware-aware scheduling. Heimdal
builds a Task Contract for every task and a Context Packet for every model call.
A Quality Factory drafts an answer, verifies it, and repairs it when
verification fails. A No-Guess Gate stops the engine from inventing facts when
sources are missing. Every run produces a Repro Pack and a Trace Pack so results
are reproducible and auditable. Heimdal runs on CPU-only machines and scales
across multiple GPUs, and OpenClaw is only its first host adapter.
