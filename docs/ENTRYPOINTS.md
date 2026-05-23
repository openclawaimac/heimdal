# Heimdal Entrypoints

Heimdal exposes several CLIs depending on who is calling it. Internal callers
(developers, scripts) get the raw Result Envelope; external hosts (Hermes,
OpenClaw) get a host-safe envelope with relative refs and no internal leakage.

## `heimdal run` -- generic / internal

The generic Heimdal runner. Takes a Host Task Envelope or a plain instruction
and emits the internal Result Envelope (absolute artifact paths included).
Use this for local debugging or in-tree tooling -- **not** for external host
integrations.

    heimdal run --input <envelope.json>
    heimdal run --instruction "Explain queues."

## `heimdal hermes run` -- Hermes-safe host result

Drives Heimdal from a Hermes-style payload and returns a Hermes-safe result
envelope validated against `schemas/hermes_result.schema.json`:

- Machine-readable `code` (e.g. `OK`, `SOURCE_MISSING`, `VERIFIER_SEMANTIC_FAIL`).
- Structured `needed_inputs` on `need_input`, with a concise `missing_topic`.
- Host-safe relative refs for `repro_pack_ref`, `trace_pack_ref`, artifacts,
  and `callback_delivery.target_ref`.
- Internal-only artifacts (`context_packet`, `task_contract`) are dropped.
- No raw prompts, no full Context Packet, no internal sub-agent graph.

Callback files land under `storage/workspace/<file>`.

    heimdal hermes run --input <hermes_payload.json>

## `heimdal openclaw run` -- OpenClaw-safe host result

Same shape and safety guarantees as `hermes run`, for OpenClaw payloads
(`outcome` instead of `status`). No formal schema yet.

    heimdal openclaw run --input <openclaw_payload.json>

## `heimdal verify --task --answer` -- verifier-only

The host has drafted its own candidate answer and only wants Heimdal to grade
it. Returns `status` (`pass`/`fail`), `code`, defects, and host-safe Repro /
Trace refs for the verification run.

    heimdal verify --task <envelope.json> --answer <answer.json> \
        --backend ollama --model qwen2.5:7b --verifier hybrid --json

The answer file is JSON; the candidate text is read from `answer["answer"]`
when present, otherwise from the file as a raw string.

## `heimdal hermes capabilities` (`heimdal openclaw capabilities`)

Reports supported backends, verifiers, and feature flags so a host can
discover what this Heimdal install supports before sending a payload.

    heimdal hermes capabilities --json

## `heimdal bridge` -- local file bridge for external local agents

A transport layer: an external local agent (e.g. another Hermes/OpenClaw
process) drops a JSON job into `<storage>/bridge/inbox/` and Heimdal writes
the result to `<storage>/bridge/outbox/`. The bridge is transport-only --
each job is routed to the existing Hermes / OpenClaw / generic handler; no
Quality Factory logic lives here.

    heimdal bridge init                            # create the bridge dirs
    heimdal bridge submit --input <job.json>       # write a job into the inbox
    heimdal bridge once --backend offline          # process one batch and exit
    heimdal bridge watch --backend ollama --model qwen2.5:7b --verifier hybrid
    heimdal bridge status                          # show counts in each dir

A job file (`examples/bridge/{heimdal,hermes,openclaw}_task.json`) selects an
`adapter` (`hermes`, `openclaw`, or `heimdal`) and embeds the corresponding
payload. Drop it into the inbox with the `.ready.json` suffix to mark it
complete (or a plain `.json` that's been on disk for at least a second). The
bridge moves the job through `inbox -> processing -> {archive | failed}`
and writes one of:

- `outbox/<job_id>.result.json` (success) -- carries the adapter result plus
  host-safe `trace_pack_ref` / `repro_pack_ref` and a `bridge` block.
- `failed/<job_id>.error.json` (failure) -- carries a machine-readable
  `code`: `JOB_SCHEMA_INVALID`, `ADAPTER_UNSUPPORTED`, `OLLAMA_UNREACHABLE`,
  `OLLAMA_MODEL_MISSING`, `INTERNAL_ERROR`.

## `heimdal hermes doctor` (`heimdal openclaw doctor`)

Integration diagnostics. Validates the payload, checks storage writability,
backend reachability, the Hermes result schema, and runs an end-to-end probe
to confirm the host result is schema-valid and free of absolute paths or
internal-only artifacts.

    heimdal hermes doctor --input <payload.json> --backend offline --json

Output is machine-readable JSON: `{status, checks, warnings, suggested_fixes}`
with `status` one of `pass | warning | fail`. Exits non-zero only on `fail`.

## Quick recommendation

| Caller | Use |
|---|---|
| Internal scripts / debugging | `heimdal run` |
| Hermes host integration | `heimdal hermes run` (+ `capabilities`, `doctor`) |
| OpenClaw host integration | `heimdal openclaw run` (+ `capabilities`, `doctor`) |
| Host grades its own answer | `heimdal verify` |
| External agent without Python / server | `heimdal bridge` |
