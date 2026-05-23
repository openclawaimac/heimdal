# Changelog

## v0.3.0 — Host Invocation File Bridge

Heimdal exposes a local file bridge so external local agents can invoke it
without importing Python or running a server. Drop a JSON job into
`storage/bridge/inbox/` and Heimdal writes a result JSON to
`storage/bridge/outbox/`.

### Bridge inbox / outbox flow

    heimdal bridge init                                 # create the bridge tree
    heimdal bridge submit --input <job.json>            # write a job into the inbox
    heimdal bridge once --backend offline               # process ready jobs and exit
    heimdal bridge watch --backend ollama --model qwen2.5:7b --verifier hybrid
    heimdal bridge status                               # counts in each subdir

A job moves `inbox -> processing -> {archive | failed}` with atomic
`os.replace`. The bridge waits for a `.ready.json` suffix (or a plain `.json`
that has been on disk at least one second) so a producer is never read
mid-write. Original jobs are never deleted without leaving an archive or
failure record. Result filenames are sanitized from `job_id` so a hostile
input can't escape the bridge tree.

### Supported payloads

A job selects an `adapter` and embeds the matching payload:

| `adapter`   | Payload                | Result shape                      |
|-------------|------------------------|-----------------------------------|
| `hermes`    | Hermes-style payload   | Hermes result (schema-validated)  |
| `openclaw`  | OpenClaw-style payload | OpenClaw result                   |
| `heimdal`   | Heimdal Host Envelope  | Sanitized Heimdal Result Envelope |

The legacy v0.2.8 name `"generic"` is accepted as a quiet alias for
`"heimdal"` and may be removed in a future release. Example jobs:
`examples/bridge/{heimdal,hermes,openclaw}_task.json`.

### Host-safe output

Every outbox file -- regardless of adapter -- is sanitized before being
written:

- `trace_pack_ref` and `repro_pack_ref` are **relative refs** (e.g.
  `logs/trace_packs/<id>.json`), never absolute filesystem paths.
- Artifacts are `{type, ref}` entries; the `context_packet` and
  `task_contract` artifacts are dropped as internal-only.
- The `bridge` wrapper carries `processed_at`, `duration_ms`, `input_ref`,
  and `output_ref` -- all relative to the storage root.
- No raw prompts, no full Context Packet contents, no internal sub-agent
  graph appears in the host-visible result.

### Failure handling

Transport-layer failures move the job to `failed/` and write a
machine-readable `<job_id>.error.json` with a stable `code`:

- `JOB_SCHEMA_INVALID` -- bad JSON or missing required fields.
- `ADAPTER_UNSUPPORTED` -- `adapter` not in `{hermes, openclaw, heimdal}`.
- `OLLAMA_UNREACHABLE` / `OLLAMA_MODEL_MISSING` -- pre-flight backend check
  fails the job cleanly without crashing the loop.
- `INTERNAL_ERROR` -- any other unexpected exception; the loop continues.

### No server required

The bridge is filesystem-only. There is no socket, no HTTP listener, no
broker. An external agent only needs to write a file and read a file.

### Known caveats

- The bridge spawns a fresh `Runtime` per job. Fine at typical bridge
  volumes; for high-throughput identical-config workloads a future revision
  could memoize one Runtime per `(backend, model, verifier)`.
- The Heimdal-adapter result still includes the Verification Result and
  metrics; only the Context Packet and Task Contract are dropped. Hosts that
  want a strictly minimal payload should prefer the Hermes adapter.
- Real-Ollama validation must be run locally (the upstream CI environment
  does not have Ollama installed).

### Local real-Ollama validation

On a machine with Ollama installed and `qwen2.5:7b` pulled:

    heimdal bridge init
    heimdal bridge submit --input examples/bridge/hermes_task.json
    heimdal bridge once --backend ollama --model qwen2.5:7b --verifier hybrid
    cat storage/bridge/outbox/job-hermes-001.result.json

Expected on success:

- `status` == `pass`
- `result.metrics.backend` == `ollama`
- `result.metrics.worker_model` == `qwen2.5:7b`
- `result.metrics.verifier_backend` == `hybrid`
- `repro_pack_ref` and `trace_pack_ref` present, relative
- No absolute paths or internal artifacts (`context_packet`,
  `task_contract`) in the result
