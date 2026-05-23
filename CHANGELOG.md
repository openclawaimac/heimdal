# Changelog

## v0.4.0 — v0.4.2 — Self-Improvement Core

Three coordinated releases that turn Heimdal from a runtime into a system that
mines its own history for improvements -- without ever silently changing
stable behavior.

### v0.4.0 — Dream Mode (commit `ad53be0`)

Heimdal can now scan its own past Trace Packs, eval summaries, and bridge
failure reports for recurring failure patterns, and emit structured
improvement proposals into `storage/dream/`. Dream Mode is invocation-driven
(`heimdal dream run`), offline, read-only against stable state, and never
applies anything automatically.

    heimdal dream run [--source failed|recent|eval|mixed] [--count N]
    heimdal dream report [--id <dream_run_id>]
    heimdal dream list

Schemas: `dream_run`, `dream_report`, `improvement_proposal`.
Storage: `storage/dream/{runs,reports,proposals,...}`.
A synthetic baseline proposal is always written so every run leaves at least
one artifact.

### v0.4.1 — Patch promotion + eval review (commit `ceaef43`)

The patch system grows from "validate JSON" into a real lifecycle:

    experimental -> beta -> stable
                 -> rejected
                 -> archived

    heimdal patch list [--channel <name>]
    heimdal patch show <patch_id>
    heimdal patch review <patch_id>
    heimdal patch eval <patch_id>
    heimdal patch promote <patch_id> --to beta|stable
    heimdal patch reject <patch_id> --reason "..."

Promotion to beta requires `intent`; promotion to stable requires a passing
candidate eval (`patches/evals/<id>.eval.json`) **and** the must-pass eval
suite green. Reviews and candidate evals are persisted alongside patches
under `storage/patches/{reviews,evals}/`. Patch types expand with
`eval_case_patch` (the only type currently allowed to auto-apply); all
others stay review-only.

### v0.4.2 — Skill Library 2.0

Skills are now versioned JSON bundles under `storage/skills/<role>/<id>.json`
with triggers, instructions, optional rubric/examples, and durable per-skill
performance stats. Context OS uses a real registry-driven selector that:

- ranks role-listed candidates first, then role-matched registry skills with
  trigger overlap on the instruction;
- caps the per-run count by hardware deployment mode
  (3 / 5 / 5 / 7 for Dev / Single Device / Pipeline / Factory);
- never injects an irrelevant skill even when there is budget room;
- records `uses` / `passes` / `fails` / `last_used` per skill after each run.

    heimdal skill list [--json]
    heimdal skill show <skill_id>
    heimdal skill search "<query>"
    heimdal skill validate <skill_file>
    heimdal skill install <skill_file>
    heimdal skill archive <skill_id>
    heimdal skill stats

Seven seed skills ship under `examples/skills/<role>/`:
`general.no_guess_answering`, `research.source_grounded_summary`,
`research.truth_vault_qa`, `verifier.semantic_quality_check`,
`ops.incident_summary`, `coding.bugfix_loop`,
`business.pricing_policy_explanation`. First runtime boot copies them into
`storage/skills/` (recursive seed walk, layout-preserving).

Schema: `schemas/skill.schema.json`. The legacy SKILL_LIBRARY built-ins are
kept as a fallback so v0.2.x role packs that name `concise_writing` /
`structured_answer` still resolve.

### Combined safety guarantees

- Dream Mode never mutates stable state.
- Patch promotion to stable is explicit, never automatic.
- No Mirror Mode, MCP, REST server, vector DB, multi-GPU work in this block.

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

The shipped examples (`examples/bridge/{hermes,openclaw,heimdal}_task.json`)
no longer pin a backend in their `runtime` block, so CLI flags act as the
defaults. On a machine with Ollama installed and `qwen2.5:7b` pulled:

    heimdal bridge init
    heimdal bridge submit --input examples/bridge/hermes_task.json
    heimdal bridge once --backend ollama --model qwen2.5:7b --verifier hybrid
    cat storage/bridge/outbox/job-hermes-001.result.json

Expected on success:

- `status` == `pass`
- `result.metrics.backend` == `ollama`
- `result.metrics.worker_model` == `qwen2.5:7b`
- `result.metrics.verifier_backend` == `hybrid`
- `result.metrics.semantic_verifier_model` == `qwen2.5:7b`
- `repro_pack_ref` and `trace_pack_ref` present, relative
- No absolute paths or internal artifacts (`context_packet`,
  `task_contract`) in the result

To run the same examples offline (CI, smoke), pass `--offline` (or
`--backend offline`) to `bridge once`; the examples carry no offline-pin
of their own, so the CLI flag wins.

### Manual testplan notes

When dropping jobs into the inbox by hand for failure-path verification:

- **Invalid JSON / unsupported adapter:** save the file as
  `<name>.ready.json` so the readiness check fires immediately, or use a
  plain `.json` filename and wait 1+ second before running `bridge once`
  (the bridge only reads files older than `MIN_FILE_AGE_SECONDS = 1.0`).
- **Unsupported-adapter test:** the job must still be a valid Bridge Job
  Envelope -- a bare `{"adapter": "mystery"}` is rejected as
  `JOB_SCHEMA_INVALID`, not `ADAPTER_UNSUPPORTED`. Use:

      {
        "job_id": "unsupported-001",
        "host": "mystery_host",
        "adapter": "mystery",
        "payload": {}
      }
