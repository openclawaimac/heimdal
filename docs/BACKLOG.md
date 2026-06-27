# Heimdal Backlog

This file drives the autonomous build loop. The loop takes the **topmost
unchecked item** from "Conservative / hardening" first, then "Standard".
It never touches "Needs approval" on its own. Edit this file to steer.

Mark progress inline: `[x]` done, `(skipped: reason)`, `(reverted: reason)`.

Every item must leave the suite green (`python -m unittest discover -s tests
-t .`) and eval at 40/40 (`python -m heimdal eval run --offline`) or be
reverted. Scope bar (do NOT build, skip if an item drifts here): multi-GPU
scheduling, REST/MCP, vector DB, finetuning, systemd/service runner, new
host adapters.

## Conservative / hardening

- [x] Refresh the stale module docstring in `heimdal/cli.py` (it stops at
      v0.5.0 — missing `models`, `profile`, `bridge submit/watch`, `mirror
      diff/proposals/promote-proposal`, `skill bootstrap`).
- [x] `heimdal logs latest` should surface the v0.6.x metrics
      (`runtime_profile`, `profile_source`, `assignment_source`) when present.
      Add a CLI test. (Done: explicit profile/model lines + `--json` mode,
      tests in tests/test_logs.py.)
- [x] Add tests for `heimdal/mirror/redaction.py` edge cases not yet covered
      (multiple secrets in one string, AWS key, bearer header) and for
      `manual_teacher.py` (file present vs absent → skipped). (Done — and the
      bearer-header test surfaced a real gap: `Authorization: Bearer <token>`
      was not redacted; regex fixed.)
- [x] Add a test asserting `heimdal/mirror/cloud_teacher.py` raises
      `CloudProviderUnavailable` (not a bare ImportError/KeyError) when the
      SDK or API key is missing — without importing any real SDK.
- [x] Unify the two hardware-class systems: `profiler.deployment_mode()`
      (Dev/Single Device/Pipeline/Factory) vs
      `capability_matrix.recommend_profile()` (cpu_only/dev/single_gpu/…).
      Keep both public, but have one delegate to the other so they can't
      drift. Add a test pinning the mapping. (Done: deployment_mode now
      delegates to recommend_profile via DEPLOYMENT_LABELS.)
- [x] Harden `config.load_config`: a malformed manifest or a missing schema
      file should fail with a clear, actionable message rather than a raw
      traceback. Add a test. (Done: ConfigError for missing/malformed/
      non-mapping manifest; load_schema raises a clear ValueError naming the
      path.)
- [x] `heimdal models capabilities` on a machine with a stored matrix that
      has zero `model_capabilities` should print a clearer hint (currently
      prints the generic "run doctor" line even when Ollama is reachable).
      (Done: hint now branches on unreachable / no-models / reachable-with-
      models and names the models in the last case.)
- [ ] Audit `tests/test_doctor.py`: it still calls the legacy
      `full_profile()`. Add a parallel test that exercises the v0.6.0
      `capability_matrix.build_matrix()` path so the new path has direct
      coverage even though doctor now uses it.
- [ ] Run `/code-review` (or a manual review) over the v0.5.x mirror modules;
      fix only genuine correctness bugs found, each with a regression test.

## Standard (backlog-driven features, in scope)

- [ ] Add `schemas/openclaw_result.schema.json` for symmetry with
      `hermes_result.schema.json`, and validate the OpenClaw result against
      it in `openclaw_host.handle`. Add tests.
- [ ] `heimdal mirror score <local_file> <teacher_file> --task <task_file>`
      (the optional v0.5.1 CLI) — run the diff engine on two files directly
      without a full mirror run. Add a test.
- [ ] `heimdal dream report --latest` convenience alias (today you must pass
      `--id`); make `report` with no id load the most recent. Add a test.
- [ ] Surface selected-skill refs in `heimdal logs latest` (they are already
      in the repro pack as `selected_skills`). Add a test.
- [ ] `heimdal patch eval` should accept `--targeted` to run only the eval
      categories relevant to the patch type (e.g. retrieval_patch → no_guess
      + must_pass), keeping full-suite as the default. Add a test.

## Needs approval (loop must NOT build these autonomously)

- [ ] **v0.7.0 — Multi-GPU Role Routing.** Per-endpoint routing
      (worker → GPU0, verifier → GPU1), concurrent role execution, parallel
      sample generation for B3/B4. NOT tensor parallelism, NOT a distributed
      cluster. Large architectural change — requires explicit go-ahead.
- [ ] **Brain role actually used at runtime** for B3/B4 planning (currently
      assigned but never invoked). Medium change to the Quality Factory —
      confirm desired behavior first.
- [ ] **Real cloud-teacher validation** with live OpenAI/Anthropic keys.
      Needs credentials + a budget decision; cannot run in CI.
