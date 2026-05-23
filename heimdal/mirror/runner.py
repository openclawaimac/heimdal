"""Mirror Mode runner -- one ``heimdal mirror run`` invocation.

The runner enforces every Mirror Mode safety gate before any teacher
provider is touched:

    1. mirror.enabled must be True in the manifest, OR the operator must
       explicitly opt in via the CLI (--enable, on a dry-run, or with a
       stub provider).
    2. privacy_mode=local_only blocks every non-stub/non-manual provider.
       Stub + manual providers run in local_only.
    3. max_teacher_calls_per_run caps how many cases the runner forwards
       to the provider. Anything past the cap is recorded as ``skipped``.
    4. redact_before_send applies redaction.redact_payload to the outbound
       TeacherInput.

The runner writes three durable artifact buckets per run:

    storage/mirror/runs/<id>.json          -- the mirror_run handle.
    storage/mirror/reports/<id>_report.json -- the human-facing report.
    storage/mirror/teacher_outputs/<case_id>.json (when store_teacher_outputs=True)
"""

from __future__ import annotations

import json
import os

from heimdal import __version__, jsonschema_min
from heimdal.config import Config, load_config
from heimdal.ids import new_id, now_iso
from heimdal.mirror import diff_engine, proposal_builder, redaction, selector
from heimdal.mirror.manual_teacher import ManualTeacher
from heimdal.mirror.provider import TeacherInput, TeacherProvider, TeacherResult
from heimdal.mirror.stub_teacher import HallucinatingStub, StubTeacher
from heimdal.storage import Storage

SOURCES = ("recent", "failed", "eval", "dream", "mixed")

DEFAULTS = {
    "max_teacher_calls_per_run": 3,
    "max_input_tokens_per_call": 8000,
    "max_output_tokens_per_call": 2000,
    "redact_before_send": True,
    "store_teacher_outputs": False,
    "store_diffs": True,
    "privacy_mode": "local_only",
    "allowed_sources": ["eval", "recent", "failed", "dream"],
}

# Providers that are safe under privacy_mode=local_only (no network egress).
_LOCAL_PROVIDERS = {"stub", "stub_hallucinator", "manual"}


def _config_value(cfg: dict, key: str):
    return cfg.get(key, DEFAULTS[key])


def _resolve_provider(name: str, *, model: str | None, storage: Storage) -> TeacherProvider:
    """Build a provider by name. Cloud providers are lazy-imported."""
    if name in ("stub", None, ""):
        return StubTeacher(model=model or "heimdal-mirror-stub")
    if name == "stub_hallucinator":
        return HallucinatingStub(model=model or "heimdal-mirror-stub-hallucinator")
    if name == "manual":
        return ManualTeacher(
            storage.path("mirror/teacher_outputs"),
            model=model or "heimdal-mirror-manual",
        )
    if name == "openai":
        from heimdal.mirror.cloud_teacher import OpenAITeacher
        return OpenAITeacher(model=model or "gpt-4o-mini")
    if name == "anthropic":
        from heimdal.mirror.cloud_teacher import AnthropicTeacher
        return AnthropicTeacher(model=model or "claude-3-5-sonnet-latest")
    raise ValueError(f"Unknown teacher provider: {name!r}")


def _privacy_block_reason(*, enabled: bool, privacy_mode: str, provider_name: str,
                          dry_run: bool) -> str | None:
    """The string reason a run is blocked, or None when it may proceed."""
    if dry_run:
        return None
    if provider_name in _LOCAL_PROVIDERS:
        return None
    if not enabled:
        return (
            "Mirror Mode is disabled (manifest mirror.enabled=false). "
            "Use --dry-run, --teacher stub/manual, or enable in manifest."
        )
    if privacy_mode == "local_only":
        return (
            f"privacy_mode=local_only blocks the {provider_name!r} provider. "
            "Set --privacy cloud_allowed (or update the manifest) to proceed."
        )
    return None


def _make_input(case: dict, *, redact: bool) -> tuple[TeacherInput, list[dict]]:
    """Build the redacted TeacherInput sent to the provider."""
    payload = {
        "case_id": case["case_id"],
        "task": case["task"],
        "local_output": case["local_output"],
        "constraints": case["task"].get("constraints", {}),
    }
    if redact:
        cleaned, redactions = redaction.redact_payload(payload)
    else:
        cleaned, redactions = payload, []
    return (
        TeacherInput(
            case_id=cleaned["case_id"],
            task=cleaned["task"],
            local_output=cleaned["local_output"],
            constraints=cleaned["constraints"],
        ),
        redactions,
    )


def _store_teacher_output(storage: Storage, case_id: str, result: TeacherResult) -> str:
    safe = "".join(c for c in str(case_id) if c.isalnum() or c in ("-", "_")) or "case"
    return storage.write_json(
        f"mirror/teacher_outputs/{safe}.json",
        {
            "provider": result.provider,
            "model": result.model,
            "output": result.output,
            "usage": result.usage,
            "metadata": result.metadata,
        },
    )


def run_mirror(
    config: Config | None = None,
    *,
    source: str = "mixed",
    teacher: str | None = None,
    teacher_model: str | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    privacy_override: str | None = None,
    max_teacher_calls: int | None = None,
) -> dict:
    """Execute one Mirror Mode run and return the report dict."""
    if source not in SOURCES:
        raise ValueError(f"Unknown mirror source: {source!r}")

    config = config or load_config()
    cfg = config.mirror or {}
    storage = Storage(config.storage_root).ensure()

    enabled = bool(cfg.get("enabled", False))
    privacy_mode = privacy_override or cfg.get("privacy_mode", DEFAULTS["privacy_mode"])
    provider_name = teacher or cfg.get("teacher_provider") or "stub"
    model = teacher_model or cfg.get("teacher_model")
    max_calls = (
        max_teacher_calls
        if max_teacher_calls is not None
        else _config_value(cfg, "max_teacher_calls_per_run")
    )
    redact = bool(_config_value(cfg, "redact_before_send"))
    store_outputs = bool(_config_value(cfg, "store_teacher_outputs"))
    allowed_sources = set(_config_value(cfg, "allowed_sources"))

    mirror_run_id = new_id("mirrorrun")
    created_at = now_iso()

    # Privacy / enable gate.
    blocked = _privacy_block_reason(
        enabled=enabled, privacy_mode=privacy_mode,
        provider_name=provider_name, dry_run=dry_run,
    )
    if blocked is not None:
        run = _empty_run(
            mirror_run_id, created_at, source, provider_name, model,
            privacy_mode, enabled, dry_run, blocked,
        )
        _persist_run(storage, config, run, report=run)
        return run

    cases = selector.select_cases(
        storage,
        source=source,
        limit=limit or max_calls,
        allowed_sources=allowed_sources,
    )

    provider = _resolve_provider(provider_name, model=model, storage=storage)
    teacher_calls: list[dict] = []
    usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "estimated_cost": 0.0}

    diffs: list[dict] = []
    proposals: list[dict] = []
    for index, case in enumerate(cases):
        if usage["calls"] >= max_calls:
            teacher_calls.append({
                "case_id": case["case_id"], "status": "skipped",
                "reason": "max_teacher_calls_per_run reached",
            })
            continue
        if dry_run:
            teacher_calls.append({
                "case_id": case["case_id"], "status": "dry_run",
                "would_send": {
                    "objective": case["task"]["objective"][:200],
                    "local_output_chars": len(case["local_output"]),
                },
            })
            continue
        teacher_input, redactions = _make_input(case, redact=redact)
        try:
            result = provider.generate(teacher_input)
        except Exception as exc:  # noqa: BLE001 - never crash the run
            teacher_calls.append({
                "case_id": case["case_id"], "status": "error",
                "provider": provider.name, "error": str(exc),
            })
            continue
        call = {
            "case_id": case["case_id"],
            "provider": result.provider,
            "model": result.model,
            "status": result.status,
            "usage": result.usage,
            "redactions": redactions,
        }
        if store_outputs:
            call["teacher_output_ref"] = os.path.relpath(
                _store_teacher_output(storage, case["case_id"], result),
                start=storage.root,
            )
        else:
            call["teacher_output_inline"] = result.output
        teacher_calls.append(call)
        if result.status == "pass":
            usage["calls"] += 1
            usage["input_tokens"] += int(result.usage.get("input_tokens", 0) or 0)
            usage["output_tokens"] += int(result.usage.get("output_tokens", 0) or 0)
            # v0.5.1: per-case diff + proposal generation. Deterministic
            # heuristic scoring -- no extra model call -- so adding it here
            # keeps `mirror run` a single command for the operator.
            diff = diff_engine.compare(
                case_id=case["case_id"],
                local_output=case["local_output"],
                teacher_output=result.output,
                task=case["task"],
                truth_refs=case.get("truth_refs"),
                mirror_run_id=mirror_run_id,
            )
            storage.write_json(
                f"mirror/diffs/{diff['diff_id']}.json", diff,
            )
            diffs.append(diff)
            for proposal in proposal_builder.build_proposals(diff, case=case):
                storage.write_json(
                    f"mirror/proposals/{proposal['id']}.json", proposal,
                )
                proposals.append(proposal)

    summary = _summary(cases, teacher_calls, source, provider_name, dry_run)
    run = {
        "mirror_run_id": mirror_run_id,
        "created_at": created_at,
        "source": source,
        "teacher_provider": provider_name,
        "teacher_model": model,
        "privacy_mode": privacy_mode,
        "enabled": enabled,
        "dry_run": dry_run,
        "cases_selected": [
            {"case_id": c["case_id"], "source_tag": c["source_tag"],
             "verification_status": c["verification_status"],
             "verification_score": c["verification_score"]}
            for c in cases
        ],
        "teacher_calls": teacher_calls,
        "diffs": diffs,
        "proposals": proposals,
        "usage": usage,
        "blocked_reason": None,
        "summary": summary,
        "heimdal_version": __version__,
    }
    _persist_run(storage, config, run, report=run)
    return run


def _empty_run(mirror_run_id, created_at, source, provider_name, model,
               privacy_mode, enabled, dry_run, blocked_reason):
    return {
        "mirror_run_id": mirror_run_id,
        "created_at": created_at,
        "source": source,
        "teacher_provider": provider_name,
        "teacher_model": model,
        "privacy_mode": privacy_mode,
        "enabled": enabled,
        "dry_run": dry_run,
        "cases_selected": [],
        "teacher_calls": [],
        "diffs": [],
        "proposals": [],
        "usage": {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                  "estimated_cost": 0.0},
        "blocked_reason": blocked_reason,
        "summary": f"Mirror Mode blocked: {blocked_reason}",
        "heimdal_version": __version__,
    }


def _summary(cases, calls, source, provider_name, dry_run):
    counts = {"pass": 0, "skipped": 0, "error": 0, "dry_run": 0}
    for call in calls:
        counts[call.get("status", "skipped")] = counts.get(call.get("status", "skipped"), 0) + 1
    return (
        f"Mirror run source={source!r} provider={provider_name!r} "
        f"cases={len(cases)} pass={counts['pass']} "
        f"skipped={counts['skipped']} dry_run={counts['dry_run']} "
        f"errors={counts['error']}"
    )


def _persist_run(storage: Storage, config: Config, run: dict, *, report: dict) -> None:
    jsonschema_min.validate_or_raise(
        run, config.schema_path("mirror_run.schema.json"), "Mirror Run"
    )
    report_path = storage.write_json(
        f"mirror/reports/{run['mirror_run_id']}_report.json", report
    )
    run = dict(run, report_ref=os.path.relpath(report_path, start=storage.root))
    storage.write_json(f"mirror/runs/{run['mirror_run_id']}.json", run)


# -- listing / loading -----------------------------------------------------
def list_mirror_runs(config: Config | None = None) -> list[dict]:
    config = config or load_config()
    storage = Storage(config.storage_root).ensure()
    runs_dir = storage.path("mirror/runs")
    if not os.path.isdir(runs_dir):
        return []
    files = [
        os.path.join(runs_dir, n)
        for n in os.listdir(runs_dir)
        if n.endswith(".json")
    ]
    files.sort(key=os.path.getmtime, reverse=True)
    out: list[dict] = []
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                out.append(json.load(fh))
        except (OSError, ValueError):
            continue
    return out


def load_mirror_report(config: Config, mirror_run_id: str | None = None) -> dict:
    storage = Storage(config.storage_root).ensure()
    reports_dir = storage.path("mirror/reports")
    if mirror_run_id and mirror_run_id != "latest":
        path = os.path.join(reports_dir, f"{mirror_run_id}_report.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Mirror report not found: {mirror_run_id}")
    else:
        path = storage.latest("mirror/reports")
        if not path:
            raise FileNotFoundError("No mirror reports written yet.")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
