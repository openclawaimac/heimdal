"""Dream Mode runner -- one ``heimdal dream run`` invocation.

The runner gathers inputs, mines patterns, builds proposals, and writes the
three durable artifacts (the dream run handle, the report, the per-proposal
files). It is offline, deterministic, and never mutates stable state.
"""

from __future__ import annotations

import json
import os

from heimdal import __version__, jsonschema_min
from heimdal.config import Config, load_config
from heimdal.dream import analysis, proposer
from heimdal.ids import new_id, now_iso
from heimdal.storage import Storage

SOURCES = ("failed", "recent", "synthetic", "eval", "mixed")
DEFAULT_LIMIT = 25


def _ensure_storage(config: Config) -> Storage:
    storage = Storage(config.storage_root).ensure()
    return storage


def _write_proposal(storage: Storage, proposal: dict, config: Config) -> str:
    jsonschema_min.validate_or_raise(
        proposal,
        config.schema_path("improvement_proposal.schema.json"),
        "Improvement Proposal",
    )
    return storage.write_json(f"dream/proposals/{proposal['id']}.json", proposal)


def _summary(patterns: list[dict], proposals: list[dict], source: str,
             counts: dict) -> str:
    if not patterns and not proposals:
        return f"Dream run over source={source!r} found no patterns."
    pat = ", ".join(f"{p['category']}({p['count']})" for p in patterns) or "none"
    kinds: dict[str, int] = {}
    for p in proposals:
        kinds[p["kind"]] = kinds.get(p["kind"], 0) + 1
    prop = ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())) or "none"
    return (
        f"Dream run over source={source!r} scanned {counts.get('trace_packs', 0)} "
        f"traces, {counts.get('eval_summaries', 0)} eval summaries, "
        f"{counts.get('bridge_failures', 0)} bridge failures. "
        f"Patterns: {pat}. Proposals: {prop}."
    )


def run_dream(
    config: Config | None = None,
    *,
    source: str = "mixed",
    count: int = 1,
    limit: int = DEFAULT_LIMIT,
) -> dict:
    """Execute one Dream Mode run; return the report dict.

    ``source`` selects which input buckets to scan. ``count`` is the maximum
    number of proposals to emit (after pattern dedup); a synthetic baseline
    proposal is always written when no failures were mined, so every run
    leaves at least one proposal on disk.
    """
    if source not in SOURCES:
        raise ValueError(f"Unknown dream source: {source!r}; one of {SOURCES}.")
    config = config or load_config()
    storage = _ensure_storage(config)

    dream_run_id = new_id("dreamrun")
    created_at = now_iso()

    mining = analysis.gather_inputs(storage, source=source, limit=limit)
    patterns = analysis.detect_patterns(mining)
    proposals = proposer.generate_proposals(patterns)
    if not proposals:
        proposals = [proposer.synthetic_proposal()]
    # Cap to requested count, but keep at least one proposal so the artifact
    # invariant holds.
    if count > 0:
        proposals = proposals[: max(1, count)]

    proposal_refs: list[str] = []
    for proposal in proposals:
        path = _write_proposal(storage, proposal, config)
        proposal_refs.append(
            os.path.relpath(path, start=storage.root)
        )

    actions, risks = proposer.categorize_actions(patterns, proposals)

    report = {
        "dream_run_id": dream_run_id,
        "created_at": created_at,
        "source": source,
        "inputs_analyzed": mining.counts(),
        "failure_patterns": patterns,
        "patch_proposals": [p for p in proposals if p["kind"] == "patch_proposal"],
        "skill_proposals": [p for p in proposals if p["kind"] == "skill_proposal"],
        "eval_case_proposals": [
            p for p in proposals if p["kind"] == "eval_case_proposal"
        ],
        "recommended_actions": actions,
        "risk_notes": risks,
        "summary": _summary(patterns, proposals, source, mining.counts()),
        "heimdal_version": __version__,
    }
    jsonschema_min.validate_or_raise(
        report,
        config.schema_path("dream_report.schema.json"),
        "Dream Report",
    )
    report_path = storage.write_json(
        f"dream/reports/{dream_run_id}_report.json", report
    )

    dream_run = {
        "dream_run_id": dream_run_id,
        "created_at": created_at,
        "source": source,
        "count": len(proposals),
        "inputs_analyzed": mining.counts(),
        "report_ref": os.path.relpath(report_path, start=storage.root),
        "proposal_refs": proposal_refs,
        "heimdal_version": __version__,
    }
    jsonschema_min.validate_or_raise(
        dream_run,
        config.schema_path("dream_run.schema.json"),
        "Dream Run",
    )
    storage.write_json(f"dream/runs/{dream_run_id}.json", dream_run)

    return report


def list_dream_runs(config: Config | None = None) -> list[dict]:
    """Return all dream-run records, newest first."""
    config = config or load_config()
    storage = _ensure_storage(config)
    runs_dir = storage.path("dream/runs")
    if not os.path.isdir(runs_dir):
        return []
    files = []
    for name in os.listdir(runs_dir):
        path = os.path.join(runs_dir, name)
        if os.path.isfile(path) and name.endswith(".json"):
            files.append(path)
    files.sort(key=os.path.getmtime, reverse=True)
    out: list[dict] = []
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                out.append(json.load(fh))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def load_dream_report(config: Config, dream_run_id: str | None = None) -> dict:
    """Load a specific report by id, or the most recent one."""
    storage = _ensure_storage(config)
    reports_dir = storage.path("dream/reports")
    if not os.path.isdir(reports_dir):
        raise FileNotFoundError("No dream reports written yet.")
    if dream_run_id:
        path = os.path.join(reports_dir, f"{dream_run_id}_report.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dream report not found: {dream_run_id}")
    else:
        latest = storage.latest("dream/reports")
        if not latest:
            raise FileNotFoundError("No dream reports written yet.")
        path = latest
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
