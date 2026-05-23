"""Failure-pattern mining for Dream Mode.

Reads recent Trace Packs, eval summaries, and (optionally) bridge failure
reports and groups what it finds into the documented failure-pattern
categories. The mining is deterministic and read-only -- it never touches
the source files.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field

# Failure-pattern categories per the v0.4.0 spec.
CATEGORIES = (
    "missing_source",
    "weak_retrieval",
    "unsupported_claim",
    "format_failure",
    "semantic_miss",
    "verifier_too_strict",
    "verifier_too_lenient",
    "schema_failure",
    "timeout_or_backend_error",
    "adapter_mapping_issue",
    "callback_or_artifact_issue",
    "unknown",
)

# Description text per category, surfaced in the Dream Report.
CATEGORY_DESCRIPTIONS = {
    "missing_source": "Source-required tasks reached the No-Guess Gate with no usable Truth Vault match.",
    "weak_retrieval": "Truth Vault returned a snippet but coverage was below the grounding threshold.",
    "unsupported_claim": "An answer made a factual claim without an aligned retrieved source.",
    "format_failure": "Output did not satisfy the requested output profile or word budget.",
    "semantic_miss": "Hybrid semantic verifier judged the answer unfit for the task.",
    "verifier_too_strict": "Rule-based verifier rejected an answer the host considered acceptable.",
    "verifier_too_lenient": "Rule-based verifier passed an answer that semantic verifier later rejected.",
    "schema_failure": "Output failed JSON schema validation.",
    "timeout_or_backend_error": "Model backend timed out or errored mid-run.",
    "adapter_mapping_issue": "Host adapter translation produced an invalid envelope.",
    "callback_or_artifact_issue": "Callback delivery or artifact write failed.",
    "unknown": "Failure did not match any documented pattern.",
}


@dataclass
class _Hit:
    category: str
    task_id: str
    detail: str
    source_ref: str = ""


@dataclass
class MiningInput:
    trace_packs: list[dict] = field(default_factory=list)
    repro_packs: list[dict] = field(default_factory=list)
    eval_summaries: list[dict] = field(default_factory=list)
    bridge_failures: list[dict] = field(default_factory=list)
    recent_runs: list[dict] = field(default_factory=list)

    def counts(self) -> dict:
        return {
            "trace_packs": len(self.trace_packs),
            "repro_packs": len(self.repro_packs),
            "eval_summaries": len(self.eval_summaries),
            "bridge_failures": len(self.bridge_failures),
            "recent_runs": len(self.recent_runs),
        }


# -- gathering -------------------------------------------------------------
def _read_jsons(directory: str, limit: int) -> list[dict]:
    """Read up to ``limit`` most-recent JSON files in ``directory``."""
    if not os.path.isdir(directory):
        return []
    files = []
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if os.path.isfile(path) and name.endswith(".json"):
            files.append(path)
    files.sort(key=os.path.getmtime, reverse=True)
    out: list[dict] = []
    for path in files[:limit]:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                obj = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(obj, dict):
            obj.setdefault("_source_ref", os.path.relpath(path, start=os.path.dirname(directory)))
            out.append(obj)
    return out


def gather_inputs(storage, *, source: str, limit: int) -> MiningInput:
    """Collect the artifacts Dream Mode should analyze.

    ``source`` controls which buckets are sampled. ``mixed`` (default) gathers
    everything available; the explicit sources are useful for focused runs.
    """
    mining = MiningInput()
    want_recent = source in ("recent", "mixed", "failed")
    want_eval = source in ("eval", "mixed", "failed")
    want_bridge = source in ("failed", "mixed")
    if want_recent:
        mining.trace_packs = _read_jsons(storage.path("logs/trace_packs"), limit)
        mining.repro_packs = _read_jsons(storage.path("logs/repro_packs"), limit)
        mining.recent_runs = mining.trace_packs[:]
    if want_eval:
        eval_runs_dir = storage.path("eval_runs")
        if os.path.isdir(eval_runs_dir):
            for run_name in sorted(os.listdir(eval_runs_dir), reverse=True)[:limit]:
                summary_path = os.path.join(eval_runs_dir, run_name, "summary.json")
                if os.path.exists(summary_path):
                    try:
                        with open(summary_path, "r", encoding="utf-8") as fh:
                            mining.eval_summaries.append(json.load(fh))
                    except (OSError, json.JSONDecodeError):
                        pass
    if want_bridge:
        failed_dir = storage.path("bridge/failed")
        mining.bridge_failures = [
            obj for obj in _read_jsons(failed_dir, limit) if obj.get("status") == "fail"
        ]
    return mining


# -- pattern detection -----------------------------------------------------
def _events(trace: dict) -> list[dict]:
    return trace.get("events", []) or []


def _has_event(trace: dict, name: str) -> bool:
    return any(e.get("name") == name for e in _events(trace))


def _semantic_fail_events(trace: dict) -> list[dict]:
    return [
        e for e in _events(trace)
        if e.get("name") == "semantic_verify"
        and (e.get("data") or {}).get("semantic_verifier_status") == "fail"
    ]


def _trace_task_id(trace: dict) -> str:
    return str(trace.get("task_id", ""))


def detect_patterns(mining: MiningInput) -> list[dict]:
    """Group hits across all mined inputs into stable pattern buckets."""
    hits: dict[str, list[_Hit]] = defaultdict(list)

    for trace in mining.trace_packs:
        task_id = _trace_task_id(trace)
        status = trace.get("status")
        for event in _events(trace):
            name = event.get("name")
            data = event.get("data") or {}
            if name == "no_guess_gate":
                code = data.get("code", "")
                category = (
                    "missing_source" if code in ("", "SOURCE_MISSING") else "weak_retrieval"
                )
                hits[category].append(
                    _Hit(category, task_id, data.get("reason", ""),
                         trace.get("_source_ref", ""))
                )
            elif name == "semantic_verify" and data.get("semantic_verifier_status") == "fail":
                hits["semantic_miss"].append(
                    _Hit("semantic_miss", task_id,
                         f"semantic verifier failed (score={data.get('semantic_verifier_score')})",
                         trace.get("_source_ref", ""))
                )
            elif name == "verify" and data.get("status") == "fail":
                # Schema failures show up as verifier fail with schema metadata
                # in the verification artifact; we tag generically here.
                if status == "fail":
                    hits.setdefault("unknown", []).append(
                        _Hit("unknown", task_id, "task failed verification",
                             trace.get("_source_ref", ""))
                    )

    for summary in mining.eval_summaries:
        for case in summary.get("results", []) or []:
            if case.get("passed"):
                continue
            category = "schema_failure" if case.get("category") == "schema" else (
                "missing_source" if case.get("category") == "no_guess" else "format_failure"
            )
            hits[category].append(
                _Hit(category, case.get("id", ""),
                     f"eval {case.get('category')} expected {case.get('expected')} got {case.get('actual')}",
                     summary.get("eval_run_id", ""))
            )

    for failure in mining.bridge_failures:
        code = failure.get("code", "")
        category = {
            "JOB_SCHEMA_INVALID": "schema_failure",
            "ADAPTER_UNSUPPORTED": "adapter_mapping_issue",
            "OLLAMA_UNREACHABLE": "timeout_or_backend_error",
            "OLLAMA_MODEL_MISSING": "timeout_or_backend_error",
            "CALLBACK_DELIVERY_FAILED": "callback_or_artifact_issue",
        }.get(code, "unknown")
        hits[category].append(
            _Hit(category, failure.get("job_id", ""),
                 f"bridge {code}: {failure.get('error', '')[:200]}",
                 failure.get("bridge", {}).get("input_ref", ""))
        )

    patterns: list[dict] = []
    for category in CATEGORIES:
        bucket = hits.get(category, [])
        if not bucket:
            continue
        patterns.append({
            "category": category,
            "count": len(bucket),
            "description": CATEGORY_DESCRIPTIONS[category],
            "examples": [
                {
                    "task_id": h.task_id,
                    "detail": h.detail,
                    "source_ref": h.source_ref,
                }
                for h in bucket[:5]
            ],
        })
    return patterns
