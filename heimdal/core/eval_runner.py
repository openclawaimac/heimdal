"""Eval Runner and Eval Gate.

Runs the eval suite (smoke / must-pass / schema / no-guess) through the real
runtime and writes a summary JSON. No patch may enter ``stable`` unless every
must-pass eval passes (docs/builder_pack/07_quality_eval/EVAL_GATE.md).
"""

from __future__ import annotations

import json
import os

from heimdal import __version__
from heimdal.core.constants import PASS
from heimdal.core.runtime import Runtime
from heimdal.ids import new_id, now_compact, now_iso, repo_root
from heimdal.storage import Storage

EVAL_DIR = "eval"
CATEGORY_FILES = {
    "smoke": "smoke.json",
    "must_pass": "must_pass.json",
    "schema": "schema.json",
    "no_guess": "no_guess.json",
}
# Minimum suite size per docs/builder_pack/07_quality_eval/EVAL_GATE.md.
MINIMUMS = {"smoke": 20, "must_pass": 10, "schema": 5, "no_guess": 5}
REGRESSION_TOLERANCE = 0.05


def _build_envelope(case: dict, category: str) -> dict:
    """Expand a compact eval case into a full Host Task Envelope."""
    role_id = case.get("role_id", "general")
    expected_outputs = case.get("expected_outputs", ["markdown_response"])
    return {
        "host": {
            "type": "cli",
            "host_task_id": f"eval-{case['id']}",
            "source_agent": "eval_runner",
            "callback": {},
        },
        "role_binding": {
            "role_id": role_id,
            "risk_mode": case.get("risk_mode", "balanced"),
            "privacy_mode": "local_only",
            "output_profiles": case.get("output_profiles", ["markdown"]),
        },
        "task_request": {
            "task_id": f"eval-{case['id']}",
            "title": case.get("title", case["id"]),
            "instruction": case["instruction"],
            "inputs": {},
            "constraints": case.get("constraints", {}),
            "priority": "P2",
            "budget": {"quality_level": case.get("quality_level", "B1")},
            "expected_outputs": expected_outputs,
        },
        "runtime_hints": {"category": category},
    }


def load_suite(eval_dir: str | None = None) -> dict[str, list[dict]]:
    """Load eval cases grouped by category."""
    base = eval_dir or os.path.join(repo_root(), EVAL_DIR)
    suite: dict[str, list[dict]] = {}
    for category, filename in CATEGORY_FILES.items():
        try:
            suite[category] = Storage.read_json(os.path.join(base, filename))
        except FileNotFoundError:
            suite[category] = []
    return suite


def _previous_pass_rate(runtime: Runtime) -> float | None:
    """Pass rate of the most recent eval run, used for regression detection."""
    runs_dir = runtime.storage.path("eval_runs")
    if not os.path.isdir(runs_dir):
        return None
    summaries = [
        os.path.join(runs_dir, name, "summary.json")
        for name in os.listdir(runs_dir)
        if os.path.exists(os.path.join(runs_dir, name, "summary.json"))
    ]
    if not summaries:
        return None
    latest = max(summaries, key=os.path.getmtime)
    try:
        return Storage.read_json(latest).get("pass_rate")
    except (OSError, json.JSONDecodeError):
        return None


def _runtime_metadata(runtime: Runtime, sample_metrics: dict) -> dict:
    """Backend/model/platform facts describing how an eval run was executed."""
    backend = runtime.backend.name
    return {
        "heimdal_version": __version__,
        "backend": backend,
        "worker_model": sample_metrics.get("worker_model"),
        "verifier_backend": sample_metrics.get("verifier_backend"),
        "verifier_model": sample_metrics.get("verifier_model"),
        "ollama_endpoint": (
            runtime.config.ollama.get("base_url") if backend == "ollama" else None
        ),
        "manifest_path": runtime.config.manifest_path,
        "platform": runtime.hardware_profile,
    }


def run_evals(runtime: Runtime | None = None, eval_dir: str | None = None) -> dict:
    """Run the full eval suite and write a summary JSON."""
    runtime = runtime or Runtime()
    suite = load_suite(eval_dir)
    prior_rate = _previous_pass_rate(runtime)

    results: list[dict] = []
    category_stats: dict[str, dict] = {}
    sample_metrics: dict = {}
    for category, cases in suite.items():
        passed = 0
        for case in cases:
            envelope = _build_envelope(case, category)
            try:
                result = runtime.run_envelope(envelope)
                actual = result["status"]
                error = None
                if not sample_metrics:
                    sample_metrics = result.get("metrics", {})
            except Exception as exc:  # noqa: BLE001 - eval must not crash the suite
                actual = "error"
                error = str(exc)
            expected = case.get("expect_status", PASS)
            ok = actual == expected
            passed += int(ok)
            results.append(
                {
                    "id": case["id"],
                    "category": category,
                    "expected": expected,
                    "actual": actual,
                    "passed": ok,
                    "error": error,
                }
            )
        category_stats[category] = {
            "total": len(cases),
            "passed": passed,
            "minimum": MINIMUMS.get(category, 0),
            "meets_minimum": len(cases) >= MINIMUMS.get(category, 0),
        }

    total = len(results)
    total_passed = sum(1 for r in results if r["passed"])
    pass_rate = round(total_passed / total, 4) if total else 0.0
    must_pass = category_stats["must_pass"]
    must_pass_all = must_pass["total"] > 0 and must_pass["passed"] == must_pass["total"]
    regressed = bool(
        prior_rate is not None and pass_rate < prior_rate - REGRESSION_TOLERANCE
    )

    summary = {
        "eval_run_id": new_id("evalrun"),
        "timestamp": now_iso(),
        "metadata": _runtime_metadata(runtime, sample_metrics),
        "total": total,
        "passed": total_passed,
        "pass_rate": pass_rate,
        "categories": category_stats,
        "must_pass_all_passed": must_pass_all,
        "prior_pass_rate": prior_rate,
        "regressed": regressed,
        "results": results,
    }

    run_dir = f"eval_runs/{now_compact()}_{summary['eval_run_id']}"
    summary["summary_path"] = runtime.storage.write_json(
        f"{run_dir}/summary.json", summary
    )
    return summary
