"""Case selection for Mirror Mode.

Walks ``storage/artifacts/run_*`` because the run dir is the only place
that bundles the local response text, the task contract, and the
verification result for the same run. Trace packs alone don't carry the
output text and re-running every task to fetch it would be expensive.

A case is candidate-shaped:

    {
      "case_id": "<task_id>",
      "task": {"objective": "...", "constraints": {...}, "role_id": "..."},
      "local_output": "<response markdown>",
      "verification_status": "pass" | "fail" | "need_input",
      "verification_score": 0.0,
      "truth_refs": [...],
      "source_tag": "recent|failed|eval|dream",
      "run_dir": "<absolute path>",
    }
"""

from __future__ import annotations

import json
import os
from typing import Iterable

from heimdal.storage import Storage


def _safe_read_json(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _candidate_run_dirs(storage: Storage, *, scan_limit: int) -> list[str]:
    root = storage.path("artifacts")
    if not os.path.isdir(root):
        return []
    dirs = [
        os.path.join(root, name)
        for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name))
    ]
    dirs.sort(key=os.path.getmtime, reverse=True)
    return dirs[:scan_limit]


def _load_case(run_dir: str) -> dict | None:
    contract = _safe_read_json(os.path.join(run_dir, "task_contract.json"))
    verification = _safe_read_json(os.path.join(run_dir, "verification_result.json"))
    response_path = os.path.join(run_dir, "response.md")
    if not (contract and os.path.isfile(response_path)):
        return None
    try:
        with open(response_path, "r", encoding="utf-8") as fh:
            local_output = fh.read()
    except OSError:
        return None
    return {
        "case_id": contract.get("task_id", os.path.basename(run_dir)),
        "task": {
            "objective": contract.get("objective", ""),
            "title": (contract.get("definition_of_done") or [""])[0],
            "constraints": contract.get("constraints", {}),
            "role_id": contract.get("role_id", "general"),
            "expected_outputs": contract.get("expected_outputs", []),
        },
        "local_output": local_output,
        "verification_status": (verification or {}).get("status", "unknown"),
        "verification_score": (verification or {}).get("score", 0.0),
        "truth_refs": [],
        "run_dir": run_dir,
        "source_tag": "recent",
    }


def _filter_for_source(cases: list[dict], source: str) -> list[dict]:
    if source == "recent":
        return cases
    if source == "failed":
        return [c for c in cases if c["verification_status"] in ("fail", "need_input")]
    if source == "eval":
        return [c for c in cases if str(c["case_id"]).startswith("eval-")]
    if source == "dream":
        # The Dream filter happens up-stack: the runner intersects case ids
        # with the latest dream report's pattern examples. From the selector
        # angle, "dream" just means "include everything; let the runner
        # narrow it" -- so we return cases unchanged here.
        return cases
    if source == "mixed":
        return cases
    return cases


def select_cases(
    storage: Storage,
    *,
    source: str = "mixed",
    limit: int = 3,
    scan_limit: int | None = None,
    allowed_sources: Iterable[str] | None = None,
) -> list[dict]:
    """Pick at most ``limit`` recent cases that have a local response artifact.

    ``allowed_sources`` is the manifest's ``mirror.allowed_sources``. The
    selector silently drops any case whose ``source_tag`` isn't on that
    list, so a privacy-tight config can confine Mirror to e.g. eval cases.
    """
    if source not in {"recent", "failed", "eval", "dream", "mixed"}:
        raise ValueError(f"Unknown mirror source: {source!r}")
    scan = scan_limit or max(limit * 8, 16)
    cases: list[dict] = []
    for run_dir in _candidate_run_dirs(storage, scan_limit=scan):
        case = _load_case(run_dir)
        if case is None:
            continue
        if source == "failed" and case["verification_status"] not in ("fail", "need_input"):
            continue
        if source == "eval" and not str(case["case_id"]).startswith("eval-"):
            continue
        case["source_tag"] = (
            "failed" if case["verification_status"] in ("fail", "need_input")
            else ("eval" if str(case["case_id"]).startswith("eval-") else "recent")
        )
        if allowed_sources is not None and case["source_tag"] not in allowed_sources:
            continue
        cases.append(case)
        if len(cases) >= limit:
            break
    return _filter_for_source(cases, source)[:limit]
