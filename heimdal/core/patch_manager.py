"""Patch Manager.

Validates Heimdal patches and enforces the channel rule: no patch reaches the
``stable`` channel without a passing eval run
(docs/builder_pack/09_storage_context/PATCH_SYSTEM.md).
"""

from __future__ import annotations

import json
import os

from heimdal import jsonschema_min

CHANNELS = ["experimental", "beta", "stable", "rejected"]
PATCH_TYPES = [
    "prompt_patch",
    "rubric_patch",
    "skill_patch",
    "retrieval_patch",
    "scheduler_patch",
    "sandbox_patch",
    "model_profile_patch",
]


class PatchError(ValueError):
    """Raised when a patch is invalid or cannot be promoted."""


def load_patch(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def validate_patch(patch: dict, config) -> list[str]:
    """Return a list of validation errors for a patch object."""
    schema = jsonschema_min.load_schema(config.schema_path("patch.schema.json"))
    return jsonschema_min.validate(patch, schema)


def validate_patch_file(path: str, config) -> tuple[bool, list[str]]:
    """Validate a patch file. Returns (is_valid, errors)."""
    if not os.path.exists(path):
        return False, [f"patch file not found: {path}"]
    try:
        patch = load_patch(path)
    except json.JSONDecodeError as exc:
        return False, [f"patch file is not valid JSON: {exc}"]
    errors = validate_patch(patch, config)
    return (not errors), errors


def can_promote_to_stable(patch: dict, eval_summary: dict | None) -> tuple[bool, str]:
    """A patch may enter ``stable`` only if its eval run passed must-pass evals."""
    if eval_summary is None:
        return False, "No eval run attached; stable promotion requires a passing eval run."
    if not eval_summary.get("must_pass_all_passed", False):
        return False, "Eval run did not pass all must-pass evals."
    if eval_summary.get("regressed", False):
        return False, "Eval run regressed beyond tolerance."
    return True, "Eval gate passed."


def promote(patch: dict, channel: str, eval_summary: dict | None, config) -> dict:
    """Return a copy of ``patch`` moved to ``channel``, enforcing the eval gate."""
    if channel not in CHANNELS:
        raise PatchError(f"Unknown channel '{channel}'.")
    errors = validate_patch(patch, config)
    if errors:
        raise PatchError("Invalid patch: " + "; ".join(errors))
    if channel == "stable":
        ok, reason = can_promote_to_stable(patch, eval_summary)
        if not ok:
            raise PatchError(reason)
    promoted = dict(patch)
    promoted["channel"] = channel
    if eval_summary is not None:
        promoted["eval_run"] = eval_summary.get("eval_run_id")
    return promoted
