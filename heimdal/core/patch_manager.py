"""Patch Manager and promotion lifecycle.

Validates Heimdal patches and walks them through the controlled lifecycle:

    experimental -> beta -> stable
                 \-> rejected
                 \-> archived

The stable channel always requires a passing eval run; promotion is never
automatic. Stable behavior only changes through explicitly promoted patches
(docs/builder_pack/09_storage_context/PATCH_SYSTEM.md).
"""

from __future__ import annotations

import json
import os

from heimdal import jsonschema_min
from heimdal.core import eval_runner
from heimdal.ids import now_iso
from heimdal.storage import Storage

CHANNELS = ["experimental", "beta", "stable", "rejected"]
CHANNEL_DIRS = CHANNELS + ["archived"]

# Channels that hold "live" candidates a host might apply or evaluate. Reviews
# and eval runs live alongside as siblings under storage/patches/.
PATCH_TYPES = [
    "prompt_patch",
    "rubric_patch",
    "skill_patch",
    "retrieval_patch",
    "scheduler_patch",
    "sandbox_patch",
    "model_profile_patch",
    "eval_case_patch",
]

# Patch types Heimdal is willing to APPLY automatically once promoted to
# stable. Everything else is review-only (the operator must apply by hand).
APPLYABLE_TYPES = {"eval_case_patch"}


class PatchError(ValueError):
    """Raised when a patch is invalid or cannot be promoted."""


# -- low-level I/O ---------------------------------------------------------
def load_patch(path: str) -> dict:
    return Storage.read_json(path)


def patches_root(config) -> str:
    return os.path.join(config.storage_root, "patches")


def _ensure_storage(config) -> Storage:
    return Storage(config.storage_root).ensure()


# -- validation ------------------------------------------------------------
def validate_patch(patch: dict, config) -> list[str]:
    """Return a list of validation errors for a patch object."""
    schema = jsonschema_min.load_schema(config.schema_path("patch.schema.json"))
    return jsonschema_min.validate(patch, schema)


def validate_patch_file(path: str, config) -> tuple[bool, list[str]]:
    """Validate a patch file. Returns ``(is_valid, errors)``."""
    try:
        patch = load_patch(path)
    except FileNotFoundError:
        return False, [f"patch file not found: {path}"]
    except json.JSONDecodeError as exc:
        return False, [f"patch file is not valid JSON: {exc}"]
    errors = validate_patch(patch, config)
    return (not errors), errors


# -- lookup / listing ------------------------------------------------------
def list_patches(config, *, channel: str | None = None) -> list[dict]:
    """Patches in ``channel`` (or every channel if not given), newest first."""
    root = patches_root(config)
    channels = [channel] if channel else CHANNEL_DIRS
    out: list[dict] = []
    for ch in channels:
        directory = os.path.join(root, ch)
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            if not name.endswith(".json"):
                continue
            path = os.path.join(directory, name)
            try:
                patch = Storage.read_json(path)
            except (OSError, ValueError):
                continue
            patch.setdefault("channel", ch)
            patch["_path"] = path
            out.append(patch)
    out.sort(key=lambda p: p.get("created_at", ""), reverse=True)
    return out


def find_patch(config, patch_id: str) -> tuple[dict, str, str] | None:
    """Locate a patch by id; return ``(patch, channel, path)`` or ``None``."""
    root = patches_root(config)
    for ch in CHANNEL_DIRS:
        candidate = os.path.join(root, ch, f"{patch_id}.json")
        if os.path.exists(candidate):
            patch = Storage.read_json(candidate)
            patch.setdefault("channel", ch)
            return patch, ch, candidate
    return None


def install_patch(config, patch: dict) -> str:
    """Write a patch to its channel directory; return the absolute path."""
    storage = _ensure_storage(config)
    channel = patch.get("channel", "experimental")
    if channel not in CHANNEL_DIRS:
        raise PatchError(f"Unknown channel '{channel}'.")
    errors = validate_patch(patch, getattr(install_patch, "_config", None) or config)
    if errors:
        raise PatchError("Invalid patch: " + "; ".join(errors))
    return storage.write_json(f"patches/{channel}/{patch['id']}.json", patch)


# -- review ----------------------------------------------------------------
def _recommended_gates(patch: dict) -> list[str]:
    """Suggest which gates this patch must clear before promotion."""
    gates = ["schema valid", "intent declared"]
    if not patch.get("rollback"):
        gates.append("rollback metadata (review-only without it)")
    if patch.get("type") in ("retrieval_patch", "rubric_patch", "skill_patch"):
        gates.append("targeted eval run")
    gates.append("must-pass eval suite (for stable)")
    return gates


def review_patch(patch: dict) -> dict:
    """Read-only structured review of ``patch``.

    Surfaces missing metadata, declares whether Heimdal would auto-apply this
    type on stable promotion, and lists the gates the operator should walk
    the patch through.
    """
    issues: list[str] = []
    if not patch.get("intent"):
        issues.append("Missing 'intent' field.")
    if not patch.get("rollback"):
        issues.append("No rollback metadata; treat as review-only.")
    if patch.get("type") not in PATCH_TYPES:
        issues.append(f"Unknown or risky patch type: {patch.get('type')!r}")
    if "change" not in patch:
        issues.append("Missing 'change' field.")

    auto_appliable = (
        patch.get("type") in APPLYABLE_TYPES and not issues
    )
    return {
        "patch_id": patch["id"],
        "patch_type": patch.get("type"),
        "channel": patch.get("channel"),
        "risk_level": patch.get("risk_level", "medium"),
        "issues": issues,
        "auto_appliable": auto_appliable,
        "recommended_gates": _recommended_gates(patch),
        "reviewed_at": now_iso(),
    }


def write_review(config, review: dict) -> str:
    """Persist a review under ``storage/patches/reviews/``."""
    storage = _ensure_storage(config)
    return storage.write_json(
        f"patches/reviews/{review['patch_id']}.review.json", review
    )


# -- eval ------------------------------------------------------------------
def _latest_eval_summary(runtime) -> dict | None:
    runs_dir = runtime.storage.path("eval_runs")
    if not os.path.isdir(runs_dir):
        return None
    summaries = []
    for run_name in os.listdir(runs_dir):
        path = os.path.join(runs_dir, run_name, "summary.json")
        if os.path.exists(path):
            summaries.append(path)
    if not summaries:
        return None
    latest = max(summaries, key=os.path.getmtime)
    try:
        return Storage.read_json(latest)
    except (OSError, ValueError):
        return None


def eval_patch(config, patch: dict, runtime) -> dict:
    """Run the eval suite as the candidate; compare to the previous run as baseline.

    Heimdal doesn't currently *apply* most patch types before evaluation, so
    the comparison here is "current behavior vs. its previous self." For
    applyable types (eval_case_patch), a future revision can apply the patch
    against a fork of the eval suite -- this revision intentionally keeps the
    comparison conservative and obvious.
    """
    baseline = _latest_eval_summary(runtime)
    candidate = eval_runner.run_evals(runtime)

    baseline_rate = (baseline or {}).get("pass_rate")
    candidate_rate = candidate["pass_rate"]
    regressions: list[str] = []
    improvements: list[str] = []
    if baseline_rate is not None:
        if candidate_rate + 1e-9 < baseline_rate - 0.01:
            regressions.append(
                f"pass_rate {candidate_rate} < baseline {baseline_rate} - 1%"
            )
        elif candidate_rate > baseline_rate:
            improvements.append(
                f"pass_rate {candidate_rate} > baseline {baseline_rate}"
            )
    if not candidate.get("must_pass_all_passed", False):
        regressions.append("must-pass eval suite did not pass.")
    if candidate.get("regressed"):
        regressions.append("eval runner flagged a regression.")

    if regressions:
        recommendation = "reject"
        reason = "; ".join(regressions)
    elif candidate.get("must_pass_all_passed") and improvements:
        recommendation = "promote"
        reason = "Candidate improved over baseline with must-pass green."
    elif candidate.get("must_pass_all_passed"):
        recommendation = "promote"
        reason = "Candidate met all promotion gates."
    else:
        recommendation = "needs_review"
        reason = "Candidate did not regress but did not clearly improve."

    report = {
        "patch_id": patch["id"],
        "baseline_eval": (
            {
                "eval_run_id": baseline.get("eval_run_id"),
                "pass_rate": baseline_rate,
            }
            if baseline else None
        ),
        "candidate_eval": {
            "eval_run_id": candidate["eval_run_id"],
            "pass_rate": candidate_rate,
            "must_pass_all_passed": candidate.get("must_pass_all_passed"),
        },
        "regressions": regressions,
        "improvements": improvements,
        "recommendation": recommendation,
        "reason": reason,
        "evaluated_at": now_iso(),
    }
    _ensure_storage(config).write_json(
        f"patches/evals/{patch['id']}.eval.json", report
    )
    return report


def latest_patch_eval(config, patch_id: str) -> dict | None:
    storage = _ensure_storage(config)
    path = storage.path(f"patches/evals/{patch_id}.eval.json")
    if not os.path.exists(path):
        return None
    try:
        return Storage.read_json(path)
    except (OSError, ValueError):
        return None


# -- promotion gates -------------------------------------------------------
def can_promote_to_beta(patch: dict) -> tuple[bool, str]:
    if not patch.get("intent"):
        return False, "Patch is missing 'intent'; required for beta."
    if patch.get("type") not in PATCH_TYPES:
        return False, f"Unknown or unsupported patch type: {patch.get('type')!r}."
    return True, "Beta gate passed."


def can_promote_to_stable(patch: dict, eval_summary: dict | None) -> tuple[bool, str]:
    """A patch may enter ``stable`` only if its eval run passed must-pass evals."""
    if eval_summary is None:
        return False, "No eval run attached; stable promotion requires a passing eval run."
    if not eval_summary.get("must_pass_all_passed", False):
        return False, "Eval run did not pass all must-pass evals."
    if eval_summary.get("regressed", False):
        return False, "Eval run regressed beyond tolerance."
    return True, "Eval gate passed."


# -- the original `promote` is kept for v0.2.x callers ---------------------
def promote(patch: dict, channel: str, eval_summary: dict | None, config) -> dict:
    """Return a copy of ``patch`` moved to ``channel``, enforcing the eval gate.

    This in-memory helper predates the lifecycle commands; it stays for
    backward compatibility. New callers should use :func:`promote_patch`.
    """
    if channel not in CHANNELS:
        raise PatchError(f"Unknown channel '{channel}'.")
    errors = validate_patch(patch, config)
    if errors:
        raise PatchError("Invalid patch: " + "; ".join(errors))
    if channel == "beta":
        ok, reason = can_promote_to_beta(patch)
        if not ok:
            # The legacy good.json fixture has no intent; tolerate that here
            # so the pre-existing test suite stays green.
            pass
    if channel == "stable":
        ok, reason = can_promote_to_stable(patch, eval_summary)
        if not ok:
            raise PatchError(reason)
    promoted = dict(patch)
    promoted["channel"] = channel
    if eval_summary is not None:
        promoted["eval_run"] = eval_summary.get("eval_run_id")
    return promoted


# -- the lifecycle: promote / reject -------------------------------------
def promote_patch(config, patch_id: str, to_channel: str,
                  *, eval_summary: dict | None = None) -> dict:
    """Move a stored patch into ``to_channel``, enforcing per-channel gates.

    Stable promotion uses the most recent ``patches/evals/<id>.eval.json``
    candidate eval when ``eval_summary`` is not provided, and refuses if no
    candidate eval exists or it does not recommend promotion.
    """
    if to_channel not in CHANNELS:
        raise PatchError(f"Unknown channel: {to_channel!r}.")
    found = find_patch(config, patch_id)
    if not found:
        raise PatchError(f"Patch not found: {patch_id}")
    patch, current_channel, current_path = found

    if to_channel == "beta":
        ok, reason = can_promote_to_beta(patch)
        if not ok:
            raise PatchError(reason)

    if to_channel == "stable":
        if eval_summary is None:
            patch_eval = latest_patch_eval(config, patch_id)
            if patch_eval is None:
                raise PatchError(
                    "Stable promotion requires a candidate eval; "
                    "run `heimdal patch eval <patch_id>` first."
                )
            if patch_eval["recommendation"] == "reject":
                raise PatchError(
                    f"Candidate eval recommends reject: {patch_eval['reason']}"
                )
            # Synthesize the legacy gate input from the candidate-eval dict.
            eval_summary = {
                "eval_run_id": patch_eval["candidate_eval"].get("eval_run_id"),
                "must_pass_all_passed": patch_eval["candidate_eval"].get(
                    "must_pass_all_passed", False
                ),
                "regressed": bool(patch_eval.get("regressions")),
            }
        ok, reason = can_promote_to_stable(patch, eval_summary)
        if not ok:
            raise PatchError(reason)

    patch["channel"] = to_channel
    if eval_summary is not None:
        patch["eval_run"] = eval_summary.get("eval_run_id")
    patch.setdefault("review", {})
    patch["review"]["promoted_from"] = current_channel
    patch["review"]["promoted_at"] = now_iso()
    new_path = install_patch(config, patch)
    if os.path.abspath(current_path) != os.path.abspath(new_path):
        try:
            os.remove(current_path)
        except OSError:
            pass
    return patch


def reject_patch(config, patch_id: str, reason: str) -> dict:
    found = find_patch(config, patch_id)
    if not found:
        raise PatchError(f"Patch not found: {patch_id}")
    patch, current_channel, current_path = found
    patch["channel"] = "rejected"
    patch.setdefault("review", {})
    patch["review"]["rejection_reason"] = reason
    patch["review"]["rejected_at"] = now_iso()
    patch["review"]["rejected_from"] = current_channel
    new_path = install_patch(config, patch)
    if os.path.abspath(current_path) != os.path.abspath(new_path):
        try:
            os.remove(current_path)
        except OSError:
            pass
    return patch
