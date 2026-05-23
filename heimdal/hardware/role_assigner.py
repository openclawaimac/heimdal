"""Model role assignment.

Maps installed Ollama models to internal roles (verifier / worker / brain /
coder / semantic_verifier / ...) using the capability matrix as the truth
source. Honours operator pins (``storage/runtime/model_pins.json``), falls
back gracefully when no installed model passes the minimum capability bar
for a role, and never assigns an embedding model to a generation role.

The assignment artifact (``storage/runtime/role_assignments.json``) is what
the Runtime reads at startup so ``heimdal run`` can launch without an
explicit ``--model`` flag.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from heimdal.ids import now_iso

# Roles we recognise. ``brain`` and ``coder`` are optional -- skipping
# them is fine on weak machines.
ROLES = (
    "worker",
    "verifier",
    "semantic_verifier",
    "brain",
    "coder",
)
OPTIONAL_ROLES = {"brain", "coder"}

# Sources we tag assignments with for trace/repro auditing.
SOURCE_EXPLICIT_CLI = "explicit_cli"
SOURCE_MANIFEST_PIN = "manifest_pin"
SOURCE_OPERATOR_PIN = "operator_pin"
SOURCE_AUTO_TUNER = "auto_tuner"
SOURCE_FALLBACK = "fallback"


@dataclass
class _Assignment:
    backend: str
    model: str | None
    reason: str
    source: str
    fallback: bool = False


def _model_meets(caps: dict, required: list[str]) -> bool:
    """True if ``caps`` reports ``pass`` for every capability in ``required``."""
    if not required:
        return True
    for cap in required:
        if caps.get(cap) != "pass":
            return False
    return True


def _pick_for_role(role: str, role_cfg: dict, installed: list[str],
                   model_caps: dict) -> _Assignment:
    preferred = list(role_cfg.get("preferred", []) or [])
    required = list(role_cfg.get("min_capabilities", []) or [])

    # 1. First installed preferred model that meets min_capabilities.
    for candidate in preferred:
        if candidate not in installed:
            continue
        caps = model_caps.get(candidate, {})
        if caps.get("skipped"):  # embedding model
            continue
        if _model_meets(caps, required):
            return _Assignment(
                backend="ollama",
                model=candidate,
                reason=(
                    f"preferred installed model passed "
                    f"{', '.join(required) or 'min_capabilities'}"
                ),
                source=SOURCE_AUTO_TUNER,
            )

    # 2. Any installed model whose matrix entry marks it candidate for
    #    this role's job class.
    candidate_key = {
        "worker": "worker_candidate",
        "verifier": "semantic_verifier_candidate",
        "semantic_verifier": "semantic_verifier_candidate",
        "brain": "worker_candidate",
        "coder": "worker_candidate",
    }.get(role)
    if candidate_key:
        for model in installed:
            caps = model_caps.get(model, {})
            if caps.get("skipped") or not caps.get(candidate_key):
                continue
            if _model_meets(caps, required):
                return _Assignment(
                    backend="ollama",
                    model=model,
                    reason=(
                        f"no preferred candidate installed; falling back to "
                        f"{model} which passes {candidate_key}"
                    ),
                    source=SOURCE_FALLBACK,
                    fallback=True,
                )

    # 3. Role-specific safe defaults.
    if role == "verifier":
        return _Assignment(
            backend="rule_based", model=None,
            reason="no model passes verifier requirements; rule-based verifier in use",
            source=SOURCE_FALLBACK, fallback=True,
        )
    if role in OPTIONAL_ROLES:
        return _Assignment(
            backend="disabled", model=None,
            reason=f"optional role {role!r} left unassigned (no suitable model)",
            source=SOURCE_FALLBACK, fallback=True,
        )
    return _Assignment(
        backend="offline", model=None,
        reason=(
            "no suitable Ollama model installed; using deterministic offline "
            "backend"
        ),
        source=SOURCE_FALLBACK, fallback=True,
    )


def load_pins(storage) -> dict:
    """Read ``storage/runtime/model_pins.json``; missing file -> empty."""
    path = storage.path("runtime", "model_pins.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_pin(storage, role: str, model: str | None) -> dict:
    pins = load_pins(storage)
    if model is None:
        pins.pop(role, None)
    else:
        pins[role] = {"model": model, "pinned_at": now_iso()}
    storage.write_json("runtime/model_pins.json", pins)
    return pins


def _apply_pins(assignments: dict, pins: dict, matrix: dict) -> None:
    installed = (matrix.get("ollama") or {}).get("models", []) or []
    model_caps = matrix.get("model_capabilities", {}) or {}
    for role, pin in pins.items():
        if role not in ROLES or not isinstance(pin, dict):
            continue
        model = pin.get("model")
        if not model:
            continue
        if model not in installed:
            assignments[role]["warnings"] = list(
                assignments[role].get("warnings", [])
            ) + [
                f"pinned model {model!r} for {role!r} is not installed; "
                "auto-assignment used instead"
            ]
            continue
        caps = model_caps.get(model, {})
        if caps.get("skipped"):
            assignments[role]["warnings"] = list(
                assignments[role].get("warnings", [])
            ) + [
                f"pinned model {model!r} for {role!r} is an embedding model"
            ]
            continue
        assignments[role] = {
            "backend": "ollama",
            "model": model,
            "reason": f"operator-pinned via heimdal models pin --role {role}",
            "source": SOURCE_OPERATOR_PIN,
            "fallback": False,
        }


def assign(matrix: dict, *, model_roles_cfg: dict,
           pins: dict | None = None) -> dict:
    """Compute the role -> {backend, model, reason, source, fallback} map."""
    installed = (matrix.get("ollama") or {}).get("models", []) or []
    model_caps = matrix.get("model_capabilities", {}) or {}
    assignments: dict[str, dict] = {}
    warnings: list[str] = []

    for role in ROLES:
        cfg = model_roles_cfg.get(role, {})
        picked = _pick_for_role(role, cfg, installed, model_caps)
        assignments[role] = {
            "backend": picked.backend,
            "model": picked.model,
            "reason": picked.reason,
            "source": picked.source,
            "fallback": picked.fallback,
        }

    if pins:
        _apply_pins(assignments, pins, matrix)

    # Top-level warnings: surface any role-level warnings + matrix warnings.
    for role, entry in assignments.items():
        for warning in entry.get("warnings", []) or []:
            warnings.append(f"{role}: {warning}")
    warnings.extend(matrix.get("warnings", []) or [])

    return {
        "created_at": now_iso(),
        "profile": matrix.get("recommended_runtime_profile", "dev"),
        "assignments": assignments,
        "warnings": warnings,
    }


def write_assignments(storage, assignments: dict) -> str:
    return storage.write_json("runtime/role_assignments.json", assignments)


def load_assignments(storage) -> dict | None:
    path = storage.path("runtime", "role_assignments.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def assigned_worker_model(storage) -> tuple[str | None, str | None]:
    """Convenience for Runtime: (model, source_tag) from worker assignment.

    Returns ``(None, None)`` when no assignment artifact exists yet or the
    worker is on a non-ollama backend (rule_based / disabled / offline).
    """
    data = load_assignments(storage)
    if not data:
        return None, None
    worker = (data.get("assignments") or {}).get("worker") or {}
    if worker.get("backend") != "ollama" or not worker.get("model"):
        return None, None
    return worker["model"], worker.get("source")
