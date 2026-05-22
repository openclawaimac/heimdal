"""Model Router.

Given a Task Contract, the router selects the budget behaviour, the worker
model, retrieval requirement, verifier strictness/backend, and repair budget
(docs/builder_pack/04_runtime/QUALITY_FACTORY.md).
"""

from __future__ import annotations

from heimdal.core.constants import HYBRID, LENIENT, RULE_BASED, STANDARD, STRICT
from heimdal.models.base import select_generative_model
from heimdal.models.offline import OFFLINE_MODEL, OfflineBackend

# Behaviour per budget level (docs/builder_pack/01_architecture/WORK_DREAM_MIRROR_MODES.md).
BUDGET_BEHAVIOUR = {
    "B0": {"verifier_strictness": LENIENT, "max_repair": 0, "samples": 1, "retrieval": False},
    "B1": {"verifier_strictness": STANDARD, "max_repair": 1, "samples": 1, "retrieval": False},
    "B2": {"verifier_strictness": STANDARD, "max_repair": 2, "samples": 1, "retrieval": True},
    "B3": {"verifier_strictness": STRICT, "max_repair": 2, "samples": 2, "retrieval": True},
    "B4": {"verifier_strictness": STRICT, "max_repair": 3, "samples": 3, "retrieval": True},
}

_BUDGET_RANK = {level: index for index, level in enumerate(BUDGET_BEHAVIOUR)}


class ModelUnavailableError(RuntimeError):
    """Raised when no usable model can be resolved for a profile."""


def _budget_at_least(level: str, minimum: str) -> bool:
    return _BUDGET_RANK.get(level, 1) >= _BUDGET_RANK.get(minimum, 2)


def _resolve_model(profile_name: str, installed: set[str] | None, config) -> str:
    """Resolve a concrete model for a profile.

    Prefers an installed manifest candidate; otherwise falls back to any
    installed generative model; raises if Ollama has no usable model.
    """
    if installed is None:  # offline backend
        return OFFLINE_MODEL
    candidates = config.model_profiles.get(profile_name, {}).get("candidates", []) or []
    for candidate in candidates:
        if candidate in installed:
            return candidate
    fallback = select_generative_model(config, sorted(installed))
    if fallback:
        return fallback
    raise ModelUnavailableError(
        f"No model for profile '{profile_name}' is installed in Ollama. "
        f"Candidates: {candidates or '(none configured)'}. "
        f"Installed: {sorted(installed) or '(none)'}. "
        f"Run: ollama pull {candidates[0] if candidates else 'qwen2.5:7b'}"
    )


def route(
    contract: dict,
    role: dict,
    backend,
    config,
    model_override: str | None = None,
    verifier_override: str | None = None,
) -> dict:
    """Return a routing decision for a contract.

    ``verifier_override`` ('rule_based' | 'hybrid') overrides the manifest
    verifier mode for this run.
    """
    quality_level = contract.get("budget", {}).get("quality_level", "B1")
    behaviour = dict(BUDGET_BEHAVIOUR.get(quality_level, BUDGET_BEHAVIOUR["B1"]))

    verification = contract.get("verification", {})
    if verification.get("requires_sources"):
        behaviour["retrieval"] = True

    worker_profile = "coder" if role.get("role_id") == "dev" else "worker"
    brain_profile = "brain" if quality_level in ("B3", "B4") else None

    max_repair = min(
        behaviour["max_repair"],
        max(0, contract.get("budget", {}).get("max_iterations", 3) - 1),
    )

    is_offline = backend.name == OfflineBackend.name
    installed = None if is_offline else set(backend.list_models())

    if model_override:
        worker_model = model_override
    else:
        worker_model = _resolve_model(worker_profile, installed, config)

    # Verification is rule-based by default. Hybrid adds a model-based semantic
    # verifier for B2-B4 tasks; the offline backend mocks it deterministically.
    verifier_cfg = config.verifier
    mode = verifier_override or verifier_cfg.get("mode", RULE_BASED)
    hybrid_requested = mode == HYBRID or verifier_cfg.get(
        "semantic_model_verifier_enabled", False
    )
    hybrid = hybrid_requested and _budget_at_least(quality_level, "B2")
    verifier_backend = HYBRID if hybrid else RULE_BASED

    if hybrid:
        semantic_verifier_model = (
            verifier_cfg.get("semantic_verifier_model")
            or _resolve_model("verifier", installed, config)
        )
    else:
        semantic_verifier_model = None

    return {
        "quality_level": quality_level,
        "worker_profile": worker_profile,
        "worker_model": worker_model,
        "brain_profile": brain_profile,
        "verifier_backend": verifier_backend,
        "semantic_verifier_model": semantic_verifier_model,
        "verifier_strictness": behaviour["verifier_strictness"],
        "retrieval_required": behaviour["retrieval"],
        "samples": behaviour["samples"],
        "max_repair_iterations": max_repair,
        "backend": backend.name,
    }
