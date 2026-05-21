"""Model Router.

Given a Task Contract, the router selects the budget behaviour, the model
profile, retrieval requirement, verifier strictness, and repair budget
(docs/builder_pack/04_runtime/QUALITY_FACTORY.md).
"""

from __future__ import annotations

# Behaviour per budget level (docs/builder_pack/01_architecture/WORK_DREAM_MIRROR_MODES.md).
BUDGET_BEHAVIOUR = {
    "B0": {"verifier_strictness": "lenient", "max_repair": 0, "samples": 1, "retrieval": False},
    "B1": {"verifier_strictness": "standard", "max_repair": 1, "samples": 1, "retrieval": False},
    "B2": {"verifier_strictness": "standard", "max_repair": 2, "samples": 1, "retrieval": True},
    "B3": {"verifier_strictness": "strict", "max_repair": 2, "samples": 2, "retrieval": True},
    "B4": {"verifier_strictness": "strict", "max_repair": 3, "samples": 3, "retrieval": True},
}


def _resolve_model(profile_name: str, backend, config) -> str:
    profile = config.model_profiles.get(profile_name, {})
    candidates = profile.get("candidates", []) or []
    if backend.name == "offline":
        return "heimdal-offline-stub"
    installed = set(backend.list_models())
    for candidate in candidates:
        if candidate in installed:
            return candidate
    return candidates[0] if candidates else "default"


def route(contract: dict, role: dict, backend, config) -> dict:
    """Return a routing decision for a contract."""
    quality_level = contract.get("budget", {}).get("quality_level", "B1")
    behaviour = dict(BUDGET_BEHAVIOUR.get(quality_level, BUDGET_BEHAVIOUR["B1"]))

    verification = contract.get("verification", {})
    if verification.get("requires_sources"):
        behaviour["retrieval"] = True

    worker_profile = "coder" if role.get("role_id") == "dev" else "worker"
    if quality_level in ("B3", "B4"):
        brain_profile = "brain"
    else:
        brain_profile = None

    max_repair = min(
        behaviour["max_repair"],
        max(0, contract.get("budget", {}).get("max_iterations", 3) - 1),
    )

    return {
        "quality_level": quality_level,
        "worker_profile": worker_profile,
        "worker_model": _resolve_model(worker_profile, backend, config),
        "verifier_profile": "verifier",
        "verifier_model": _resolve_model("verifier", backend, config),
        "brain_profile": brain_profile,
        "verifier_strictness": behaviour["verifier_strictness"],
        "retrieval_required": behaviour["retrieval"],
        "samples": behaviour["samples"],
        "max_repair_iterations": max_repair,
        "backend": backend.name,
    }
