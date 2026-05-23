"""Task Contract Builder.

Beta rule: every Work Mode task gets a Task Contract before any model call
(docs/builder_pack/02_contracts/TASK_CONTRACT_SPEC.md).
"""

from __future__ import annotations

from heimdal import jsonschema_min
from heimdal.ids import new_id

QUALITY_LEVELS = ["B0", "B1", "B2", "B3", "B4"]


class ContractError(ValueError):
    """Raised when a built Task Contract fails schema validation."""


def _requires_sources(task_request: dict, role: dict) -> bool:
    constraints = task_request.get("constraints", {}) or {}
    if constraints.get("requires_sources"):
        return True
    return role.get("role_id") in ("research", "finance")


def requires_grounding(verification: dict) -> bool:
    """Whether a contract's verification block demands source-grounded output."""
    return bool(
        verification.get("requires_sources") or verification.get("requires_citations")
    )


def _requires_schema(task_request: dict) -> bool:
    outputs = task_request.get("expected_outputs", []) or []
    return any("json" in str(o).lower() or "schema" in str(o).lower() for o in outputs)


def build_contract(envelope: dict, role: dict, config,
                   *, profile_overrides: dict | None = None) -> dict:
    """Build and schema-validate a Task Contract from a Host Task Envelope.

    ``profile_overrides`` (v0.6.2) supplies hardware-adaptive defaults
    from the active runtime profile. Per-task ``budget`` constraints in
    the envelope still win; the profile only fills the gaps left by the
    manifest's static defaults.
    """
    task_request = envelope.get("task_request", {}) or {}
    constraints = dict(task_request.get("constraints", {}) or {})
    budgets = config.budgets
    profile_overrides = profile_overrides or {}

    requested = task_request.get("budget", {}) or {}
    quality_level = (
        requested.get("quality_level")
        or profile_overrides.get("default_quality_level")
        or budgets.get("default_quality_level", "B1")
    )
    if quality_level not in QUALITY_LEVELS:
        quality_level = "B1"

    requires_sources = _requires_sources(task_request, role)
    expected_outputs = task_request.get("expected_outputs", []) or []

    definition_of_done = [
        f"Address the objective: {task_request.get('instruction', '').strip()}",
        "Pass the Heimdal verifier (PASS) with no high/critical defects",
    ]
    if expected_outputs:
        definition_of_done.append(
            "Produce expected outputs: " + ", ".join(str(o) for o in expected_outputs)
        )
    if requires_sources:
        definition_of_done.append("Ground every factual claim in a retrieved source")

    contract = {
        "contract_id": new_id("contract"),
        "task_id": task_request.get("task_id", new_id("task")),
        "role_id": role.get("role_id", "general"),
        "objective": task_request.get("instruction", "").strip(),
        "definition_of_done": definition_of_done,
        "expected_outputs": expected_outputs,
        "constraints": constraints,
        "required_sources": list(task_request.get("inputs", {}).get("sources", []) or []),
        "tool_requirements": list(role.get("allowed_tools", [])),
        "risk_profile": {
            "risk_mode": role.get("risk_mode", "balanced"),
            "priority": task_request.get("priority", "P2"),
        },
        "budget": {
            "quality_level": quality_level,
            "max_iterations": int(
                requested.get(
                    "max_iterations",
                    profile_overrides.get("max_repair_iterations")
                    or budgets.get("max_repair_iterations", 2),
                )
                + 1
            ),
            "max_input_tokens": int(
                requested.get(
                    "max_input_tokens",
                    profile_overrides.get("max_context_tokens")
                    or budgets.get("max_input_tokens", 8000),
                )
            ),
            "max_output_tokens": int(
                requested.get("max_output_tokens", budgets.get("max_output_tokens", 2000))
            ),
        },
        "verification": {
            "rubric_id": role.get("rubric_id", "general_v1"),
            "requires_citations": requires_sources,
            "requires_schema_validation": _requires_schema(task_request),
            "no_guess_gate": True,
            "requires_sources": requires_sources,
        },
    }

    jsonschema_min.validate_or_raise(
        contract,
        config.schema_path("task_contract.schema.json"),
        "Task Contract",
        ContractError,
    )
    return contract
