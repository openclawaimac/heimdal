"""Role Binding Resolver.

A host assigns Heimdal a role binding; Heimdal resolves it into a concrete Role
Pack (system context, allowed tools, skills, rubric, output profiles, risk
defaults) per docs/builder_pack/09_storage_context/ROLE_PACKS_AND_SKILLS.md.
"""

from __future__ import annotations

ROLE_PACKS: dict[str, dict] = {
    "general": {
        "system_context": "You are Heimdal, a careful general-purpose assistant. "
        "Be accurate, concise, and explicit about uncertainty.",
        "allowed_tools": ["retrieval"],
        "skills": ["concise_writing", "structured_answer"],
        "rubric_id": "general_v1",
        "output_profiles": ["markdown", "json"],
        "risk_default": "balanced",
    },
    "research": {
        "system_context": "You are Heimdal in a research role. Ground every factual "
        "claim in a source. Never invent numbers, policies, or citations.",
        "allowed_tools": ["retrieval"],
        "skills": ["source_grounding", "citation_check", "structured_answer"],
        "rubric_id": "research_v1",
        "output_profiles": ["markdown", "json"],
        "risk_default": "conservative",
    },
    "dev": {
        "system_context": "You are Heimdal in a developer role. Produce correct, "
        "minimal code and explain assumptions.",
        "allowed_tools": ["retrieval", "code"],
        "skills": ["code_generation", "structured_answer"],
        "rubric_id": "dev_v1",
        "output_profiles": ["markdown", "code", "json"],
        "risk_default": "balanced",
    },
    "ops": {
        "system_context": "You are Heimdal in an operations role. Prefer safe, "
        "reversible steps and call out risk.",
        "allowed_tools": ["retrieval"],
        "skills": ["structured_answer", "risk_callout"],
        "rubric_id": "ops_v1",
        "output_profiles": ["markdown", "json"],
        "risk_default": "conservative",
    },
    "finance": {
        "system_context": "You are Heimdal in a finance role. Every number must be "
        "sourced. Never estimate figures without saying so.",
        "allowed_tools": ["retrieval"],
        "skills": ["source_grounding", "citation_check"],
        "rubric_id": "finance_v1",
        "output_profiles": ["markdown", "json"],
        "risk_default": "conservative",
    },
}

DEFAULT_ROLE = "general"


def resolve_role(role_binding: dict) -> dict:
    """Merge a host role binding with its built-in Role Pack."""
    role_binding = role_binding or {}
    role_id = role_binding.get("role_id") or DEFAULT_ROLE
    pack = dict(ROLE_PACKS.get(role_id, ROLE_PACKS[DEFAULT_ROLE]))

    risk_mode = role_binding.get("risk_mode") or pack["risk_default"]
    output_profiles = role_binding.get("output_profiles") or pack["output_profiles"]

    return {
        "role_id": role_id,
        "role_pack": role_binding.get("role_pack"),
        "system_context": pack["system_context"],
        "allowed_tools": pack["allowed_tools"],
        "skills": list(pack["skills"]),
        "rubric_id": pack["rubric_id"],
        "output_profiles": list(output_profiles),
        "risk_mode": risk_mode,
        "privacy_mode": role_binding.get("privacy_mode", "local_only"),
        "tool_policy": role_binding.get("tool_policy", {}),
        "memory_scope": role_binding.get("memory_scope", {}),
    }
