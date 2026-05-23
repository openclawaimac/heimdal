"""Skill Selector -- Context OS's gate against skill dump.

In v0.4.2 the heavy lifting moves to :class:`heimdal.skills.registry.SkillRegistry`;
this module stays as the public selection surface Context OS calls. When a
candidate skill id matches a Skill Library 2.0 entry on disk, that entry is
used; otherwise the small built-in library below is consulted so v0.2.x role
packs that name ``concise_writing`` / ``structured_answer`` still resolve.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from heimdal.skills.registry import (
    DEFAULT_MAX_SKILLS,
    SkillRegistry,
    max_skills_for,
)

# Legacy built-in skill library kept for backward compatibility with v0.2.x
# role packs that name these IDs. New work should add JSON skills under
# storage/skills/<role>/.
SKILL_LIBRARY: dict[str, dict] = {
    "concise_writing": {
        "guidance": "Be concise. Lead with the answer. Cut filler.",
        "keywords": ["short", "brief", "summary", "explain"],
    },
    "structured_answer": {
        "guidance": "Use clear structure: a direct answer, then supporting points.",
        "keywords": ["explain", "describe", "compare", "list"],
    },
    "source_grounding": {
        "guidance": "Cite a retrieved source for every factual claim. If a source "
        "is missing, say so instead of guessing.",
        "keywords": ["policy", "pricing", "fact", "number", "source", "exact"],
    },
    "citation_check": {
        "guidance": "Verify each citation points to real retrieved content. Never "
        "fabricate a reference.",
        "keywords": ["cite", "citation", "reference", "source"],
    },
    "code_generation": {
        "guidance": "Produce correct, minimal code. State assumptions explicitly.",
        "keywords": ["code", "function", "implement", "script", "bug"],
    },
    "risk_callout": {
        "guidance": "Call out risky or irreversible steps before recommending them.",
        "keywords": ["deploy", "delete", "production", "risk", "migration"],
    },
}

# Kept for v0.2.x callers that imported this constant directly.
MAX_SKILLS = DEFAULT_MAX_SKILLS


@dataclass
class SkillCard:
    """The trimmed view of a skill Context OS embeds in the Context Packet."""

    skill_id: str
    guidance: str
    source: str  # "registry" or "builtin"


def _keywords(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


class SkillSelector:
    """Pick at most ``max_skills`` relevant skills for one task.

    ``skills_dir`` points at ``storage/skills``; when it exists, the Skill
    Library 2.0 registry handles ranking. Role-listed candidate ids that
    don't match any registry skill fall through to the built-in library so
    v0.2.x role packs continue to work.
    """

    def __init__(
        self,
        skills_dir: str | None = None,
        *,
        max_skills: int = MAX_SKILLS,
        role_id: str = "general",
    ):
        self.skills_dir = skills_dir
        self.role_id = role_id
        self.max_skills = max_skills
        self._registry = (
            SkillRegistry(skills_dir) if skills_dir and os.path.isdir(skills_dir) else None
        )

    def select(self, candidate_ids: list[str], instruction: str) -> list[SkillCard]:
        """Score candidate ids + role-matched registry skills against the task."""
        instruction_kw = _keywords(instruction)
        cards: list[SkillCard] = []
        seen: set[str] = set()

        # 1. Registry-first selection if a Skill Library 2.0 tree exists.
        if self._registry is not None:
            registry_skills = self._registry.select(
                role_id=self.role_id,
                candidate_ids=candidate_ids,
                instruction=instruction,
                max_skills=self.max_skills,
            )
            for skill in registry_skills:
                cards.append(SkillCard(skill.id, skill.guidance, "registry"))
                seen.add(skill.id)

        # 2. Fall back to the legacy built-in library for any role candidate
        #    id that didn't resolve to a registry skill. Same scoring rule as
        #    before: any positive overlap on the keyword hint qualifies.
        if len(cards) < self.max_skills:
            for index, skill_id in enumerate(candidate_ids):
                if skill_id in seen:
                    continue
                spec = SKILL_LIBRARY.get(skill_id)
                if spec is None:
                    continue
                overlap = len(instruction_kw & set(spec.get("keywords", [])))
                relevance = overlap + (1.0 if index == 0 else 0.0)
                if relevance <= 0:
                    continue
                cards.append(SkillCard(skill_id, spec["guidance"], "builtin"))
                seen.add(skill_id)
                if len(cards) >= self.max_skills:
                    break

        return cards[: self.max_skills]


def selector_for(role: dict, skills_dir: str, hardware_profile: dict | None = None):
    """Helper: build a SkillSelector calibrated to the role + deployment mode."""
    deployment = (hardware_profile or {}).get("deployment_mode")
    return SkillSelector(
        skills_dir,
        max_skills=max_skills_for(deployment),
        role_id=role.get("role_id", "general"),
    )
