"""Skill Selector.

The Context Builder must inject only relevant skills, never all of them
(docs/builder_pack/09_storage_context/ROLE_PACKS_AND_SKILLS.md). Skills come
from a built-in library plus any user skills dropped into storage/skills.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

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

MAX_SKILLS = 3


@dataclass
class SkillCard:
    skill_id: str
    guidance: str
    source: str  # "builtin" or "storage"


def _keywords(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class SkillSelector:
    def __init__(self, skills_dir: str | None = None):
        self.skills_dir = skills_dir

    def _storage_skills(self) -> dict[str, dict]:
        skills: dict[str, dict] = {}
        if not self.skills_dir or not os.path.isdir(self.skills_dir):
            return skills
        for name in sorted(os.listdir(self.skills_dir)):
            if not name.lower().endswith((".md", ".txt")):
                continue
            path = os.path.join(self.skills_dir, name)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    content = fh.read().strip()
            except OSError:
                continue
            skill_id = os.path.splitext(name)[0]
            skills[skill_id] = {"guidance": content, "keywords": list(_keywords(content))}
        return skills

    def select(self, candidate_ids: list[str], instruction: str) -> list[SkillCard]:
        """Pick at most MAX_SKILLS skills relevant to the instruction."""
        storage_skills = self._storage_skills()
        instruction_kw = _keywords(instruction)

        scored: list[tuple[float, SkillCard]] = []
        for index, skill_id in enumerate(candidate_ids):
            if skill_id in storage_skills:
                spec, source = storage_skills[skill_id], "storage"
            elif skill_id in SKILL_LIBRARY:
                spec, source = SKILL_LIBRARY[skill_id], "builtin"
            else:
                continue
            overlap = len(instruction_kw & set(spec.get("keywords", [])))
            # The first role skill is core and always retained; others ranked.
            relevance = overlap + (1.0 if index == 0 else 0.0)
            scored.append(
                (relevance, SkillCard(skill_id, spec["guidance"], source))
            )

        scored.sort(key=lambda item: item[0], reverse=True)
        return [card for score, card in scored if score > 0][:MAX_SKILLS]
