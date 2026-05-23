"""Skill Library 2.0 -- registry, scoring, stats.

Skills are versioned JSON (or YAML) bundles living under
``storage/skills/<role>/<skill_id>.json``. The registry walks that tree,
exposes per-skill cards to the selector, and persists per-skill usage
stats so Dream Mode and the patch system can reason about which skills are
working.

Context OS calls :meth:`SkillRegistry.select_cards` -- it never inlines all
skills. The selection rules cap per-run skill count by deployment-mode and
prefer role-specific, high-trigger-overlap, well-performing skills.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from heimdal import jsonschema_min
from heimdal.ids import now_iso

# Per-run skill caps by deployment mode (hardware profile).
MAX_SKILLS_BY_DEPLOYMENT = {
    "Dev": 3,
    "Single Device": 5,
    "Pipeline": 5,
    "Factory": 7,
}
DEFAULT_MAX_SKILLS = 5


# Performance "neutral" baseline for skills with no history yet -- mid-range
# so they aren't penalized vs. an unproven one in the first round.
_NEUTRAL_PASS_RATE = 0.5

_STATS_FILE = "_stats.json"


@dataclass
class Skill:
    """One loaded skill, plus the runtime stats Heimdal tracks for it."""

    id: str
    version: str
    role: str
    title: str
    description: str
    triggers: list[str]
    instructions: list[str]
    raw: dict
    source_path: str
    performance: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict, *, source_path: str = "") -> "Skill":
        return cls(
            id=str(data["id"]),
            version=str(data.get("version", "0.0.0")),
            role=str(data.get("role", "general")),
            title=str(data.get("title", data["id"])),
            description=str(data.get("description", "")),
            triggers=list(data.get("triggers", []) or []),
            instructions=list(data.get("instructions", []) or []),
            raw=data,
            source_path=source_path,
            performance=dict(data.get("performance", {}) or {}),
        )

    @property
    def guidance(self) -> str:
        if self.instructions:
            return " ".join(self.instructions)
        return self.description or self.title

    @property
    def pass_rate(self) -> float:
        uses = int(self.performance.get("uses", 0))
        if uses == 0:
            return _NEUTRAL_PASS_RATE
        passes = int(self.performance.get("passes", 0))
        return passes / uses

    def to_dict(self) -> dict:
        return dict(self.raw, performance=self.performance)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def max_skills_for(deployment_mode: str | None) -> int:
    return MAX_SKILLS_BY_DEPLOYMENT.get(deployment_mode or "", DEFAULT_MAX_SKILLS)


class SkillRegistry:
    """Loads skills from ``storage/skills`` and tracks per-skill stats.

    The registry is intentionally cheap: it walks the directory each time
    :meth:`load` is called so newly-installed skills become visible without a
    daemon restart. Stats live in a small ``_stats.json`` keyed by skill_id.
    """

    def __init__(self, skills_root: str):
        self.skills_root = skills_root
        self._stats_cache: dict | None = None

    # -- loading ----------------------------------------------------------
    def _read_skill(self, path: str) -> Skill | None:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict) or "id" not in data:
            return None
        return Skill.from_dict(data, source_path=path)

    def load(self) -> list[Skill]:
        """Walk skills_root and return every parseable skill (with stats)."""
        if not os.path.isdir(self.skills_root):
            return []
        skills: list[Skill] = []
        for root, _dirs, files in os.walk(self.skills_root):
            for name in sorted(files):
                if name.startswith("_") or not name.endswith(".json"):
                    continue
                skill = self._read_skill(os.path.join(root, name))
                if skill is None:
                    continue
                stats = self._stats().get(skill.id)
                if stats:
                    skill.performance = dict(skill.performance, **stats)
                skills.append(skill)
        return skills

    def find(self, skill_id: str) -> Skill | None:
        for skill in self.load():
            if skill.id == skill_id:
                return skill
        return None

    def search(self, query: str) -> list[Skill]:
        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        scored: list[tuple[int, Skill]] = []
        for skill in self.load():
            hay = " ".join(
                [skill.title, skill.description, " ".join(skill.triggers),
                 " ".join(skill.instructions), skill.id]
            )
            overlap = len(query_tokens & _tokens(hay))
            if overlap:
                scored.append((overlap, skill))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [skill for _, skill in scored]

    # -- selection --------------------------------------------------------
    def select(
        self,
        *,
        role_id: str,
        candidate_ids: list[str],
        instruction: str,
        max_skills: int = DEFAULT_MAX_SKILLS,
    ) -> list[Skill]:
        """Pick at most ``max_skills`` relevant skills for one task.

        Role-listed candidates are seeded first; registry skills with high
        trigger overlap on the instruction are added next. The pass-rate of
        each skill's recent history breaks ties. Skills with zero relevance
        are never injected, even if there is budget room.
        """
        instruction_tokens = _tokens(instruction)
        loaded = {s.id: s for s in self.load()}
        scored: list[tuple[float, int, Skill]] = []
        seen: set[str] = set()

        # 1. Role's own candidate list comes first; each gets a small boost
        #    so the role pack stays authoritative for borderline relevance.
        for index, skill_id in enumerate(candidate_ids):
            skill = loaded.get(skill_id)
            if skill is None or skill.id in seen:
                continue
            score = self._score(skill, instruction_tokens, role_id) + 0.25
            scored.append((score, -index, skill))
            seen.add(skill.id)

        # 2. Any other role-matching skill in the registry whose triggers fire.
        #    These aren't on the role's curated candidate list, so they MUST
        #    earn their slot via trigger overlap -- otherwise we'd inject any
        #    same-role skill regardless of relevance.
        for skill in loaded.values():
            if skill.id in seen or skill.role != role_id:
                continue
            trigger_overlap = len(
                instruction_tokens & _tokens(" ".join(skill.triggers))
            )
            if trigger_overlap == 0:
                continue
            score = self._score(skill, instruction_tokens, role_id)
            scored.append((score, 0, skill))
            seen.add(skill.id)

        scored.sort(key=lambda triple: (triple[0], triple[1]), reverse=True)
        return [skill for score, _, skill in scored if score > 0][:max_skills]

    def _score(self, skill: Skill, instruction_tokens: set[str], role_id: str) -> float:
        trigger_overlap = len(instruction_tokens & _tokens(" ".join(skill.triggers)))
        role_match = 1.0 if skill.role == role_id else 0.0
        return trigger_overlap + role_match + (skill.pass_rate - _NEUTRAL_PASS_RATE)

    # -- stats ------------------------------------------------------------
    def _stats_path(self) -> str:
        return os.path.join(self.skills_root, _STATS_FILE)

    def _stats(self) -> dict:
        if self._stats_cache is not None:
            return self._stats_cache
        path = self._stats_path()
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    self._stats_cache = json.load(fh)
                    return self._stats_cache
            except (OSError, ValueError):
                pass
        self._stats_cache = {}
        return self._stats_cache

    def _write_stats(self) -> None:
        os.makedirs(self.skills_root, exist_ok=True)
        with open(self._stats_path(), "w", encoding="utf-8") as fh:
            json.dump(self._stats_cache or {}, fh, indent=2, sort_keys=True)

    def record_usage(self, skill_ids: list[str], *, passed: bool) -> None:
        """Bump per-skill counters after a runtime task completes."""
        if not skill_ids:
            return
        stats = self._stats()
        for skill_id in skill_ids:
            entry = stats.setdefault(
                skill_id,
                {"uses": 0, "passes": 0, "fails": 0, "last_used": None},
            )
            entry["uses"] += 1
            if passed:
                entry["passes"] += 1
            else:
                entry["fails"] += 1
            entry["last_used"] = now_iso()
        self._write_stats()


def validate_skill(skill: dict, config) -> list[str]:
    """Return a list of schema errors for one skill dict (empty = valid)."""
    schema = jsonschema_min.load_schema(config.schema_path("skill.schema.json"))
    return jsonschema_min.validate(skill, schema)


def install_skill_file(config, source_path: str) -> str:
    """Copy a skill JSON file into ``storage/skills/<role>/`` under its id."""
    with open(source_path, "r", encoding="utf-8") as fh:
        skill = json.load(fh)
    errors = validate_skill(skill, config)
    if errors:
        raise ValueError("Invalid skill: " + "; ".join(errors))
    role = str(skill.get("role", "general"))
    skill_id = str(skill["id"])
    dest_dir = os.path.join(config.storage_root, "skills", role)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, f"{skill_id}.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(skill, fh, indent=2, sort_keys=True)
    return dest


def archive_skill(config, skill_id: str) -> str | None:
    """Move a skill JSON file from its role dir into ``skills/archived/``."""
    registry = SkillRegistry(os.path.join(config.storage_root, "skills"))
    skill = registry.find(skill_id)
    if skill is None or not skill.source_path:
        return None
    archived_dir = os.path.join(config.storage_root, "skills", "archived")
    os.makedirs(archived_dir, exist_ok=True)
    dest = os.path.join(archived_dir, os.path.basename(skill.source_path))
    os.replace(skill.source_path, dest)
    return dest
