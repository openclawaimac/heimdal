"""v0.4.2: Skill Library 2.0 -- registry, selection, stats, CLI."""

import contextlib
import io
import json
import os
import tempfile
import unittest

from tests.helpers import repo_path, temp_config, write_temp_manifest

from heimdal import jsonschema_min
from heimdal.cli import main
from heimdal.core.runtime import Runtime
from heimdal.skills import registry as skill_registry
from heimdal.skills.registry import Skill, SkillRegistry
from heimdal.skills.selector import SkillSelector
from heimdal.storage import Storage


def _skill(skill_id: str, role: str = "general", *, triggers=None,
           instructions=None) -> dict:
    return {
        "id": skill_id,
        "version": "0.1.0",
        "role": role,
        "title": skill_id.replace("_", " ").title(),
        "description": f"Test skill {skill_id}.",
        "triggers": list(triggers or ["test", "skill"]),
        "instructions": list(instructions or [f"Apply {skill_id} guidance."]),
        "performance": {"uses": 0, "passes": 0, "fails": 0, "last_used": None},
    }


def _install(storage: Storage, skill: dict) -> str:
    role_dir = storage.path("skills", skill["role"])
    os.makedirs(role_dir, exist_ok=True)
    path = os.path.join(role_dir, f"{skill['id']}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(skill, fh)
    return path


class SkillRegistryTests(unittest.TestCase):
    def setUp(self):
        self.config = temp_config(tempfile.mkdtemp())
        self.storage = Storage(self.config.storage_root).ensure()
        self.registry = SkillRegistry(self.storage.path("skills"))

    def test_load_walks_role_subdirectories(self):
        _install(self.storage, _skill("alpha", "general"))
        _install(self.storage, _skill("beta", "research"))
        loaded = {s.id for s in self.registry.load()}
        self.assertEqual(loaded, {"alpha", "beta"})

    def test_find_returns_matching_skill(self):
        _install(self.storage, _skill("alpha", "general"))
        skill = self.registry.find("alpha")
        self.assertIsNotNone(skill)
        self.assertEqual(skill.role, "general")
        self.assertIsNone(self.registry.find("missing"))

    def test_search_ranks_by_token_overlap(self):
        _install(self.storage, _skill(
            "summary_skill", "research", triggers=["summary", "source"],
            instructions=["Summarize from sources only."],
        ))
        _install(self.storage, _skill(
            "unrelated", "general", triggers=["coffee"],
            instructions=["Brew coffee carefully."],
        ))
        hits = self.registry.search("summary source")
        self.assertTrue(hits)
        self.assertEqual(hits[0].id, "summary_skill")

    def test_record_usage_tracks_per_skill_stats(self):
        _install(self.storage, _skill("alpha"))
        self.registry.record_usage(["alpha"], passed=True)
        self.registry.record_usage(["alpha"], passed=False)
        # Reload via a new registry instance to confirm stats are durable.
        reloaded = SkillRegistry(self.storage.path("skills"))
        skill = reloaded.find("alpha")
        self.assertEqual(skill.performance["uses"], 2)
        self.assertEqual(skill.performance["passes"], 1)
        self.assertEqual(skill.performance["fails"], 1)
        self.assertIsNotNone(skill.performance["last_used"])

    def test_select_prefers_role_match_and_trigger_overlap(self):
        _install(self.storage, _skill(
            "research_summary", "research", triggers=["summary", "source"],
        ))
        _install(self.storage, _skill(
            "general_intro", "general", triggers=["intro"],
        ))
        picked = self.registry.select(
            role_id="research",
            candidate_ids=["research_summary"],
            instruction="Summarize the source policy.",
            max_skills=3,
        )
        self.assertTrue(picked)
        self.assertEqual(picked[0].id, "research_summary")

    def test_select_skips_irrelevant_skills_even_when_room_remains(self):
        _install(self.storage, _skill(
            "match", "general", triggers=["queue", "stack"]
        ))
        _install(self.storage, _skill(
            "miss", "general", triggers=["coffee", "tea"]
        ))
        picked = self.registry.select(
            role_id="general", candidate_ids=[],
            instruction="Explain what a queue is.",
            max_skills=5,
        )
        ids = [s.id for s in picked]
        self.assertIn("match", ids)
        self.assertNotIn("miss", ids)


class SkillValidationTests(unittest.TestCase):
    def setUp(self):
        self.config = temp_config(tempfile.mkdtemp())

    def test_valid_skill_passes(self):
        self.assertEqual(
            skill_registry.validate_skill(_skill("alpha"), self.config), []
        )

    def test_invalid_skill_rejected(self):
        bad = _skill("alpha")
        del bad["triggers"]
        errors = skill_registry.validate_skill(bad, self.config)
        self.assertTrue(errors)
        self.assertTrue(any("triggers" in e for e in errors))


class SkillInstallArchiveTests(unittest.TestCase):
    def setUp(self):
        self.config = temp_config(tempfile.mkdtemp())
        self.storage = Storage(self.config.storage_root).ensure()

    def test_install_skill_file_writes_into_role_dir(self):
        src = os.path.join(tempfile.mkdtemp(), "src.json")
        with open(src, "w", encoding="utf-8") as fh:
            json.dump(_skill("ops_skill", "ops"), fh)
        dest = skill_registry.install_skill_file(self.config, src)
        self.assertTrue(dest.endswith("skills/ops/ops_skill.json"))
        self.assertTrue(os.path.exists(dest))

    def test_archive_skill_moves_into_archived_dir(self):
        _install(self.storage, _skill("to_archive", "research"))
        dest = skill_registry.archive_skill(self.config, "to_archive")
        self.assertIsNotNone(dest)
        self.assertTrue(dest.endswith("skills/archived/to_archive.json"))
        self.assertFalse(os.path.exists(
            self.storage.path("skills", "research", "to_archive.json")
        ))


class SeedSkillsTests(unittest.TestCase):
    """The seven seed skills must round-trip the schema and seed into storage."""

    def test_seed_skills_validate_against_schema(self):
        config = temp_config(tempfile.mkdtemp())
        schema = jsonschema_min.load_schema(config.schema_path("skill.schema.json"))
        seed_root = repo_path("examples/skills")
        seen: list[str] = []
        for root, _dirs, files in os.walk(seed_root):
            for name in files:
                if not name.endswith(".json"):
                    continue
                with open(os.path.join(root, name), "r", encoding="utf-8") as fh:
                    skill = json.load(fh)
                seen.append(skill["id"])
                errors = jsonschema_min.validate(skill, schema)
                self.assertEqual(errors, [], f"{skill['id']}: {errors}")
        # Spec lists 7 seed skills; this is the canonical sanity check.
        self.assertEqual(len(seen), 7, f"Expected 7 seed skills, found: {seen}")

    def test_runtime_seeds_skills_into_storage_skills_tree(self):
        config = temp_config(tempfile.mkdtemp())
        Runtime(config, prefer_backend="offline")  # triggers _seed_storage
        registry = SkillRegistry(os.path.join(config.storage_root, "skills"))
        ids = {s.id for s in registry.load()}
        self.assertIn("research.source_grounded_summary", ids)
        self.assertIn("general.no_guess_answering", ids)


class ContextOSSkillIntegrationTests(unittest.TestCase):
    """Context OS must select skills from the registry and log them in Trace."""

    def test_research_task_pulls_source_grounded_skill(self):
        config = temp_config(tempfile.mkdtemp())
        runtime = Runtime(config, prefer_backend="offline")
        envelope = {
            "host": {"type": "cli", "host_task_id": "skill-int-1",
                     "source_agent": None, "callback": {}},
            "role_binding": {
                "role_id": "research", "risk_mode": "balanced",
                "privacy_mode": "local_only", "output_profiles": ["markdown"],
            },
            "task_request": {
                "task_id": "skill-int-1", "title": "Summary",
                "instruction": "Using local sources, summarize Heimdal's modes.",
                "inputs": {}, "constraints": {},
                "priority": "P2", "budget": {"quality_level": "B2"},
                "expected_outputs": ["markdown_response"],
            },
            "runtime_hints": {},
        }
        result = runtime.run_envelope(envelope)
        # The trace records the skills Context OS chose.
        trace = Storage.read_json(result["trace_pack"]["path"])
        packet_event = next(
            e for e in trace["events"] if e["name"] == "context_packet_ready"
        )
        selected = packet_event["data"]["skills"]
        self.assertTrue(selected, "expected at least one skill selected")
        # Skill usage stats were bumped after the run completed.
        registry = SkillRegistry(runtime.storage.path("skills"))
        any_with_uses = any(
            (registry.find(sid) or Skill.from_dict(_skill(sid))).performance.get("uses", 0) > 0
            for sid in selected
        )
        self.assertTrue(any_with_uses, "expected at least one skill's stats bumped")


class SkillCLITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.manifest = write_temp_manifest(self.tmp, self.tmp)
        # Seed storage by booting a runtime once, so the registry has skills.
        config = temp_config(self.tmp)
        Runtime(config, prefer_backend="offline")

    def test_skill_list_command(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["skill", "list", "--json", "--manifest", self.manifest])
        self.assertEqual(code, 0)
        skills = json.loads(buf.getvalue())
        self.assertTrue(any(
            s["id"] == "research.source_grounded_summary" for s in skills
        ))

    def test_skill_search_finds_source_grounded(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["skill", "search", "source grounded",
                         "--json", "--manifest", self.manifest])
        self.assertEqual(code, 0)
        hits = json.loads(buf.getvalue())
        self.assertTrue(any(s["id"] == "research.source_grounded_summary" for s in hits))

    def test_skill_show_returns_full_skill(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["skill", "show", "research.source_grounded_summary",
                         "--manifest", self.manifest])
        self.assertEqual(code, 0)
        self.assertIn("source_grounded_summary", buf.getvalue())

    def test_skill_stats_returns_role_breakdown(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["skill", "stats", "--json", "--manifest", self.manifest])
        self.assertEqual(code, 0)
        stats = json.loads(buf.getvalue())
        self.assertGreaterEqual(stats["total_skills"], 7)
        self.assertIn("research", stats["by_role"])


if __name__ == "__main__":
    unittest.main()
