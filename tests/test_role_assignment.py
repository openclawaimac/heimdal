"""v0.6.1: Model Role Assignment."""

import contextlib
import io
import json
import os
import tempfile
import unittest

from tests.helpers import temp_config, write_temp_manifest

from heimdal.cli import main
from heimdal.hardware import role_assigner
from heimdal.storage import Storage


def _matrix(*, installed: list[str], caps: dict, profile: str = "single_gpu") -> dict:
    return {
        "ollama": {"reachable": True, "base_url": "http://localhost:11434",
                   "models": installed},
        "model_capabilities": caps,
        "recommended_runtime_profile": profile,
        "warnings": [],
    }


def _passing(extra: dict | None = None) -> dict:
    base = {
        "basic_generation": "pass",
        "json_output": "pass",
        "semantic_judgment": "pass",
        "worker_candidate": True,
        "semantic_verifier_candidate": True,
    }
    if extra:
        base.update(extra)
    return base


_ROLES_CFG = {
    "worker": {"preferred": ["qwen2.5:7b"], "min_capabilities": ["basic_generation", "json_output"]},
    "verifier": {"preferred": ["qwen3:4b"], "min_capabilities": ["json_output", "semantic_judgment"]},
    "semantic_verifier": {"preferred": ["qwen2.5:7b"], "min_capabilities": ["json_output", "semantic_judgment"]},
    "brain": {"preferred": ["qwen2.5:14b"], "min_capabilities": ["basic_generation", "json_output"]},
    "coder": {"preferred": ["qwen2.5-coder:14b"], "min_capabilities": ["basic_generation", "json_output"]},
}


class AssignTests(unittest.TestCase):
    def test_preferred_model_installed_and_passes_wins_for_worker(self):
        matrix = _matrix(
            installed=["qwen2.5:7b", "llama3.2:3b"],
            caps={"qwen2.5:7b": _passing(), "llama3.2:3b": _passing()},
        )
        result = role_assigner.assign(matrix, model_roles_cfg=_ROLES_CFG)
        worker = result["assignments"]["worker"]
        self.assertEqual(worker["backend"], "ollama")
        self.assertEqual(worker["model"], "qwen2.5:7b")
        self.assertEqual(worker["source"], role_assigner.SOURCE_AUTO_TUNER)
        self.assertFalse(worker["fallback"])

    def test_missing_preferred_falls_back_to_any_candidate(self):
        matrix = _matrix(
            installed=["llama3.2:3b"],
            caps={"llama3.2:3b": _passing()},
        )
        result = role_assigner.assign(matrix, model_roles_cfg=_ROLES_CFG)
        worker = result["assignments"]["worker"]
        self.assertEqual(worker["model"], "llama3.2:3b")
        self.assertTrue(worker["fallback"])
        self.assertEqual(worker["source"], role_assigner.SOURCE_FALLBACK)

    def test_no_installed_models_uses_safe_role_defaults(self):
        matrix = _matrix(installed=[], caps={})
        result = role_assigner.assign(matrix, model_roles_cfg=_ROLES_CFG)
        # Worker falls back to deterministic offline; verifier to rule-based;
        # brain + coder optional roles report disabled.
        self.assertEqual(result["assignments"]["worker"]["backend"], "offline")
        self.assertEqual(result["assignments"]["verifier"]["backend"], "rule_based")
        self.assertEqual(result["assignments"]["brain"]["backend"], "disabled")
        self.assertEqual(result["assignments"]["coder"]["backend"], "disabled")

    def test_embedding_models_are_never_assigned_to_generation_roles(self):
        matrix = _matrix(
            installed=["nomic-embed-text", "qwen2.5:7b"],
            caps={
                "nomic-embed-text": {"skipped": True,
                                     "reason": "embedding model"},
                "qwen2.5:7b": _passing(),
            },
        )
        result = role_assigner.assign(matrix, model_roles_cfg=_ROLES_CFG)
        for role in ("worker", "verifier", "semantic_verifier"):
            self.assertNotEqual(
                result["assignments"][role]["model"], "nomic-embed-text",
            )

    def test_operator_pin_overrides_auto_assignment(self):
        matrix = _matrix(
            installed=["qwen2.5:7b", "llama3.2:3b"],
            caps={"qwen2.5:7b": _passing(), "llama3.2:3b": _passing()},
        )
        pins = {"worker": {"model": "llama3.2:3b", "pinned_at": "t"}}
        result = role_assigner.assign(
            matrix, model_roles_cfg=_ROLES_CFG, pins=pins,
        )
        worker = result["assignments"]["worker"]
        self.assertEqual(worker["model"], "llama3.2:3b")
        self.assertEqual(worker["source"], role_assigner.SOURCE_OPERATOR_PIN)

    def test_pin_to_missing_model_falls_back_with_warning(self):
        matrix = _matrix(
            installed=["qwen2.5:7b"],
            caps={"qwen2.5:7b": _passing()},
        )
        pins = {"worker": {"model": "not-installed", "pinned_at": "t"}}
        result = role_assigner.assign(
            matrix, model_roles_cfg=_ROLES_CFG, pins=pins,
        )
        # Auto-assignment still wins; warning is recorded on the role.
        self.assertEqual(result["assignments"]["worker"]["model"], "qwen2.5:7b")
        self.assertTrue(any("not-installed" in w for w in result["warnings"]))


class WritePinsAndAssignmentsTests(unittest.TestCase):
    def setUp(self):
        self.config = temp_config(tempfile.mkdtemp())
        self.storage = Storage(self.config.storage_root).ensure()

    def test_write_and_load_assignments_round_trips(self):
        matrix = _matrix(
            installed=["qwen2.5:7b"], caps={"qwen2.5:7b": _passing()},
        )
        assignments = role_assigner.assign(matrix, model_roles_cfg=_ROLES_CFG)
        role_assigner.write_assignments(self.storage, assignments)
        loaded = role_assigner.load_assignments(self.storage)
        self.assertEqual(
            loaded["assignments"]["worker"]["model"], "qwen2.5:7b",
        )

    def test_write_pin_then_unpin(self):
        role_assigner.write_pin(self.storage, "worker", "qwen2.5:7b")
        pins = role_assigner.load_pins(self.storage)
        self.assertEqual(pins["worker"]["model"], "qwen2.5:7b")
        role_assigner.write_pin(self.storage, "worker", None)
        self.assertEqual(role_assigner.load_pins(self.storage), {})

    def test_assigned_worker_model_returns_none_when_no_artifact(self):
        model, source = role_assigner.assigned_worker_model(self.storage)
        self.assertIsNone(model)
        self.assertIsNone(source)


class ModelsCLITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.manifest = write_temp_manifest(self.tmp, self.tmp)

    def test_models_list_without_ollama_still_works(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["models", "list", "--json", "--manifest", self.manifest])
        self.assertEqual(code, 0)
        data = json.loads(buf.getvalue())
        self.assertIn("reachable", data)

    def test_capabilities_hint_when_reachable_but_no_results(self):
        # Stored matrix: Ollama reachable, models present, but no capability
        # results -> the hint must name the models, not the generic line.
        Storage(self.tmp).ensure().write_json(
            "runtime/capability_matrix.json",
            {
                "ollama": {"reachable": True, "base_url": "http://localhost:11434",
                           "models": ["qwen2.5:7b"]},
                "model_capabilities": {},
                "recommended_runtime_profile": "single_gpu",
                "warnings": [],
            },
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["models", "capabilities", "--manifest", self.manifest])
        self.assertEqual(code, 0)
        out = buf.getvalue()
        self.assertIn("reachable", out)
        self.assertIn("qwen2.5:7b", out)

    def test_models_assign_write_produces_artifact(self):
        code = main([
            "models", "assign", "--write", "--manifest", self.manifest,
        ])
        self.assertEqual(code, 0)
        artifact = os.path.join(self.tmp, "runtime", "role_assignments.json")
        self.assertTrue(os.path.exists(artifact))
        assignments = Storage.read_json(artifact)
        self.assertIn("worker", assignments["assignments"])

    def test_models_pin_then_roles_shows_pin(self):
        main(["models", "pin", "--role", "worker", "--model", "qwen2.5:7b",
              "--manifest", self.manifest])
        # Roles needs the assign artifact too -- generate it.
        main(["models", "assign", "--write", "--manifest", self.manifest])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(
                main(["models", "roles", "--json", "--manifest", self.manifest]),
                0,
            )
        data = json.loads(buf.getvalue())
        self.assertIn("worker", data["assignments"])


if __name__ == "__main__":
    unittest.main()
