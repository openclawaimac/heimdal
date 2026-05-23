"""v0.6.2: Runtime Profiles -- hardware-adaptive budgets."""

import contextlib
import io
import json
import os
import tempfile
import unittest

from tests.helpers import temp_config, write_temp_manifest

from heimdal.cli import main
from heimdal.core.runtime import Runtime
from heimdal.hardware import runtime_profile
from heimdal.storage import Storage


class ProfileLimitsTests(unittest.TestCase):
    def test_builtin_profiles_have_required_keys(self):
        for name in runtime_profile.PROFILES:
            limits = runtime_profile.limits_for(name)
            for key in ("max_context_tokens", "default_quality_level",
                        "max_repair_iterations"):
                self.assertIn(key, limits)

    def test_manifest_override_wins_over_builtin(self):
        overrides = {"dev": {"max_context_tokens": 99999}}
        limits = runtime_profile.limits_for(
            "dev", manifest_profiles=overrides,
        )
        self.assertEqual(limits["max_context_tokens"], 99999)

    def test_unknown_name_falls_back_to_dev(self):
        limits = runtime_profile.limits_for("not-a-real-profile")
        dev_limits = runtime_profile.limits_for("dev")
        self.assertEqual(limits["max_context_tokens"], dev_limits["max_context_tokens"])

    def test_profile_progression_is_monotonic_in_context_budget(self):
        # CPU is tightest; factory is biggest. Useful sanity check.
        cpu = runtime_profile.limits_for("cpu_only")["max_context_tokens"]
        factory = runtime_profile.limits_for("factory")["max_context_tokens"]
        self.assertLess(cpu, factory)


class ActiveProfileTests(unittest.TestCase):
    def setUp(self):
        self.config = temp_config(tempfile.mkdtemp())
        self.storage = Storage(self.config.storage_root).ensure()

    def test_active_auto_detects_when_no_stored_profile(self):
        active = runtime_profile.active(self.storage, self.config)
        self.assertEqual(active["source"], "auto")
        self.assertIn(active["name"], runtime_profile.PROFILES)

    def test_active_reads_stored_profile_with_manual_source(self):
        runtime_profile.write(self.storage, "single_gpu")
        active = runtime_profile.active(self.storage, self.config)
        self.assertEqual(active["name"], "single_gpu")
        self.assertEqual(active["source"], "manual")

    def test_write_rejects_unknown_profile_name(self):
        with self.assertRaises(ValueError):
            runtime_profile.write(self.storage, "nope")


class RuntimeIntegrationTests(unittest.TestCase):
    def test_runtime_metrics_include_profile_fields(self):
        config = temp_config(tempfile.mkdtemp())
        # Pin to single_gpu so its specific limits show up in metrics.
        storage = Storage(config.storage_root).ensure()
        runtime_profile.write(storage, "single_gpu")
        runtime = Runtime(config, prefer_backend="offline")
        result = runtime.run_envelope({
            "host": {"type": "cli", "host_task_id": "rp-1",
                     "source_agent": None, "callback": {}},
            "role_binding": {
                "role_id": "general", "risk_mode": "balanced",
                "privacy_mode": "local_only", "output_profiles": ["markdown"],
            },
            "task_request": {
                "task_id": "rp-1", "title": "Demo",
                "instruction": "Explain a queue.", "inputs": {},
                "constraints": {}, "priority": "P2",
                "budget": {"quality_level": "B1"},
                "expected_outputs": ["markdown_response"],
            },
            "runtime_hints": {},
        })
        metrics = result["metrics"]
        for key in ("runtime_profile", "profile_source", "profile_limits"):
            self.assertIn(key, metrics)
        self.assertEqual(metrics["runtime_profile"], "single_gpu")
        self.assertEqual(metrics["profile_source"], "manual")

    def test_profile_overrides_default_quality_when_task_does_not_set_it(self):
        # No `budget.quality_level` on the task -> profile default wins.
        config = temp_config(tempfile.mkdtemp())
        storage = Storage(config.storage_root).ensure()
        runtime_profile.write(storage, "factory")  # default_quality_level=B3
        runtime = Runtime(config, prefer_backend="offline")
        result = runtime.run_envelope({
            "host": {"type": "cli", "host_task_id": "rp-2",
                     "source_agent": None, "callback": {}},
            "role_binding": {
                "role_id": "general", "risk_mode": "balanced",
                "privacy_mode": "local_only", "output_profiles": ["markdown"],
            },
            "task_request": {
                "task_id": "rp-2", "title": "Demo",
                "instruction": "Explain a queue.", "inputs": {},
                "constraints": {"max_words": 60}, "priority": "P2",
                "budget": {},  # intentionally empty
                "expected_outputs": ["markdown_response"],
            },
            "runtime_hints": {},
        })
        self.assertEqual(result["metrics"]["quality_level"], "B3")

    def test_explicit_task_budget_overrides_profile_default(self):
        config = temp_config(tempfile.mkdtemp())
        storage = Storage(config.storage_root).ensure()
        runtime_profile.write(storage, "factory")
        runtime = Runtime(config, prefer_backend="offline")
        result = runtime.run_envelope({
            "host": {"type": "cli", "host_task_id": "rp-3",
                     "source_agent": None, "callback": {}},
            "role_binding": {
                "role_id": "general", "risk_mode": "balanced",
                "privacy_mode": "local_only", "output_profiles": ["markdown"],
            },
            "task_request": {
                "task_id": "rp-3", "title": "Demo",
                "instruction": "Explain a queue.", "inputs": {},
                "constraints": {}, "priority": "P2",
                "budget": {"quality_level": "B1"},
                "expected_outputs": ["markdown_response"],
            },
            "runtime_hints": {},
        })
        self.assertEqual(result["metrics"]["quality_level"], "B1")


class ProfileCLITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.manifest = write_temp_manifest(self.tmp, self.tmp)

    def test_profile_detect_prints_name(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["profile", "detect", "--manifest", self.manifest])
        self.assertEqual(code, 0)
        self.assertIn(buf.getvalue().strip(), runtime_profile.PROFILES)

    def test_profile_set_then_show_round_trips(self):
        self.assertEqual(
            main(["profile", "set", "pipeline", "--manifest", self.manifest]),
            0,
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(
                main(["profile", "show", "--json", "--manifest", self.manifest]),
                0,
            )
        active = json.loads(buf.getvalue())
        self.assertEqual(active["name"], "pipeline")
        self.assertEqual(active["source"], "manual")

    def test_profile_explain_includes_limits(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(
                main(["profile", "explain", "cpu_only", "--manifest", self.manifest]),
                0,
            )
        out = buf.getvalue()
        self.assertIn("cpu_only", out)
        self.assertIn("max_context_tokens", out)


if __name__ == "__main__":
    unittest.main()
