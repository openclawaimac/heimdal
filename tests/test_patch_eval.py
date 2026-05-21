"""Acceptance: patch validation rejects bad schemas; eval runner writes a summary."""

import os
import tempfile
import unittest

from tests.helpers import repo_path, temp_config

from heimdal.core import eval_runner, patch_manager
from heimdal.core.runtime import Runtime


class PatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.config = temp_config(self.tmp)

    def test_valid_patch_accepted(self):
        ok, errors = patch_manager.validate_patch_file(
            repo_path("examples/patches/good.json"), self.config
        )
        self.assertTrue(ok, errors)

    def test_invalid_patch_rejected(self):
        ok, errors = patch_manager.validate_patch_file(
            repo_path("examples/patches/bad.json"), self.config
        )
        self.assertFalse(ok)
        self.assertTrue(errors)

    def test_stable_promotion_requires_eval_pass(self):
        patch = patch_manager.load_patch(repo_path("examples/patches/good.json"))
        with self.assertRaises(patch_manager.PatchError):
            patch_manager.promote(patch, "stable", None, self.config)
        passing = {"eval_run_id": "evalrun_x", "must_pass_all_passed": True, "regressed": False}
        promoted = patch_manager.promote(patch, "stable", passing, self.config)
        self.assertEqual(promoted["channel"], "stable")


class EvalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.config = temp_config(self.tmp)

    def test_eval_run_writes_summary_and_passes_suite(self):
        runtime = Runtime(self.config, prefer_backend="offline")
        summary = eval_runner.run_evals(runtime)
        self.assertTrue(os.path.exists(summary["summary_path"]))
        self.assertEqual(summary["total"], 40)
        self.assertEqual(summary["pass_rate"], 1.0)
        self.assertTrue(summary["must_pass_all_passed"])
        for category, stats in summary["categories"].items():
            self.assertTrue(stats["meets_minimum"], f"{category} below minimum")
        metadata = summary["metadata"]
        for field in ("heimdal_version", "backend", "manifest_path", "platform"):
            self.assertIn(field, metadata)
        self.assertEqual(metadata["backend"], "offline")


if __name__ == "__main__":
    unittest.main()
