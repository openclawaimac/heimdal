"""Acceptance: patch validation, the v0.4.1 promotion lifecycle, and the
eval-runner summary writer."""

import contextlib
import io
import json
import os
import tempfile
import unittest

from tests.helpers import repo_path, temp_config, write_temp_manifest

from heimdal.cli import main
from heimdal.core import eval_runner, patch_manager
from heimdal.core.runtime import Runtime
from heimdal.ids import new_id, now_iso
from heimdal.storage import Storage


def _proposal_patch(patch_id: str | None = None, **overrides) -> dict:
    """A Dream-Mode-shaped patch with the new v0.4.1 metadata."""
    patch = {
        "id": patch_id or new_id("patch"),
        "type": "prompt_patch",
        "channel": "experimental",
        "target": "role_pack:general:system_context",
        "change": {"append": "Lead with the direct answer."},
        "rationale": "Demo runs were too abstract.",
        "created_at": now_iso(),
        "eval_run": None,
        "source": "dream_mode",
        "created_by": "dream_mode",
        "intent": "Improve answer directness for general role.",
        "risk_level": "low",
        "rollback": {"note": "Remove the appended sentence."},
    }
    patch.update(overrides)
    return patch


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


class PatchLifecycleTests(unittest.TestCase):
    """v0.4.1: list / show / review / eval / promote / reject lifecycle."""

    def setUp(self):
        self.config = temp_config(tempfile.mkdtemp())
        self.storage = Storage(self.config.storage_root).ensure()

    def _install(self, patch: dict) -> str:
        return patch_manager.install_patch(self.config, patch)

    def test_list_returns_installed_patches_by_channel(self):
        self._install(_proposal_patch("patch_alpha"))
        promoted = _proposal_patch("patch_beta", channel="beta")
        self._install(promoted)
        all_patches = patch_manager.list_patches(self.config)
        ids = {p["id"] for p in all_patches}
        self.assertEqual(ids, {"patch_alpha", "patch_beta"})
        only_beta = patch_manager.list_patches(self.config, channel="beta")
        self.assertEqual([p["id"] for p in only_beta], ["patch_beta"])

    def test_find_patch_returns_patch_channel_and_path(self):
        self._install(_proposal_patch("patch_findme"))
        found = patch_manager.find_patch(self.config, "patch_findme")
        self.assertIsNotNone(found)
        patch, channel, path = found
        self.assertEqual(patch["id"], "patch_findme")
        self.assertEqual(channel, "experimental")
        self.assertTrue(os.path.exists(path))
        self.assertIsNone(patch_manager.find_patch(self.config, "patch_missing"))

    def test_review_flags_missing_intent_and_rollback(self):
        patch = _proposal_patch("patch_review_1")
        del patch["intent"]
        del patch["rollback"]
        review = patch_manager.review_patch(patch)
        self.assertFalse(review["auto_appliable"])
        self.assertTrue(any("intent" in i.lower() for i in review["issues"]))
        self.assertTrue(any("rollback" in i.lower() for i in review["issues"]))

    def test_review_artifact_persists(self):
        patch = _proposal_patch("patch_review_2")
        self._install(patch)
        review = patch_manager.review_patch(patch)
        path = patch_manager.write_review(self.config, review)
        self.assertTrue(os.path.exists(path))
        self.assertEqual(Storage.read_json(path)["patch_id"], "patch_review_2")

    def test_eval_patch_writes_candidate_eval_and_recommendation(self):
        patch = _proposal_patch("patch_eval_1")
        self._install(patch)
        runtime = Runtime(self.config, prefer_backend="offline")
        report = patch_manager.eval_patch(self.config, patch, runtime)
        self.assertEqual(report["patch_id"], "patch_eval_1")
        self.assertIn(report["recommendation"], ("promote", "needs_review", "reject"))
        # Candidate eval file lives at the documented path.
        eval_path = self.storage.path("patches/evals/patch_eval_1.eval.json")
        self.assertTrue(os.path.exists(eval_path))

    def test_promote_experimental_to_beta_requires_intent(self):
        patch = _proposal_patch("patch_promote_no_intent")
        del patch["intent"]
        self._install(patch)
        with self.assertRaises(patch_manager.PatchError):
            patch_manager.promote_patch(
                self.config, "patch_promote_no_intent", "beta"
            )

    def test_promote_to_beta_moves_file(self):
        self._install(_proposal_patch("patch_to_beta"))
        patch_manager.promote_patch(self.config, "patch_to_beta", "beta")
        found = patch_manager.find_patch(self.config, "patch_to_beta")
        self.assertEqual(found[1], "beta")
        # Old experimental file is gone.
        old = self.storage.path("patches/experimental/patch_to_beta.json")
        self.assertFalse(os.path.exists(old))

    def test_promote_to_stable_requires_candidate_eval(self):
        self._install(_proposal_patch("patch_stable_blocked"))
        with self.assertRaises(patch_manager.PatchError):
            patch_manager.promote_patch(self.config, "patch_stable_blocked", "stable")

    def test_promote_to_stable_with_passing_candidate_eval(self):
        patch = _proposal_patch("patch_stable_ok")
        self._install(patch)
        runtime = Runtime(self.config, prefer_backend="offline")
        report = patch_manager.eval_patch(self.config, patch, runtime)
        # Offline backend hits 40/40, so the candidate eval should recommend
        # promotion. Stable promotion then walks the gate.
        self.assertIn(report["recommendation"], ("promote", "needs_review"))
        if report["recommendation"] == "promote":
            patch_manager.promote_patch(self.config, "patch_stable_ok", "stable")
            found = patch_manager.find_patch(self.config, "patch_stable_ok")
            self.assertEqual(found[1], "stable")

    def test_reject_records_reason_and_moves_to_rejected(self):
        self._install(_proposal_patch("patch_to_reject"))
        patch_manager.reject_patch(self.config, "patch_to_reject", "noisy change")
        found = patch_manager.find_patch(self.config, "patch_to_reject")
        self.assertEqual(found[1], "rejected")
        self.assertEqual(
            found[0]["review"]["rejection_reason"], "noisy change"
        )


class PatchCLITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.manifest = write_temp_manifest(self.tmp, self.tmp)
        config = temp_config(self.tmp)
        patch_manager.install_patch(config, _proposal_patch("cli_patch_001"))

    def test_patch_list_command(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["patch", "list", "--json", "--manifest", self.manifest])
        self.assertEqual(code, 0)
        patches = json.loads(buf.getvalue())
        self.assertTrue(any(p["id"] == "cli_patch_001" for p in patches))

    def test_patch_show_command(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(
                ["patch", "show", "cli_patch_001", "--manifest", self.manifest]
            )
        self.assertEqual(code, 0)
        self.assertIn("cli_patch_001", buf.getvalue())

    def test_patch_review_writes_artifact(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(
                ["patch", "review", "cli_patch_001",
                 "--json", "--manifest", self.manifest]
            )
        self.assertEqual(code, 0)
        review = json.loads(buf.getvalue())
        self.assertEqual(review["patch_id"], "cli_patch_001")
        review_file = os.path.join(
            self.tmp, "patches", "reviews", "cli_patch_001.review.json"
        )
        self.assertTrue(os.path.exists(review_file))

    def test_patch_reject_command(self):
        code = main(
            [
                "patch", "reject", "cli_patch_001",
                "--reason", "noisy", "--manifest", self.manifest,
            ]
        )
        self.assertEqual(code, 0)
        self.assertTrue(
            os.path.exists(os.path.join(
                self.tmp, "patches", "rejected", "cli_patch_001.json"
            ))
        )


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
        # Default run is rule_based; no semantic verifier model.
        self.assertEqual(metadata["verifier_backend"], "rule_based")
        self.assertIsNone(metadata["semantic_verifier_model"])


if __name__ == "__main__":
    unittest.main()
