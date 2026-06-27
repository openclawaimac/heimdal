"""v0.6.0: Hardware Auto-Tuner & Capability Matrix."""

import contextlib
import io
import json
import os
import tempfile
import unittest

from tests.helpers import temp_config, write_temp_manifest

from heimdal.cli import main
from heimdal.hardware import capability_matrix


class RecommendProfileTests(unittest.TestCase):
    def test_no_gpu_wsl2_resolves_to_dev(self):
        hw = {"os": {"flavour": "wsl2"}, "gpu": {"count": 0, "metal": False}}
        self.assertEqual(capability_matrix.recommend_profile(hw), "dev")

    def test_no_gpu_linux_native_resolves_to_cpu_only(self):
        hw = {"os": {"flavour": "linux_native"}, "gpu": {"count": 0, "metal": False}}
        self.assertEqual(capability_matrix.recommend_profile(hw), "cpu_only")

    def test_apple_silicon_resolves_to_single_gpu(self):
        hw = {"os": {"flavour": "macos"}, "gpu": {"count": 0, "metal": True}}
        self.assertEqual(capability_matrix.recommend_profile(hw), "single_gpu")

    def test_one_gpu_single_gpu(self):
        hw = {"os": {"flavour": "linux_native"}, "gpu": {"count": 1, "metal": False}}
        self.assertEqual(capability_matrix.recommend_profile(hw), "single_gpu")

    def test_two_to_three_gpus_pipeline(self):
        hw = {"os": {"flavour": "linux_native"}, "gpu": {"count": 2, "metal": False}}
        self.assertEqual(capability_matrix.recommend_profile(hw), "pipeline")

    def test_four_or_more_gpus_factory(self):
        hw = {"os": {"flavour": "linux_native"}, "gpu": {"count": 8, "metal": False}}
        self.assertEqual(capability_matrix.recommend_profile(hw), "factory")


class SafeContextTokensTests(unittest.TestCase):
    def test_cpu_only_gets_smallest_budget(self):
        self.assertEqual(capability_matrix.safe_context_tokens(0, has_metal=False), 2048)

    def test_metal_uses_unified_budget(self):
        self.assertEqual(capability_matrix.safe_context_tokens(0, has_metal=True), 8192)

    def test_vram_scales_budget(self):
        self.assertLess(
            capability_matrix.safe_context_tokens(8_000, has_metal=False),
            capability_matrix.safe_context_tokens(24_000, has_metal=False),
        )


class BuildMatrixTests(unittest.TestCase):
    def setUp(self):
        self.config = temp_config(tempfile.mkdtemp())

    def test_build_matrix_without_ollama_keeps_running(self):
        matrix = capability_matrix.build_matrix(
            self.config, run_capability_tests=False,
        )
        for key in ("platform", "hardware", "ollama", "model_capabilities",
                    "recommended_runtime_profile", "warnings"):
            self.assertIn(key, matrix)
        # No ollama in CI -> warning is emitted, not an exception.
        self.assertFalse(matrix["ollama"]["reachable"])
        self.assertTrue(any("Ollama" in w for w in matrix["warnings"]))

    def test_recommended_profile_is_known_value(self):
        matrix = capability_matrix.build_matrix(
            self.config, run_capability_tests=False,
        )
        self.assertIn(
            matrix["recommended_runtime_profile"], capability_matrix.PROFILES
        )

    def test_test_model_skips_embedding_models(self):
        # No backend needed -- the embedding-skip path returns before any call.
        result = capability_matrix.test_model(None, "nomic-embed-text")
        self.assertTrue(result.get("skipped"))


class DeploymentModeDelegationTests(unittest.TestCase):
    """deployment_mode must stay consistent with recommend_profile -- they
    share the same count thresholds via DEPLOYMENT_LABELS."""

    def test_deployment_mode_matches_recommend_profile_for_counts(self):
        from heimdal.hardware.capability_matrix import DEPLOYMENT_LABELS
        from heimdal.hardware.profiler import deployment_mode
        for count in (0, 1, 2, 3, 4, 8):
            profile = capability_matrix.recommend_profile(
                {"gpu": {"count": count, "metal": False},
                 "os": {"flavour": "linux_native"}}
            )
            self.assertEqual(deployment_mode(count), DEPLOYMENT_LABELS[profile])

    def test_legacy_labels_preserved(self):
        from heimdal.hardware.profiler import deployment_mode
        self.assertEqual(deployment_mode(0), "Dev")
        self.assertEqual(deployment_mode(1), "Single Device")
        self.assertEqual(deployment_mode(2), "Pipeline")
        self.assertEqual(deployment_mode(8), "Factory")


class DoctorCLITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.manifest = write_temp_manifest(self.tmp, self.tmp)

    def test_doctor_profile_flag_prints_only_profile_name(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["doctor", "--profile", "--manifest", self.manifest])
        self.assertEqual(code, 0)
        self.assertIn(buf.getvalue().strip(), capability_matrix.PROFILES)

    def test_doctor_write_profile_persists_canonical_copy(self):
        code = main([
            "doctor", "--json", "--write-profile",
            "--manifest", self.manifest,
        ])
        self.assertEqual(code, 0)
        canonical = os.path.join(self.tmp, "runtime", "capability_matrix.json")
        self.assertTrue(os.path.exists(canonical))
        # Also one timestamped copy under logs/capability_matrix/.
        log_dir = os.path.join(self.tmp, "logs", "capability_matrix")
        self.assertTrue(os.path.isdir(log_dir) and os.listdir(log_dir))

    def test_doctor_json_exposes_capability_matrix_structure(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(
                main(["doctor", "--json", "--manifest", self.manifest]), 0
            )
        matrix = json.loads(buf.getvalue())
        for key in ("platform", "hardware", "ollama",
                    "recommended_runtime_profile", "model_capabilities"):
            self.assertIn(key, matrix)


if __name__ == "__main__":
    unittest.main()
