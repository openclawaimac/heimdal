"""Acceptance: doctor runs cleanly without Ollama and writes a profile."""

import os
import tempfile
import unittest

from tests.helpers import temp_config, write_temp_manifest

from heimdal.cli import main
from heimdal.hardware.profiler import deployment_mode, full_profile


class DoctorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_full_profile_structure(self):
        config = temp_config(self.tmp)
        profile = full_profile(config, run_capability_tests=True)
        for key in ("os", "cpu", "ram_gb", "gpu", "ollama", "deployment_mode"):
            self.assertIn(key, profile)
        # Capability tests are a list; empty is fine when Ollama is absent.
        self.assertIsInstance(profile["capability_tests"], list)

    def test_doctor_without_ollama_warns(self):
        config = temp_config(self.tmp)
        profile = full_profile(config, run_capability_tests=False)
        if not profile["ollama"]["reachable"]:
            self.assertTrue(
                any("Ollama" in w for w in profile["warnings"]),
                "expected an Ollama warning when it is unreachable",
            )

    def test_doctor_cli_exits_zero_and_writes_profile(self):
        manifest = write_temp_manifest(self.tmp, self.tmp)
        exit_code = main(["doctor", "--json", "--manifest", manifest])
        self.assertEqual(exit_code, 0)
        profiles_dir = os.path.join(self.tmp, "logs", "hardware_profiles")
        self.assertTrue(os.path.isdir(profiles_dir))
        self.assertTrue(os.listdir(profiles_dir), "doctor should write a hardware profile")

    def test_deployment_mode(self):
        self.assertEqual(deployment_mode(0), "Dev")
        self.assertEqual(deployment_mode(1), "Single Device")
        self.assertEqual(deployment_mode(8), "Factory")


if __name__ == "__main__":
    unittest.main()
