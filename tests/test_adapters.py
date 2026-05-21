"""Acceptance: OpenClaw payload maps to a valid envelope and back; CLI adapter wraps input."""

import tempfile
import unittest

from tests.helpers import repo_path, temp_config

from heimdal.adapters.cli_adapter import CLIAdapter
from heimdal.adapters.openclaw_adapter import OpenClawAdapter
from heimdal.core import intake
from heimdal.core.runtime import Runtime
from heimdal.storage import Storage


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.config = temp_config(self.tmp)

    def test_openclaw_payload_becomes_valid_envelope(self):
        payload = Storage.read_json(repo_path("examples/tasks/openclaw_task.example.json"))
        envelope = OpenClawAdapter().to_host_task_envelope(payload)
        self.assertEqual(envelope["host"]["type"], "openclaw")
        self.assertEqual(intake.intake(envelope, self.config), envelope)

    def test_openclaw_round_trip(self):
        payload = Storage.read_json(repo_path("examples/tasks/openclaw_task.example.json"))
        adapter = OpenClawAdapter()
        envelope = adapter.to_host_task_envelope(payload)
        result = Runtime(self.config, prefer_backend="offline").run_envelope(envelope)
        host_result = adapter.from_heimdal_result(result)
        self.assertEqual(host_result["openclaw_task_id"], "oc-7781")
        self.assertIn(host_result["outcome"], ("pass", "need_input", "fail"))

    def test_cli_adapter_wraps_instruction_string(self):
        envelope = CLIAdapter().to_host_task_envelope("Explain what a queue is.")
        self.assertEqual(envelope["host"]["type"], "cli")
        self.assertEqual(intake.intake(envelope, self.config), envelope)

    def test_cli_adapter_passes_through_envelope(self):
        envelope = Storage.read_json(repo_path("examples/tasks/simple_task.json"))
        self.assertIs(CLIAdapter().to_host_task_envelope(envelope), envelope)


if __name__ == "__main__":
    unittest.main()
