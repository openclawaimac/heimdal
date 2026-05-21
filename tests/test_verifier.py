"""Acceptance: verifier returns structured JSON; No-Guess Gate blocks unsourced claims."""

import tempfile
import unittest

from tests.helpers import repo_path, temp_config

from heimdal import jsonschema_min
from heimdal.core.runtime import Runtime
from heimdal.storage import Storage


class VerifierTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.config = temp_config(self.tmp)
        self.runtime = Runtime(self.config, prefer_backend="offline")

    def test_verification_result_is_schema_valid(self):
        envelope = Storage.read_json(repo_path("examples/tasks/simple_task.json"))
        result = self.runtime.run_envelope(envelope)
        verification_path = next(
            a["path"] for a in result["artifacts"] if a["type"] == "verification_result"
        )
        verification = Storage.read_json(verification_path)
        schema = jsonschema_min.load_schema(
            self.config.schema_path("verification_result.schema.json")
        )
        self.assertEqual(jsonschema_min.validate(verification, schema), [])
        self.assertIn(verification["status"], ("pass", "fail"))

    def test_no_guess_gate_returns_need_input(self):
        envelope = Storage.read_json(repo_path("examples/tasks/no_guess_task.json"))
        result = self.runtime.run_envelope(envelope)
        self.assertEqual(result["status"], "need_input")
        self.assertTrue(result["questions"], "need_input must include a question")

    def test_demo_task_passes_verification(self):
        result = self.runtime.run_demo()
        self.assertEqual(result["status"], "pass")


if __name__ == "__main__":
    unittest.main()
