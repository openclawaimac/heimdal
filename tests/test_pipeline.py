"""Acceptance: demo run writes Repro Pack and Trace Pack."""

import os
import tempfile
import unittest

from tests.helpers import repo_path, temp_config

from heimdal.core.runtime import Runtime
from heimdal.storage import Storage


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.config = temp_config(self.tmp)
        self.runtime = Runtime(self.config, prefer_backend="offline")

    def test_demo_writes_repro_and_trace_packs(self):
        result = self.runtime.run_demo()
        repro_path = result["repro_pack"]["path"]
        trace_path = result["trace_pack"]["path"]
        self.assertTrue(os.path.exists(repro_path), "Repro Pack must be written")
        self.assertTrue(os.path.exists(trace_path), "Trace Pack must be written")

        repro = Storage.read_json(repro_path)
        for field in ("id", "timestamp", "models", "params", "hashes", "versions"):
            self.assertIn(field, repro)
        trace = Storage.read_json(trace_path)
        for field in ("id", "task_id", "events"):
            self.assertIn(field, trace)

    def test_run_input_task_passes(self):
        envelope = Storage.read_json(repo_path("examples/tasks/simple_task.json"))
        result = self.runtime.run_envelope(envelope)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["metrics"]["backend"], "offline")

    def test_response_artifact_respects_word_limit(self):
        result = self.runtime.run_demo()
        response_path = next(
            (a["path"] for a in result["artifacts"] if a["type"] == "response"), None
        )
        self.assertIsNotNone(response_path)
        with open(response_path, "r", encoding="utf-8") as fh:
            text = fh.read()
        self.assertLessEqual(len(text.split()), 120)


if __name__ == "__main__":
    unittest.main()
