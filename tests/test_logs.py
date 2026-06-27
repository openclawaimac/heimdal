"""`heimdal logs latest` -- surfaces the v0.6.x runtime decision fields."""

import contextlib
import io
import json
import tempfile
import unittest

from tests.helpers import temp_config, write_temp_manifest

from heimdal.cli import main
from heimdal.core.runtime import Runtime


def _seed_run(tmp: str) -> None:
    """Produce one trace pack so `logs latest` has something to read."""
    runtime = Runtime(temp_config(tmp), prefer_backend="offline")
    runtime.run_envelope({
        "host": {"type": "cli", "host_task_id": "log-1",
                 "source_agent": None, "callback": {}},
        "role_binding": {"role_id": "general", "risk_mode": "balanced",
                         "privacy_mode": "local_only", "output_profiles": ["markdown"]},
        "task_request": {"task_id": "log-1", "title": "Demo",
                         "instruction": "Explain a queue.", "inputs": {},
                         "constraints": {}, "priority": "P2", "budget": {},
                         "expected_outputs": ["markdown_response"]},
        "runtime_hints": {},
    })


class LogsLatestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.manifest = write_temp_manifest(self.tmp, self.tmp)

    def test_no_runs_yet(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["logs", "latest", "--manifest", self.manifest])
        self.assertEqual(code, 0)
        self.assertIn("No runs logged yet", buf.getvalue())

    def test_json_output_includes_runtime_profile_metrics(self):
        _seed_run(self.tmp)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["logs", "latest", "--json", "--manifest", self.manifest])
        self.assertEqual(code, 0)
        latest = json.loads(buf.getvalue())
        metrics = latest["metrics"]
        for key in ("runtime_profile", "profile_source"):
            self.assertIn(key, metrics)

    def test_human_output_surfaces_profile_line(self):
        _seed_run(self.tmp)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(
                main(["logs", "latest", "--manifest", self.manifest]), 0
            )
        out = buf.getvalue()
        # The v0.6.x profile line is surfaced explicitly, not just buried in
        # the metrics JSON blob.
        self.assertIn("profile:", out)


if __name__ == "__main__":
    unittest.main()
