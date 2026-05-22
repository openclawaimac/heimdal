"""v0.2.4: OpenClaw live integration.

handle() drives Heimdal end to end from an OpenClaw payload; callback files
land under storage/workspace; the `heimdal openclaw run` CLI works.
"""

import contextlib
import io
import json
import os
import tempfile
import unittest

from tests.helpers import repo_path, temp_config, write_temp_manifest

from heimdal.adapters.openclaw_host import handle
from heimdal.cli import main
from heimdal.core.runtime import Runtime
from heimdal.storage import Storage


def _payload(prompt, *, role="general", task_id="oc-1", callback=None,
             constraints=None, budget="B1"):
    return {
        "openclaw_task_id": task_id,
        "assigned_role": role,
        "from_agent": "planner",
        "callback": callback or {},
        "task": {
            "id": f"{task_id}-t1",
            "title": "OpenClaw task",
            "prompt": prompt,
            "constraints": constraints or {},
            "output_profiles": ["markdown"],
            "budget": {"quality_level": budget},
            "expected_outputs": ["markdown_response"],
        },
        "policy": {"privacy_mode": "local_only"},
    }


class OpenClawHandleTests(unittest.TestCase):
    def _runtime(self) -> Runtime:
        return Runtime(temp_config(tempfile.mkdtemp()), prefer_backend="offline")

    def test_passing_task_round_trips(self):
        result = handle(_payload("Explain what a queue is."), self._runtime())
        self.assertEqual(result["openclaw_task_id"], "oc-1")
        self.assertEqual(result["heimdal_task_id"], "oc-1-t1")
        self.assertEqual(result["outcome"], "pass")
        self.assertTrue(result["answer"].strip())
        self.assertTrue(result["repro_pack_ref"])
        self.assertTrue(result["trace_pack_ref"])

    def test_source_required_task_returns_need_input(self):
        result = handle(
            _payload(
                "State the exact subscription price of Product Zeta.",
                role="research",
                constraints={"requires_sources": True},
                budget="B2",
            ),
            self._runtime(),
        )
        self.assertEqual(result["outcome"], "need_input")
        self.assertTrue(result["questions"])

    def test_callback_file_written_under_workspace(self):
        runtime = self._runtime()
        result = handle(
            _payload("Explain what a stack is.", callback={"file": "oc_out.json"}),
            runtime,
        )
        delivery = result["callback_delivery"]
        self.assertEqual(delivery["status"], "success")
        self.assertEqual(delivery["target_ref"], "workspace/oc_out.json")
        written = Storage.read_json(
            os.path.join(runtime.storage.root, delivery["target_ref"])
        )
        self.assertEqual(written["openclaw_task_id"], "oc-1")
        self.assertEqual(written["outcome"], "pass")

    def test_callback_path_traversal_is_contained(self):
        runtime = self._runtime()
        result = handle(
            _payload("Explain what a list is.", callback={"file": "../../etc/evil.json"}),
            runtime,
        )
        delivery = result["callback_delivery"]
        # Directory components stripped: the file stays inside workspace.
        self.assertEqual(delivery["target_ref"], "workspace/evil.json")
        path = os.path.join(runtime.storage.root, "workspace", "evil.json")
        self.assertTrue(os.path.exists(path))

    def test_no_callback_delivers_nothing(self):
        result = handle(_payload("Explain what a tree is."), self._runtime())
        self.assertIsNone(result["callback_delivery"])


class OpenClawCLITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.manifest = write_temp_manifest(self.tmp, self.tmp)

    def test_openclaw_run_command(self):
        code = main(
            [
                "openclaw", "run",
                "--input", repo_path("examples/tasks/openclaw_task.example.json"),
                "--offline", "--json", "--manifest", self.manifest,
            ]
        )
        self.assertEqual(code, 0)
        # The example payload requests a file callback.
        self.assertTrue(
            os.path.exists(os.path.join(self.tmp, "workspace", "openclaw_result.json"))
        )

    def test_openclaw_run_requires_input(self):
        self.assertEqual(
            main(["openclaw", "run", "--manifest", self.manifest]), 2
        )

    def test_openclaw_capabilities_command(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(
                ["openclaw", "capabilities", "--json", "--manifest", self.manifest]
            )
        self.assertEqual(code, 0)
        caps = json.loads(buf.getvalue())
        self.assertTrue(caps["supports_openclaw_adapter"])
        self.assertIn("hybrid", caps["supported_verifiers"])

    def test_openclaw_doctor_offline_passes(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(
                [
                    "openclaw", "doctor",
                    "--input", repo_path("examples/tasks/openclaw_task.example.json"),
                    "--backend", "offline", "--json", "--manifest", self.manifest,
                ]
            )
        self.assertEqual(code, 0)
        report = json.loads(buf.getvalue())
        self.assertEqual(report["status"], "pass")


if __name__ == "__main__":
    unittest.main()
