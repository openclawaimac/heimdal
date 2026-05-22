"""v0.2.5: Hermes adapter / host integration.

Hermes hands Heimdal a task as a single external skill; Heimdal runs the full
Quality Factory internally. The adapter only translates; hermes_host
orchestrates.
"""

import json
import os
import tempfile
import unittest

from tests.helpers import repo_path, temp_config, write_temp_manifest

from heimdal.adapters.hermes_adapter import HermesAdapter
from heimdal.adapters.hermes_host import handle
from heimdal.cli import main
from heimdal.core import intake
from heimdal.core.runtime import Runtime
from heimdal.storage import Storage


def _payload(instruction, *, role="general", session="hermes-s1", callback=None,
             constraints=None, budget="B1"):
    return {
        "hermes_session_id": session,
        "invocation_id": "inv-1",
        "from_agent": "Hermes",
        "role": role,
        "request": {
            "id": "inv-1-t1",
            "title": "Hermes task",
            "instruction": instruction,
            "inputs": {},
            "constraints": constraints or {},
            "budget": {"quality_level": budget},
            "output_profiles": ["markdown"],
            "expected_outputs": ["markdown_response"],
        },
        "policy": {"privacy_mode": "local_only", "risk_mode": "balanced"},
        "callback": callback or {},
    }


class HermesAdapterTests(unittest.TestCase):
    def setUp(self):
        self.config = temp_config(tempfile.mkdtemp())

    def test_payload_translates_to_valid_envelope(self):
        envelope = HermesAdapter().to_host_task_envelope(_payload("Explain queues."))
        self.assertEqual(envelope["host"]["type"], "hermes")
        self.assertEqual(envelope["host"]["source_agent"], "Hermes")
        self.assertEqual(intake.intake(envelope, self.config), envelope)

    def test_example_payload_is_valid(self):
        payload = Storage.read_json(repo_path("examples/tasks/hermes_task.example.json"))
        envelope = HermesAdapter().to_host_task_envelope(payload)
        self.assertEqual(intake.intake(envelope, self.config), envelope)


class HermesHandleTests(unittest.TestCase):
    def _runtime(self) -> Runtime:
        return Runtime(temp_config(tempfile.mkdtemp()), prefer_backend="offline")

    def test_simple_task_round_trips(self):
        result = handle(_payload("Explain what a queue is."), self._runtime())
        self.assertEqual(result["hermes_session_id"], "hermes-s1")
        self.assertEqual(result["invocation_id"], "inv-1")
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["answer"].strip())
        self.assertTrue(result["repro_pack_ref"])
        self.assertTrue(result["trace_pack_ref"])
        self.assertIn("backend", result["verifier"])

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
        self.assertEqual(result["status"], "need_input")
        self.assertTrue(result["questions"])

    def test_result_hides_internal_sub_agent_details(self):
        result = handle(_payload("Explain what a stack is."), self._runtime())
        # Heimdal is presented as one agent: no internal orchestration graph.
        for internal in ("routing", "packet", "context_packet", "models_used"):
            self.assertNotIn(internal, result)

    def test_callback_file_written_under_workspace(self):
        result = handle(
            _payload("Explain what a list is.", callback={"file": "hermes_out.json"}),
            self._runtime(),
        )
        path = result["callback_delivered"]
        self.assertIsNotNone(path)
        self.assertEqual(os.path.basename(os.path.dirname(path)), "workspace")
        self.assertEqual(Storage.read_json(path)["hermes_session_id"], "hermes-s1")


class HermesCLITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.manifest = write_temp_manifest(self.tmp, self.tmp)

    def test_hermes_run_command(self):
        code = main(
            [
                "hermes", "run",
                "--input", repo_path("examples/tasks/hermes_task.example.json"),
                "--offline", "--json", "--manifest", self.manifest,
            ]
        )
        self.assertEqual(code, 0)
        self.assertTrue(
            os.path.exists(os.path.join(self.tmp, "workspace", "hermes_result.json"))
        )

    def test_hermes_run_requires_input(self):
        with self.assertRaises(SystemExit):
            main(["hermes", "run", "--manifest", self.manifest])


if __name__ == "__main__":
    unittest.main()
