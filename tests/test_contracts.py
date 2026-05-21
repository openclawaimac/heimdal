"""Acceptance: Task Contract and Context Packet are created before any model call."""

import json
import tempfile
import unittest

from tests.helpers import repo_path, temp_config

from heimdal.core import intake
from heimdal.core.runtime import Runtime
from heimdal.storage import Storage


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.config = temp_config(self.tmp)
        self.envelope = Storage.read_json(repo_path("examples/tasks/simple_task.json"))

    def test_intake_rejects_invalid_envelope(self):
        bad = {"host": {"type": "cli"}}  # missing host_task_id, role_binding, task_request
        with self.assertRaises(intake.IntakeError):
            intake.intake(bad, self.config)

    def test_intake_accepts_valid_envelope(self):
        self.assertEqual(intake.intake(self.envelope, self.config), self.envelope)

    def test_contract_and_packet_created_before_model_call(self):
        runtime = Runtime(self.config, prefer_backend="offline")
        result = runtime.run_envelope(self.envelope)

        # Artifacts prove the contract and packet were materialised.
        kinds = {a["type"] for a in result["artifacts"]}
        self.assertIn("task_contract", kinds)
        self.assertIn("context_packet", kinds)

        # Trace ordering proves they came before the worker draft.
        trace = Storage.read_json(result["trace_pack"]["path"])
        names = [event["name"] for event in trace["events"]]
        self.assertIn("contract_ready", names)
        self.assertIn("context_packet_ready", names)
        self.assertIn("worker_draft", names)
        self.assertLess(names.index("contract_ready"), names.index("worker_draft"))
        self.assertLess(names.index("context_packet_ready"), names.index("worker_draft"))

    def test_contract_is_schema_valid(self):
        runtime = Runtime(self.config, prefer_backend="offline")
        result = runtime.run_envelope(self.envelope)
        contract_path = next(
            a["path"] for a in result["artifacts"] if a["type"] == "task_contract"
        )
        contract = Storage.read_json(contract_path)
        for field in ("contract_id", "task_id", "objective", "budget", "verification"):
            self.assertIn(field, contract)


if __name__ == "__main__":
    unittest.main()
