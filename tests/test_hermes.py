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

    def test_source_missing_task_returns_need_input_not_fail(self):
        # A source-required task whose source is absent from the Truth Vault
        # must return need_input -- the hybrid semantic verifier must never
        # convert a missing source into a verification fail. The Trace Pack
        # records a no_guess_gate event with a reason and retrieval refs.
        runtime = self._runtime()
        result = handle(
            _payload(
                "Using only the local Truth Vault, state the refund policy "
                "for Product Y.",
                role="research",
                constraints={"requires_sources": True},
                budget="B2",
            ),
            runtime,
        )
        self.assertEqual(result["status"], "need_input")
        self.assertTrue(result["questions"])
        trace = Storage.read_json(
            os.path.join(runtime.storage.root, result["trace_pack_ref"])
        )
        gate = next(e for e in trace["events"] if e["name"] == "no_guess_gate")
        self.assertEqual(gate["data"]["outcome"], "need_input")
        self.assertIn("reason", gate["data"])
        self.assertIn("retrieval_refs", gate["data"])

    def test_callback_delivery_is_traced(self):
        runtime = self._runtime()
        result = handle(
            _payload("Explain what a queue is.", callback={"file": "cb.json"}),
            runtime,
        )
        self.assertIsNotNone(result["callback_delivered"])
        trace = Storage.read_json(
            os.path.join(runtime.storage.root, result["trace_pack_ref"])
        )
        names = [e["name"] for e in trace["events"]]
        self.assertIn("callback_delivery_start", names)
        self.assertIn("callback_delivery_success", names)
        success = next(
            e for e in trace["events"] if e["name"] == "callback_delivery_success"
        )
        # Only the sanitized target is traced -- not the result payload.
        self.assertEqual(success["data"]["target"], "workspace/cb.json")

    def test_result_exposes_no_absolute_internal_paths(self):
        runtime = self._runtime()
        result = handle(
            _payload("Explain what a tree is.", callback={"file": "out.json"}),
            runtime,
        )
        # Pack and artifact refs are host-safe relative refs.
        self.assertFalse(os.path.isabs(result["repro_pack_ref"]))
        self.assertFalse(os.path.isabs(result["trace_pack_ref"]))
        for artifact in result["artifacts"]:
            self.assertIn("ref", artifact)
            self.assertFalse(os.path.isabs(artifact["ref"]))
        # The internal Context Packet artifact is not exposed externally.
        self.assertNotIn("context_packet", [a["type"] for a in result["artifacts"]])
        # The delivered callback file carries no absolute internal path.
        written = json.dumps(Storage.read_json(result["callback_delivered"]))
        self.assertNotIn(runtime.storage.root, written)

    def test_handle_accepts_backend_and_verifier_overrides(self):
        # handle() is a supported integration entrypoint: it accepts the same
        # backend/model/verifier overrides as `heimdal hermes run`.
        result = handle(
            _payload("Explain what a queue is.", budget="B2"),
            backend="offline",
            verifier="hybrid",
        )
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["verifier"]["backend"], "hybrid")


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
