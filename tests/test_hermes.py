"""v0.2.5: Hermes adapter / host integration.

Hermes hands Heimdal a task as a single external skill; Heimdal runs the full
Quality Factory internally. The adapter only translates; hermes_host
orchestrates.
"""

import contextlib
import io
import json
import os
import tempfile
import unittest

from tests.helpers import repo_path, temp_config, write_temp_manifest

from heimdal import __version__, jsonschema_min
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
        runtime = self._runtime()
        result = handle(
            _payload("Explain what a list is.", callback={"file": "hermes_out.json"}),
            runtime,
        )
        delivery = result["callback_delivery"]
        self.assertEqual(delivery["status"], "success")
        self.assertEqual(delivery["target_ref"], "workspace/hermes_out.json")
        written = Storage.read_json(
            os.path.join(runtime.storage.root, delivery["target_ref"])
        )
        self.assertEqual(written["hermes_session_id"], "hermes-s1")

    def test_need_input_carries_code_and_needed_inputs(self):
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
        self.assertEqual(result["code"], "SOURCE_MISSING")
        self.assertTrue(result["needed_inputs"])
        entry = result["needed_inputs"][0]
        for key in ("type", "reason", "missing_topic", "suggested_action"):
            self.assertIn(key, entry)

    def test_result_validates_against_hermes_schema(self):
        # handle() validates its output against the Hermes result schema; a
        # passing run round-trips a schema-valid result with no internal leakage.
        result = handle(_payload("Explain what a queue is."), self._runtime())
        self.assertEqual(result["code"], "OK")
        errors = jsonschema_min.validate(
            result, jsonschema_min.load_schema(repo_path("schemas/hermes_result.schema.json"))
        )
        self.assertEqual(errors, [])
        for internal in ("prompt", "system", "routing", "packet", "context_packet"):
            self.assertNotIn(internal, result)
        self.assertNotIn("context_packet", [a["type"] for a in result["artifacts"]])

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
        self.assertEqual(result["callback_delivery"]["status"], "success")
        self.assertEqual(result["callback_delivery"]["target_ref"], "workspace/cb.json")
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
        # Internal Context Packet + Task Contract artifacts are not exposed.
        types = [a["type"] for a in result["artifacts"]]
        self.assertNotIn("context_packet", types)
        self.assertNotIn("task_contract", types)
        # The legacy absolute-path key is gone.
        self.assertNotIn("callback_delivered", result)
        # callback_delivery exposes a relative ref, never an absolute path.
        delivery = result["callback_delivery"]
        self.assertFalse(os.path.isabs(delivery["target_ref"]))
        # The delivered callback file carries no absolute internal path.
        written = json.dumps(
            Storage.read_json(os.path.join(runtime.storage.root, delivery["target_ref"]))
        )
        self.assertNotIn(runtime.storage.root, written)

    def test_missing_topic_is_concise(self):
        # _missing_topic distills "...refund policy for Product Y." down to a
        # short topic so the host knows what to supply, not the whole prompt.
        result = handle(
            _payload(
                "Using only the local Truth Vault, state the refund policy "
                "for Product Y.",
                role="research",
                constraints={"requires_sources": True},
                budget="B2",
            ),
            self._runtime(),
        )
        topic = result["needed_inputs"][0]["missing_topic"]
        self.assertEqual(topic, "Product Y refund policy")

    def test_host_visible_artifacts_exclude_internals(self):
        result = handle(_payload("Explain what a queue is."), self._runtime())
        types = [a["type"] for a in result["artifacts"]]
        self.assertIn("response", types)
        self.assertNotIn("context_packet", types)
        self.assertNotIn("task_contract", types)

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
        self.assertEqual(main(["hermes", "run", "--manifest", self.manifest]), 2)

    def test_hermes_capabilities_command(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["hermes", "capabilities", "--json", "--manifest", self.manifest])
        self.assertEqual(code, 0)
        caps = json.loads(buf.getvalue())
        self.assertEqual(caps["heimdal_version"], __version__)
        self.assertTrue(caps["supports_verify_only"])
        self.assertTrue(caps["supports_needed_inputs"])
        self.assertIn("hybrid", caps["supported_verifiers"])

    def test_hermes_doctor_offline_passes(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(
                [
                    "hermes", "doctor",
                    "--input", repo_path("examples/tasks/hermes_task.example.json"),
                    "--backend", "offline", "--json", "--manifest", self.manifest,
                ]
            )
        self.assertEqual(code, 0)
        report = json.loads(buf.getvalue())
        self.assertEqual(report["status"], "pass")
        names = [c["name"] for c in report["checks"]]
        for required in (
            "payload_valid",
            "hermes_schema_loadable",
            "end_to_end_run",
            "no_absolute_paths",
            "no_internal_artifacts",
            "no_internal_fields",
        ):
            self.assertIn(required, names)

    def test_hermes_doctor_requires_input(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["hermes", "doctor", "--json", "--manifest", self.manifest])
        self.assertEqual(code, 1)
        report = json.loads(buf.getvalue())
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["checks"][0]["name"], "input_provided")


class MissingTopicTests(unittest.TestCase):
    """v0.2.7: deterministic concise-topic extraction for needed_inputs."""

    def test_for_pivot(self):
        from heimdal.core.quality_factory import _missing_topic
        self.assertEqual(
            _missing_topic(
                "Using only the local Truth Vault, state the refund policy "
                "for Product Y."
            ),
            "Product Y refund policy",
        )

    def test_of_pivot(self):
        from heimdal.core.quality_factory import _missing_topic
        self.assertEqual(
            _missing_topic("State the exact subscription price of Product Zeta."),
            "Product Zeta subscription price",
        )

    def test_falls_back_to_cleaned_instruction(self):
        from heimdal.core.quality_factory import _missing_topic
        self.assertEqual(_missing_topic("Hello world."), "Hello world")


if __name__ == "__main__":
    unittest.main()
