"""v0.2.3: Hybrid semantic verifier.

Gate order: deterministic rule-based -> optional model-based semantic -> final
deterministic check. A model-based verifier must never override a
deterministic hard fail.
"""

import json
import os
import tempfile
import unittest

from tests.helpers import temp_config

from heimdal import jsonschema_min
from heimdal.core import eval_runner, model_router, verifier
from heimdal.core.runtime import Runtime
from heimdal.models.offline import OfflineBackend
from heimdal.storage import Storage

_CONTRACT = {"objective": "Explain what a queue is.", "verification": {}, "constraints": {}}
_PACKET = {"truth_context": []}
_HYBRID_ROUTING = {
    "verifier_strictness": "standard",
    "verifier_backend": "hybrid",
    "quality_level": "B2",
    "semantic_verifier_model": "heimdal-offline-stub",
}
_RULE_ROUTING = {
    "verifier_strictness": "standard",
    "verifier_backend": "rule_based",
    "quality_level": "B2",
}


def _b2_task(instruction: str) -> dict:
    return {
        "host": {"type": "cli", "host_task_id": "sv-test", "source_agent": None, "callback": {}},
        "role_binding": {
            "role_id": "general",
            "risk_mode": "balanced",
            "privacy_mode": "local_only",
            "output_profiles": ["markdown"],
        },
        "task_request": {
            "task_id": "sv-test",
            "title": "Hybrid task",
            "instruction": instruction,
            "inputs": {},
            "constraints": {"max_words": 120},
            "priority": "P2",
            "budget": {"quality_level": "B2"},
            "expected_outputs": ["markdown_response"],
        },
        "runtime_hints": {},
    }


class DeterministicGateTests(unittest.TestCase):
    """Test 1 + 5: rule-based behaviour and deterministic short-circuit."""

    def setUp(self):
        self.config = temp_config(tempfile.mkdtemp())
        self.backend = OfflineBackend()

    def test_rule_based_mode_runs_no_semantic_verifier(self):
        result = verifier.verify(
            "A queue is a first-in first-out data structure.",
            _CONTRACT, _PACKET, _RULE_ROUTING, self.config, self.backend,
        )
        self.assertEqual(result["status"], "pass")
        self.assertIsNone(result["semantic"])  # rule_based never invokes the model
        self.assertEqual(result["verifier_backend"], "rule_based")

    def test_rule_based_failure_short_circuits_before_semantic(self):
        # An empty answer is a deterministic hard fail; the semantic verifier
        # must not run, so result["semantic"] stays None.
        result = verifier.verify(
            "", _CONTRACT, _PACKET, _HYBRID_ROUTING, self.config, self.backend
        )
        self.assertEqual(result["status"], "fail")
        self.assertIsNone(result["semantic"])


class HybridSemanticTests(unittest.TestCase):
    """Test 3 + 4 + 6: semantic verdicts and schema validity."""

    def setUp(self):
        self.config = temp_config(tempfile.mkdtemp())
        self.backend = OfflineBackend()

    def test_semantically_wrong_but_valid_json_fails(self):
        # Structurally fine, but the answer dodges the task by asking back.
        dodge = json.dumps({"reply": "Which queue did you mean?"})
        result = verifier.verify(
            dodge, _CONTRACT, _PACKET, _HYBRID_ROUTING, self.config, self.backend
        )
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["semantic"]["status"], "fail")
        self.assertTrue(
            any("Semantic verifier" in d["message"] for d in result["defects"])
        )

    def test_semantically_correct_answer_passes(self):
        good = json.dumps(
            {"answer": "A queue is a first-in first-out collection that "
                       "processes items in their arrival order."}
        )
        result = verifier.verify(
            good, _CONTRACT, _PACKET, _HYBRID_ROUTING, self.config, self.backend
        )
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["semantic"]["status"], "pass")

    def test_semantic_result_validates_against_schema(self):
        result = verifier.verify(
            "A queue processes items in arrival order, first in first out.",
            _CONTRACT, _PACKET, _HYBRID_ROUTING, self.config, self.backend,
        )
        schema = jsonschema_min.load_schema(
            self.config.schema_path("semantic_verification.schema.json")
        )
        self.assertEqual(jsonschema_min.validate(result["semantic"], schema), [])
        for field in ("status", "score", "confidence", "defects", "rationale_short"):
            self.assertIn(field, result["semantic"])


class RouterModeTests(unittest.TestCase):
    def setUp(self):
        self.config = temp_config(tempfile.mkdtemp())
        self.backend = OfflineBackend()
        self.role = {"role_id": "general"}

    def _route(self, quality_level, verifier_override=None):
        contract = {
            "budget": {"quality_level": quality_level, "max_iterations": 3},
            "verification": {},
        }
        return model_router.route(
            contract, self.role, self.backend, self.config, None, verifier_override
        )

    def test_rule_based_is_the_default(self):
        self.assertEqual(self._route("B2")["verifier_backend"], "rule_based")

    def test_hybrid_override_only_applies_to_b2_plus(self):
        self.assertEqual(
            self._route("B2", "hybrid")["verifier_backend"], "hybrid"
        )
        self.assertEqual(
            self._route("B3", "hybrid")["verifier_backend"], "hybrid"
        )
        # B0/B1 stay rule_based even when hybrid is requested.
        self.assertEqual(
            self._route("B1", "hybrid")["verifier_backend"], "rule_based"
        )

    def test_hybrid_route_resolves_a_semantic_verifier_model(self):
        routing = self._route("B2", "hybrid")
        self.assertIsNotNone(routing["semantic_verifier_model"])


class HybridRuntimeTests(unittest.TestCase):
    """Test 2 + 7: default behaviour unchanged; hybrid metadata in Trace/Repro."""

    def test_default_run_stays_rule_based(self):
        runtime = Runtime(temp_config(tempfile.mkdtemp()), prefer_backend="offline")
        result = runtime.run_envelope(_b2_task("Explain what a stack is."))
        self.assertEqual(result["metrics"]["verifier_backend"], "rule_based")
        trace = Storage.read_json(result["trace_pack"]["path"])
        self.assertFalse(
            any(e["name"] == "semantic_verify" for e in trace["events"])
        )

    def test_hybrid_run_records_semantic_metadata(self):
        runtime = Runtime(
            temp_config(tempfile.mkdtemp()),
            prefer_backend="offline",
            verifier_override="hybrid",
        )
        result = runtime.run_envelope(_b2_task("Explain what a stack is."))
        self.assertEqual(result["metrics"]["verifier_backend"], "hybrid")
        self.assertIsNotNone(result["metrics"]["semantic_verifier_model"])

        trace = Storage.read_json(result["trace_pack"]["path"])
        routing_event = next(e for e in trace["events"] if e["name"] == "routing")
        self.assertEqual(routing_event["data"]["verifier_backend"], "hybrid")
        semantic_event = next(
            e for e in trace["events"] if e["name"] == "semantic_verify"
        )
        for field in (
            "semantic_verifier_model",
            "semantic_verifier_status",
            "semantic_verifier_score",
            "semantic_verifier_confidence",
        ):
            self.assertIn(field, semantic_event["data"])

        repro = Storage.read_json(result["repro_pack"]["path"])
        self.assertEqual(repro["params"]["verifier_backend"], "hybrid")
        self.assertIsNotNone(repro["params"]["semantic_verifier_model"])


class HybridEvalTests(unittest.TestCase):
    """The --verifier override must reach eval execution and the summary."""

    def test_hybrid_eval_reports_hybrid_metadata_and_runs_semantic_verifier(self):
        runtime = Runtime(
            temp_config(tempfile.mkdtemp()),
            prefer_backend="offline",
            verifier_override="hybrid",
        )
        summary = eval_runner.run_evals(runtime)

        meta = summary["metadata"]
        self.assertEqual(meta["verifier_backend"], "hybrid")
        self.assertIsNotNone(meta["semantic_verifier_model"])
        self.assertEqual(summary["pass_rate"], 1.0)

        # A B2 eval task must actually exercise the semantic verifier.
        trace_dir = runtime.storage.path("logs/trace_packs")
        ran_semantic = False
        for name in os.listdir(trace_dir):
            trace = Storage.read_json(os.path.join(trace_dir, name))
            if any(e["name"] == "semantic_verify" for e in trace["events"]):
                ran_semantic = True
                break
        self.assertTrue(ran_semantic, "no eval trace recorded a semantic_verify event")


if __name__ == "__main__":
    unittest.main()
