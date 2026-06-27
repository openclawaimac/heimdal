"""v0.6.x: brain/planner role runs for B3/B4 tasks only.

The brain role was assigned by the role-assigner since v0.6.1 but never
invoked at runtime. It now runs a planning step before the worker drafts,
gated to B3/B4 so B0-B2 (and the eval suite) are unaffected.
"""

import tempfile
import unittest

from tests.helpers import temp_config

from heimdal.core.runtime import Runtime
from heimdal.storage import Storage


def _envelope(task_id: str, quality_level: str) -> dict:
    return {
        "host": {"type": "cli", "host_task_id": task_id,
                 "source_agent": None, "callback": {}},
        "role_binding": {"role_id": "general", "risk_mode": "balanced",
                         "privacy_mode": "local_only", "output_profiles": ["markdown"]},
        "task_request": {"task_id": task_id, "title": "Brain demo",
                         "instruction": "Explain what a queue is and how it behaves.",
                         "inputs": {}, "constraints": {}, "priority": "P2",
                         "budget": {"quality_level": quality_level},
                         "expected_outputs": ["markdown_response"]},
        "runtime_hints": {},
    }


class BrainRoleTests(unittest.TestCase):
    def setUp(self):
        self.config = temp_config(tempfile.mkdtemp())

    def _run(self, quality_level: str) -> dict:
        runtime = Runtime(self.config, prefer_backend="offline")
        return runtime.run_envelope(_envelope(f"brain-{quality_level}", quality_level))

    def test_b3_task_runs_brain_plan(self):
        result = self._run("B3")
        trace = Storage.read_json(result["trace_pack"]["path"])
        names = [e["name"] for e in trace["events"]]
        self.assertIn("brain_plan", names)
        # The brain model appears in the repro pack's models list.
        repro = Storage.read_json(result["repro_pack"]["path"])
        roles = {m["role"] for m in repro["models"]}
        self.assertIn("brain", roles)

    def test_b1_task_does_not_run_brain_plan(self):
        result = self._run("B1")
        trace = Storage.read_json(result["trace_pack"]["path"])
        names = [e["name"] for e in trace["events"]]
        self.assertNotIn("brain_plan", names)
        repro = Storage.read_json(result["repro_pack"]["path"])
        roles = {m["role"] for m in repro["models"]}
        self.assertNotIn("brain", roles)

    def test_b3_still_passes_verification(self):
        # The plan is prepended to the worker prompt; the run must still pass.
        result = self._run("B3")
        self.assertEqual(result["status"], "pass")


class RoutingBrainModelTests(unittest.TestCase):
    def test_router_sets_brain_model_for_b3_only(self):
        from heimdal.core import model_router
        from heimdal.core.role_binding import resolve_role
        from heimdal.core.task_contract import build_contract
        from heimdal.models.offline import OfflineBackend

        config = temp_config(tempfile.mkdtemp())
        backend = OfflineBackend()
        role = resolve_role({"role_id": "general"})

        def _route(ql):
            env = _envelope("r", ql)
            contract = build_contract(env, role, config)
            return model_router.route(contract, role, backend, config)

        self.assertIsNotNone(_route("B3")["brain_model"])
        self.assertIsNone(_route("B1")["brain_model"])
        self.assertIsNone(_route("B1")["brain_profile"])


if __name__ == "__main__":
    unittest.main()
