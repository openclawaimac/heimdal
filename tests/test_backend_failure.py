"""A dead model backend is a result, not a traceback.

When Ollama cannot be reached -- or every endpoint in a multi-GPU pool has
failed -- there is no draft to verify and no quality verdict to report. That
is infrastructure, and a host calling `handle()` in-process must still get a
Result Envelope carrying a machine-readable code, with the Trace Pack written
so the outage stays diagnosable.
"""

import contextlib
import io
import json
import os
import socket
import tempfile
import unittest
import urllib.error

from tests.helpers import temp_config, write_temp_manifest

from heimdal.adapters.hermes_host import handle as hermes_handle
from heimdal.adapters.openclaw_host import handle as openclaw_handle
from heimdal.cli import main
import yaml

from heimdal.core import model_router, status_codes
from heimdal.core.runtime import Runtime
from heimdal.models.base import ModelBackend
from heimdal.models.ollama import OllamaError, _error_code
from heimdal.storage import Storage


def _envelope(task_id: str = "bf-1") -> dict:
    return {
        "host": {"type": "cli", "host_task_id": task_id,
                 "source_agent": None, "callback": {}},
        "role_binding": {"role_id": "general", "risk_mode": "balanced",
                         "privacy_mode": "local_only",
                         "output_profiles": ["markdown"]},
        "task_request": {"task_id": task_id, "title": "Backend failure demo",
                         "instruction": "Explain what a queue is and how it behaves.",
                         "inputs": {}, "constraints": {}, "priority": "P2",
                         "budget": {}, "expected_outputs": ["markdown_response"]},
        "runtime_hints": {},
    }


class DeadBackend(ModelBackend):
    """Raises on every generation, as a downed Ollama server would."""

    name = "ollama"

    def __init__(self, error: Exception | None = None):
        self.error = error or OllamaError(
            "Ollama is not reachable at http://localhost:11434 (refused).",
            code=status_codes.OLLAMA_UNREACHABLE,
        )

    def is_available(self) -> bool:
        return False

    def list_models(self) -> list[str]:
        return ["stub-model"]

    def generate(self, prompt, **kwargs):
        raise self.error


def _runtime_with(backend: ModelBackend, config=None) -> Runtime:
    config = config or temp_config(tempfile.mkdtemp())
    runtime = Runtime(config, prefer_backend="offline")
    runtime.backend = backend
    runtime.endpoint_pool.default_backend = backend
    return runtime


class ErrorClassificationTests(unittest.TestCase):
    """Each code implies a different fix, so they must not be conflated."""

    def test_missing_model_is_not_reported_as_unreachable(self):
        exc = urllib.error.HTTPError("u", 404, "not found", {}, None)
        self.assertEqual(_error_code(exc), status_codes.OLLAMA_MODEL_MISSING)

    def test_server_error_means_the_server_answered(self):
        exc = urllib.error.HTTPError("u", 500, "boom", {}, None)
        self.assertEqual(_error_code(exc), status_codes.OLLAMA_REQUEST_FAILED)

    def test_timeout_is_distinct_from_unreachable(self):
        self.assertEqual(_error_code(socket.timeout()), status_codes.OLLAMA_TIMEOUT)

    def test_connection_failure_is_unreachable(self):
        exc = urllib.error.URLError("connection refused")
        self.assertEqual(_error_code(exc), status_codes.OLLAMA_UNREACHABLE)

    def test_unparseable_response_is_a_request_failure(self):
        self.assertEqual(_error_code(ValueError("bad json")),
                         status_codes.OLLAMA_REQUEST_FAILED)

    def test_unknown_failure_defaults_to_unreachable(self):
        self.assertEqual(_error_code(None), status_codes.OLLAMA_UNREACHABLE)

    def test_every_backend_code_is_registered(self):
        for code in status_codes.BACKEND_CODES:
            self.assertIn(code, status_codes.ALL_CODES)

    def test_error_carries_its_code(self):
        exc = OllamaError("x", code=status_codes.OLLAMA_TIMEOUT)
        self.assertEqual(exc.code, status_codes.OLLAMA_TIMEOUT)
        # The default keeps older raise sites working.
        self.assertEqual(OllamaError("x").code, status_codes.OLLAMA_UNREACHABLE)


class RunEnvelopeBackendFailureTests(unittest.TestCase):
    def test_dead_backend_returns_a_result_rather_than_raising(self):
        result = _runtime_with(DeadBackend()).run_envelope(_envelope())
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["code"], status_codes.OLLAMA_UNREACHABLE)
        self.assertIn("not reachable", result["message"])

    def test_the_code_reflects_the_underlying_failure(self):
        cases = {
            status_codes.OLLAMA_MODEL_MISSING: "model 'x' is not installed",
            status_codes.OLLAMA_TIMEOUT: "timed out after 120s",
            status_codes.OLLAMA_REQUEST_FAILED: "returned HTTP 500",
        }
        for code, message in cases.items():
            with self.subTest(code=code):
                runtime = _runtime_with(DeadBackend(OllamaError(message, code=code)))
                result = runtime.run_envelope(_envelope("bf-" + code))
                self.assertEqual(result["code"], code)

    def test_trace_pack_is_written_so_the_outage_is_diagnosable(self):
        result = _runtime_with(DeadBackend()).run_envelope(_envelope("bf-trace"))
        path = result["trace_pack"]["path"]
        self.assertTrue(os.path.exists(path))
        trace = Storage.read_json(path)
        names = [e["name"] for e in trace["events"]]
        # The run got as far as building context before the model call failed.
        self.assertIn("intake_ok", names)
        self.assertIn("context_packet_ready", names)
        self.assertIn("backend_failure", names)
        failure = next(e for e in trace["events"] if e["name"] == "backend_failure")
        self.assertEqual(failure["data"]["code"], status_codes.OLLAMA_UNREACHABLE)
        self.assertEqual(trace["status"], "fail")

    def test_repro_pack_is_written_but_carries_no_models(self):
        result = _runtime_with(DeadBackend()).run_envelope(_envelope("bf-repro"))
        repro = Storage.read_json(result["repro_pack"]["path"])
        self.assertEqual(repro["models"], [])
        self.assertIn("contract", repro["hashes"])

    def test_metrics_report_endpoint_health(self):
        result = _runtime_with(DeadBackend()).run_envelope(_envelope("bf-metrics"))
        metrics = result["metrics"]
        self.assertIn("endpoint_health", metrics)
        self.assertIn("endpoint_failovers", metrics)
        self.assertIn("runtime_profile", metrics)

    def test_unrelated_exceptions_still_propagate(self):
        # Only backend outages become results; a bug must still surface.
        class BuggyBackend(DeadBackend):
            def generate(self, prompt, **kwargs):
                raise KeyError("worker_model")

        with self.assertRaises(KeyError):
            _runtime_with(BuggyBackend()).run_envelope(_envelope("bf-bug"))

    def test_a_healthy_run_is_untouched(self):
        config = temp_config(tempfile.mkdtemp())
        result = Runtime(config, prefer_backend="offline").run_envelope(
            _envelope("bf-ok")
        )
        self.assertEqual(result["status"], "pass")
        self.assertNotIn(result.get("code"), status_codes.BACKEND_CODES)


class VerifyEnvelopeBackendFailureTests(unittest.TestCase):
    """`heimdal verify` must honour the same contract as a full run."""

    def test_rule_based_verification_does_not_need_the_backend_at_all(self):
        # The default verifier is deterministic, so a dead Ollama is simply
        # irrelevant to it -- it must not be turned into a backend failure.
        result = _runtime_with(DeadBackend()).verify_envelope(
            _envelope("bf-verify"), "A queue is first-in, first-out."
        )
        self.assertNotIn(result.get("code"), status_codes.BACKEND_CODES)

    def test_unresolvable_model_is_a_fail_not_a_lenient_pass(self):
        # Verification without a model is not a verdict. Returning pass here
        # would silently bless an unverified answer.
        class Unreachable(DeadBackend):
            base_url = "http://127.0.0.1:9"

            def list_models(self) -> list[str]:
                return []

        config = temp_config(tempfile.mkdtemp())
        config.manifest["verifier"] = dict(
            config.manifest.get("verifier", {}), mode="hybrid",
        )
        runtime = _runtime_with(Unreachable(), config)
        result = runtime.verify_envelope(
            _envelope("bf-verify-2"), "A queue is first-in, first-out."
        )
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["code"], status_codes.OLLAMA_UNREACHABLE)
        trace = Storage.read_json(result["trace_pack"]["path"])
        self.assertIn("backend_failure", [e["name"] for e in trace["events"]])


class HostAdapterBackendFailureTests(unittest.TestCase):
    """Hosts call handle() in-process; a traceback there breaks the host."""

    _INSTRUCTION = "Explain what a queue is and how it behaves."

    def _hermes_payload(self) -> dict:
        return {
            "hermes_session_id": "s1",
            "invocation_id": "inv-1",
            "from_agent": "Hermes",
            "role": "general",
            "request": {
                "id": "inv-1-t1", "title": "Hermes task",
                "instruction": self._INSTRUCTION, "inputs": {},
                "constraints": {}, "budget": {"quality_level": "B1"},
                "output_profiles": ["markdown"],
                "expected_outputs": ["markdown_response"],
            },
            "policy": {"privacy_mode": "local_only", "risk_mode": "balanced"},
            "callback": {},
        }

    def _openclaw_payload(self) -> dict:
        return {
            "openclaw_task_id": "oc-1",
            "assigned_role": "general",
            "from_agent": "planner",
            "callback": {},
            "task": {
                "id": "oc-1-t1", "title": "OpenClaw task",
                "prompt": self._INSTRUCTION, "constraints": {},
                "output_profiles": ["markdown"],
                "budget": {"quality_level": "B1"},
                "expected_outputs": ["markdown_response"],
            },
            "policy": {"privacy_mode": "local_only"},
        }

    def test_hermes_gets_a_schema_valid_result_with_the_code(self):
        # handle() validates against hermes_result.schema.json before
        # returning, so reaching the assertions proves schema validity.
        result = hermes_handle(self._hermes_payload(),
                               runtime=_runtime_with(DeadBackend()))
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["code"], status_codes.OLLAMA_UNREACHABLE)
        self.assertEqual(result["hermes_session_id"], "s1")
        self.assertTrue(result["trace_pack_ref"])

    def test_openclaw_gets_a_result_with_the_code(self):
        result = openclaw_handle(self._openclaw_payload(),
                                 runtime=_runtime_with(DeadBackend()))
        self.assertEqual(result["outcome"], "fail")
        self.assertEqual(result["code"], status_codes.OLLAMA_UNREACHABLE)


class RouterModelResolutionTests(unittest.TestCase):
    """An empty model list has two causes with two different fixes."""

    class Unreachable(ModelBackend):
        name = "ollama"
        base_url = "http://127.0.0.1:9"

        def is_available(self) -> bool:
            return False

        def list_models(self) -> list[str]:
            return []

    class EmptyButUp(Unreachable):
        def is_available(self) -> bool:
            return True

    def _resolve(self, backend):
        config = temp_config(tempfile.mkdtemp())
        return model_router._resolve_model("worker", set(), config, backend)

    def test_unreachable_server_is_not_reported_as_a_missing_model(self):
        with self.assertRaises(model_router.ModelUnavailableError) as ctx:
            self._resolve(self.Unreachable())
        self.assertEqual(ctx.exception.code, status_codes.OLLAMA_UNREACHABLE)
        self.assertIn("not reachable", str(ctx.exception))
        # Telling the user to pull a model would send them the wrong way.
        self.assertNotIn("ollama pull", str(ctx.exception))

    def test_reachable_server_with_no_models_asks_for_a_pull(self):
        with self.assertRaises(model_router.ModelUnavailableError) as ctx:
            self._resolve(self.EmptyButUp())
        self.assertEqual(ctx.exception.code, status_codes.OLLAMA_MODEL_MISSING)
        self.assertIn("ollama pull", str(ctx.exception))


class CLIBackendFailureTests(unittest.TestCase):
    """`heimdal run` against a dead Ollama: a result, not a stack trace."""

    def setUp(self):
        self.manifest = write_temp_manifest(tempfile.mkdtemp(), tempfile.mkdtemp())

    def _point_at_dead_ollama(self) -> None:
        # Port 9 (discard) refuses immediately, so this stays fast.
        with open(self.manifest, encoding="utf-8") as fh:
            manifest = yaml.safe_load(fh)
        manifest["ollama"]["base_url"] = "http://127.0.0.1:9"
        with open(self.manifest, "w", encoding="utf-8") as fh:
            yaml.safe_dump(manifest, fh)

    def _run(self, argv) -> tuple[int, str]:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(argv)
        return code, buf.getvalue()

    def test_outage_exits_2_and_prints_the_code(self):
        # A caller scripting `heimdal run` could tell a crash (2) from a
        # failed answer (1) before outages became structured results; that
        # distinction must survive.
        self._point_at_dead_ollama()
        code, out = self._run([
            "run", "--instruction", "Explain what a queue is and how it behaves.",
            "--backend", "ollama", "--manifest", self.manifest,
        ])
        self.assertEqual(code, 2)
        self.assertIn(status_codes.OLLAMA_UNREACHABLE, out)
        self.assertIn("status : fail", out)
        # The packs are still written and pointed at.
        self.assertIn("trace_pack:", out)

    def test_outage_json_output_carries_the_code(self):
        self._point_at_dead_ollama()
        code, out = self._run([
            "run", "--instruction", "Explain what a queue is and how it behaves.",
            "--backend", "ollama", "--json", "--manifest", self.manifest,
        ])
        self.assertEqual(code, 2)
        payload = json.loads(out)
        self.assertEqual(payload["code"], status_codes.OLLAMA_UNREACHABLE)
        self.assertEqual(payload["status"], "fail")

    def test_a_passing_run_exits_0_and_prints_no_code_line(self):
        code, out = self._run([
            "run", "--instruction", "Explain what a queue is and how it behaves.",
            "--backend", "offline", "--manifest", self.manifest,
        ])
        self.assertEqual(code, 0)
        self.assertNotIn("code   :", out)
