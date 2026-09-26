"""v0.7.1: health-aware endpoint failover.

There are no GPUs and no Ollama in CI, so these tests drive the mechanics
with stub backends that fail on demand: the circuit breaker's open/cooldown
cycle, candidate ordering per failover mode, rerouting a request to a
healthy endpoint, and a full Quality Factory run surviving a dead endpoint
with the fallback recorded in its Trace Pack.
"""

import itertools
import tempfile
import threading
import unittest
import urllib.error

from tests.helpers import temp_config

from heimdal.core import status_codes
from heimdal.core.runtime import Runtime
from heimdal.models.base import GenerationResult, ModelBackend
from heimdal.models.endpoint_pool import EndpointPool
from heimdal.models.failover import (
    Candidate,
    EndpointHealth,
    FailoverBackend,
)
from heimdal.models.ollama import OllamaBackend, OllamaError
from heimdal.storage import Storage

_ENDPOINTS = [
    {"name": "gpu0", "base_url": "http://localhost:11434",
     "roles": ["worker", "brain"]},
    {"name": "gpu1", "base_url": "http://localhost:11435",
     "roles": ["semantic_verifier"]},
]


class StubBackend(ModelBackend):
    """Answers, or raises a chosen error, and counts calls."""

    name = "ollama"

    def __init__(self, tag: str, error: Exception | None = None):
        self.tag = tag
        self.error = error
        self.calls = 0
        self.base_url = f"http://{tag}:11434"

    def is_available(self) -> bool:
        return self.error is None

    def list_models(self) -> list[str]:
        return [] if self.error else [f"{self.tag}-model"]

    def generate(self, prompt, **kwargs):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return GenerationResult(
            text=f"answer from {self.tag}", model=kwargs.get("model", "m"),
            backend=self.name,
        )


def _chain(*backends: StubBackend) -> list[Candidate]:
    return [Candidate(b.tag, b, preferred=(i == 0))
            for i, b in enumerate(backends)]


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class EndpointHealthTests(unittest.TestCase):
    def test_failure_opens_the_circuit(self):
        health = EndpointHealth(cooldown_seconds=60, clock=FakeClock())
        self.assertFalse(health.is_open("gpu0"))
        health.record_failure("gpu0", OllamaError("boom"))
        self.assertTrue(health.is_open("gpu0"))

    def test_threshold_above_one_needs_repeated_failures(self):
        health = EndpointHealth(
            failure_threshold=2, cooldown_seconds=60, clock=FakeClock(),
        )
        health.record_failure("gpu0", OllamaError("boom"))
        self.assertFalse(health.is_open("gpu0"))
        health.record_failure("gpu0", OllamaError("boom"))
        self.assertTrue(health.is_open("gpu0"))

    def test_success_resets_the_failure_run(self):
        health = EndpointHealth(
            failure_threshold=2, cooldown_seconds=60, clock=FakeClock(),
        )
        health.record_failure("gpu0", OllamaError("boom"))
        health.record_success("gpu0")
        health.record_failure("gpu0", OllamaError("boom"))
        self.assertFalse(health.is_open("gpu0"))

    def test_circuit_recloses_after_cooldown(self):
        clock = FakeClock()
        health = EndpointHealth(cooldown_seconds=60, clock=clock)
        health.record_failure("gpu0", OllamaError("boom"))
        clock.advance(59)
        self.assertTrue(health.is_open("gpu0"))
        clock.advance(2)
        self.assertFalse(health.is_open("gpu0"))

    def test_snapshot_reports_counts_and_last_error(self):
        health = EndpointHealth(clock=FakeClock())
        health.record_success("http://gpu0:11434")
        health.record_failure("http://gpu1:11434", OllamaError("connection refused"))
        by_endpoint = {e["endpoint"]: e for e in health.snapshot()}
        self.assertTrue(by_endpoint["http://gpu0:11434"]["healthy"])
        self.assertEqual(by_endpoint["http://gpu0:11434"]["successes"], 1)
        self.assertFalse(by_endpoint["http://gpu1:11434"]["healthy"])
        self.assertIn("connection refused",
                      by_endpoint["http://gpu1:11434"]["last_error"])

    def test_ledger_is_thread_safe(self):
        health = EndpointHealth(failure_threshold=10_000, clock=FakeClock())
        def hammer():
            for _ in range(200):
                health.record_failure("gpu0", OllamaError("x"))
                health.record_success("gpu1")
        threads = [threading.Thread(target=hammer) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        by_endpoint = {e["endpoint"]: e for e in health.snapshot()}
        self.assertEqual(by_endpoint["gpu0"]["failures"], 800)
        self.assertEqual(by_endpoint["gpu1"]["successes"], 800)


class FailoverBackendTests(unittest.TestCase):
    def test_healthy_primary_is_used_and_spare_untouched(self):
        good, spare = StubBackend("gpu0"), StubBackend("gpu1")
        backend = FailoverBackend("worker", _chain(good, spare), EndpointHealth())
        self.assertEqual(backend.generate("hi", model="m").text, "answer from gpu0")
        self.assertEqual((good.calls, spare.calls), (1, 0))

    def test_dead_primary_reroutes_to_the_spare(self):
        dead = StubBackend("gpu0", OllamaError("not reachable"))
        spare = StubBackend("gpu1")
        health = EndpointHealth()
        backend = FailoverBackend("worker", _chain(dead, spare), health)
        self.assertEqual(backend.generate("hi", model="m").text, "answer from gpu1")
        self.assertEqual((dead.calls, spare.calls), (1, 1))
        self.assertEqual(health.failover_count, 1)

    def test_open_circuit_skips_the_dead_endpoint_entirely(self):
        dead = StubBackend("gpu0", OllamaError("down"))
        spare = StubBackend("gpu1")
        backend = FailoverBackend("worker", _chain(dead, spare), EndpointHealth())
        backend.generate("first", model="m")
        backend.generate("second", model="m")
        # The dead endpoint is tried once, then taken out of rotation.
        self.assertEqual(dead.calls, 1)
        self.assertEqual(spare.calls, 2)

    def test_every_request_off_the_primary_counts_as_a_failover(self):
        # Once the circuit is open the primary is skipped rather than tried,
        # so "did we fall back?" cannot be inferred from attempt position --
        # it is measured against the configured first choice.
        dead = StubBackend("gpu0", OllamaError("down"))
        spare = StubBackend("gpu1")
        health = EndpointHealth()
        backend = FailoverBackend("worker", _chain(dead, spare), health)
        events: list[str] = []
        backend.event_sink = lambda name, **data: events.append(name)
        for _ in range(3):
            backend.generate("hi", model="m")
        self.assertEqual(health.failover_count, 3)
        self.assertEqual(events.count("endpoint_failover"), 3)
        # The dead endpoint was only actually contacted once.
        self.assertEqual(events.count("endpoint_unhealthy"), 1)

    def test_skipped_endpoints_are_named_in_the_event(self):
        dead, spare = StubBackend("gpu0", OllamaError("down")), StubBackend("gpu1")
        backend = FailoverBackend("worker", _chain(dead, spare), EndpointHealth())
        payloads: list[dict] = []
        backend.event_sink = lambda name, **data: (
            payloads.append(data) if name == "endpoint_failover" else None
        )
        backend.generate("first", model="m")
        backend.generate("second", model="m")
        # First call attempted and failed on gpu0; second skipped it.
        self.assertEqual(payloads[0]["failed"], ["gpu0"])
        self.assertEqual(payloads[0]["skipped"], [])
        self.assertEqual(payloads[1]["failed"], [])
        self.assertEqual(payloads[1]["skipped"], ["gpu0"])

    def test_recovered_primary_is_used_again_after_cooldown(self):
        clock = FakeClock()
        health = EndpointHealth(cooldown_seconds=60, clock=clock)
        flaky = StubBackend("gpu0", OllamaError("down"))
        spare = StubBackend("gpu1")
        backend = FailoverBackend("worker", _chain(flaky, spare), health)
        backend.generate("hi", model="m")
        flaky.error = None            # the GPU comes back
        clock.advance(61)
        self.assertEqual(backend.generate("hi", model="m").text, "answer from gpu0")
        self.assertEqual(health.failover_count, 1)

    def test_connection_errors_also_trigger_failover(self):
        dead = StubBackend("gpu0", urllib.error.URLError("refused"))
        spare = StubBackend("gpu1")
        backend = FailoverBackend("worker", _chain(dead, spare), EndpointHealth())
        self.assertEqual(backend.generate("hi", model="m").text, "answer from gpu1")

    def test_a_missing_model_fails_over_to_an_endpoint_that_has_it(self):
        # Documented in docs/MULTI_GPU.md: a 404 is worth rerouting, because
        # another node may well have the model pulled.
        missing = OllamaError("model 'x' is not installed (HTTP 404)",
                              code=status_codes.OLLAMA_MODEL_MISSING)
        without, with_it = StubBackend("gpu0", missing), StubBackend("gpu1")
        backend = FailoverBackend(
            "worker", _chain(without, with_it), EndpointHealth(),
        )
        self.assertEqual(backend.generate("hi", model="m").text, "answer from gpu1")

    def test_a_successful_generate_never_health_checks_first(self):
        # Documented as a non-goal: endpoints are not probed ahead of a run,
        # so the happy path costs no extra round trips.
        good = StubBackend("gpu0")
        probed = []
        good.is_available = lambda: probed.append(1) or True
        backend = FailoverBackend("worker", _chain(good), EndpointHealth())
        backend.generate("hi", model="m")
        self.assertEqual(probed, [])

    def test_unrelated_errors_are_not_swallowed(self):
        # A bug in prompt construction must surface, not be retried away.
        broken = StubBackend("gpu0", ValueError("bad prompt"))
        spare = StubBackend("gpu1")
        backend = FailoverBackend("worker", _chain(broken, spare), EndpointHealth())
        with self.assertRaises(ValueError):
            backend.generate("hi", model="m")
        self.assertEqual(spare.calls, 0)

    def test_all_endpoints_down_raises_naming_the_role(self):
        a = StubBackend("gpu0", OllamaError("down"))
        b = StubBackend("gpu1", OllamaError("down"))
        backend = FailoverBackend("semantic_verifier", _chain(a, b), EndpointHealth())
        with self.assertRaises(OllamaError) as ctx:
            backend.generate("hi", model="m")
        message = str(ctx.exception)
        self.assertIn("semantic_verifier", message)
        self.assertIn("gpu0", message)
        self.assertIn("gpu1", message)

    def test_every_circuit_open_still_attempts_rather_than_refusing(self):
        # Both marked down, but one has since recovered. Refusing outright
        # would strand a working cluster behind a stale ledger.
        health = EndpointHealth(cooldown_seconds=10_000, clock=FakeClock())
        recovered, still_dead = StubBackend("gpu0"), StubBackend("gpu1", OllamaError("x"))
        health.record_failure("gpu0", OllamaError("x"))
        health.record_failure("gpu1", OllamaError("x"))
        backend = FailoverBackend("worker", _chain(recovered, still_dead), health)
        self.assertEqual(backend.generate("hi", model="m").text, "answer from gpu0")

    def test_events_report_the_reroute(self):
        dead, spare = StubBackend("gpu0", OllamaError("down")), StubBackend("gpu1")
        backend = FailoverBackend("worker", _chain(dead, spare), EndpointHealth())
        events: list[tuple] = []
        backend.event_sink = lambda name, **data: events.append((name, data))
        backend.generate("hi", model="m")
        names = [n for n, _ in events]
        self.assertIn("endpoint_unhealthy", names)
        self.assertIn("endpoint_failover", names)
        failover = dict(events[names.index("endpoint_failover")][1])
        self.assertEqual(failover["served_by"], "gpu1")
        self.assertEqual(failover["failed"], ["gpu0"])
        self.assertTrue(failover["borrowed"])

    def test_event_sink_reaches_the_wrapped_backends(self):
        good = StubBackend("gpu0")
        backend = FailoverBackend("worker", _chain(good), EndpointHealth())
        sink = lambda name, **data: None
        backend.event_sink = sink
        self.assertIs(good.event_sink, sink)
        backend.event_sink = None
        self.assertIsNone(good.event_sink)

    def test_wrapper_reports_the_primary_backend_identity(self):
        good, spare = StubBackend("gpu0"), StubBackend("gpu1")
        backend = FailoverBackend("worker", _chain(good, spare), EndpointHealth())
        self.assertEqual(backend.name, "ollama")
        self.assertIs(backend.primary, good)
        self.assertEqual(backend.base_url, "http://gpu0:11434")

    def test_list_models_falls_through_to_a_live_endpoint(self):
        dead, spare = StubBackend("gpu0", OllamaError("x")), StubBackend("gpu1")
        backend = FailoverBackend("worker", _chain(dead, spare), EndpointHealth())
        self.assertEqual(backend.list_models(), ["gpu1-model"])


class PoolFailoverPolicyTests(unittest.TestCase):
    def _pool(self, mode="auto", endpoints=None) -> EndpointPool:
        config = temp_config(tempfile.mkdtemp())
        config.manifest["ollama"] = dict(
            config.manifest.get("ollama", {}),
            endpoints=endpoints if endpoints is not None else _ENDPOINTS,
            failover=mode,
        )
        return EndpointPool(OllamaBackend("http://localhost:11434"), config)

    def test_auto_borrows_other_endpoints_as_spares(self):
        pool = self._pool("auto")
        self.assertEqual(pool.failover_map()["semantic_verifier"], ["gpu1", "gpu0"])

    def test_strict_stays_within_the_roles_own_endpoints(self):
        pool = self._pool("strict")
        self.assertEqual(pool.failover_map()["semantic_verifier"], ["gpu1"])

    def test_off_disables_failover(self):
        pool = self._pool("off")
        for chain in pool.failover_map().values():
            self.assertEqual(len(chain), 1)
        # A single candidate means no wrapper at all.
        self.assertNotIsInstance(pool.backend_for_role("worker"), FailoverBackend)

    def test_unknown_mode_falls_back_to_auto(self):
        self.assertEqual(self._pool("nonsense").failover_mode, "auto")

    def test_strict_still_covers_a_role_with_several_endpoints(self):
        pool = self._pool("strict", endpoints=[
            {"name": "gpu0", "base_url": "http://localhost:11434", "roles": ["worker"]},
            {"name": "gpu1", "base_url": "http://localhost:11435", "roles": ["worker"]},
        ])
        self.assertEqual(pool.failover_map()["worker"], ["gpu0", "gpu1"])

    def test_duplicate_urls_are_not_tried_twice(self):
        pool = self._pool("auto", endpoints=[
            {"name": "primary", "base_url": "http://localhost:11434", "roles": ["worker"]},
            {"name": "alias", "base_url": "http://localhost:11434", "roles": []},
        ])
        self.assertEqual(pool.failover_map()["worker"], ["primary"])

    def test_no_endpoints_configured_means_no_wrapper(self):
        pool = self._pool("auto", endpoints=[])
        self.assertIs(pool.backend_for_role("worker"), pool.default_backend)
        self.assertEqual(pool.failover_count(), 0)

    def test_each_worker_slot_keeps_its_own_first_choice(self):
        pool = self._pool("auto", endpoints=[
            {"name": "gpu0", "base_url": "http://localhost:11434", "roles": ["worker"]},
            {"name": "gpu1", "base_url": "http://localhost:11435", "roles": ["worker"]},
        ])
        primaries = [b.primary.base_url for b in pool.worker_backends()]
        self.assertEqual(primaries,
                         ["http://localhost:11434", "http://localhost:11435"])


class AnsweringBackend(ModelBackend):
    """Produces pipeline-valid output, optionally failing the first N calls."""

    name = "ollama"

    def __init__(self, tag: str, fail_first: int = 0):
        self.tag = tag
        self.base_url = f"http://{tag}:11434"
        self.calls = 0
        self._remaining_failures = fail_first
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        return True

    def list_models(self) -> list[str]:
        return [f"{self.tag}-model"]

    def generate(self, prompt, *, model, system="", json_mode=False,
                 max_tokens=512, temperature=0.2, structured=None):
        import json as _json
        with self._lock:
            self.calls += 1
            if self._remaining_failures > 0:
                self._remaining_failures -= 1
                raise OllamaError(f"{self.tag} is not reachable")
        spec = structured or {}
        if spec.get("verify_task") == "semantic":
            text = _json.dumps({"status": "pass", "score": 0.9,
                                "confidence": 0.8, "defects": [],
                                "rationale_short": "ok"})
        elif spec.get("brain_task") == "plan":
            text = "1. Plan step one.\n2. Plan step two."
        else:
            text = ("A queue is a first-in, first-out structure where items "
                    "are added at the back and removed from the front, which "
                    "keeps processing in arrival order.")
        return GenerationResult(text=text, model=model, backend=self.name)


def _envelope(task_id: str, quality_level: str = "B1") -> dict:
    return {
        "host": {"type": "cli", "host_task_id": task_id,
                 "source_agent": None, "callback": {}},
        "role_binding": {"role_id": "general", "risk_mode": "balanced",
                         "privacy_mode": "local_only",
                         "output_profiles": ["markdown"]},
        "task_request": {"task_id": task_id, "title": "Failover demo",
                         "instruction": "Explain what a queue is and how it behaves.",
                         "inputs": {}, "constraints": {}, "priority": "P2",
                         "budget": {"quality_level": quality_level},
                         "expected_outputs": ["markdown_response"]},
        "runtime_hints": {},
    }


class RunSurvivesDeadEndpointTests(unittest.TestCase):
    """A dead endpoint must degrade the run, not end it."""

    def _runtime_with(self, gpu0: ModelBackend, gpu1: ModelBackend):
        config = temp_config(tempfile.mkdtemp())
        config.manifest["ollama"] = dict(
            config.manifest.get("ollama", {}), endpoints=_ENDPOINTS,
        )
        runtime = Runtime(config, prefer_backend="offline")
        pool = EndpointPool(OllamaBackend("http://localhost:11434"), config)
        # Pre-seed the per-endpoint backend cache with stubs so no HTTP
        # happens; the pool's own routing and failover logic is untouched.
        pool._backends["gpu0"] = gpu0
        pool._backends["gpu1"] = gpu1
        runtime.endpoint_pool = pool
        return runtime, pool

    def test_run_completes_when_the_worker_endpoint_is_down(self):
        dead = AnsweringBackend("gpu0", fail_first=99)
        spare = AnsweringBackend("gpu1")
        runtime, pool = self._runtime_with(dead, spare)

        result = runtime.run_envelope(_envelope("fo-1"))

        self.assertEqual(result["status"], "pass")
        self.assertGreaterEqual(pool.failover_count(), 1)
        self.assertEqual(result["metrics"]["endpoint_failovers"],
                         pool.failover_count())
        self.assertGreater(spare.calls, 0)

    def test_trace_pack_records_the_reroute(self):
        runtime, _ = self._runtime_with(
            AnsweringBackend("gpu0", fail_first=99), AnsweringBackend("gpu1"),
        )
        result = runtime.run_envelope(_envelope("fo-2"))
        trace = Storage.read_json(result["trace_pack"]["path"])
        names = [e["name"] for e in trace["events"]]
        self.assertIn("endpoint_failover_policy", names)
        self.assertIn("endpoint_unhealthy", names)
        self.assertIn("endpoint_failover", names)
        reroute = next(e for e in trace["events"]
                       if e["name"] == "endpoint_failover")
        self.assertEqual(reroute["data"]["served_by"], "gpu1")

    def test_healthy_cluster_records_no_failover(self):
        runtime, pool = self._runtime_with(
            AnsweringBackend("gpu0"), AnsweringBackend("gpu1"),
        )
        result = runtime.run_envelope(_envelope("fo-3"))
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["metrics"]["endpoint_failovers"], 0)
        trace = Storage.read_json(result["trace_pack"]["path"])
        names = [e["name"] for e in trace["events"]]
        self.assertNotIn("endpoint_failover", names)
        self.assertNotIn("endpoint_unhealthy", names)

    def test_a_brain_role_reroute_reaches_the_trace_pack(self):
        # The planner call used to run before the trace sink was attached, so
        # a brain reroute incremented the metric while leaving no trace event
        # -- the two could not be reconciled.
        config = temp_config(tempfile.mkdtemp())
        config.manifest["ollama"] = dict(
            config.manifest.get("ollama", {}),
            endpoints=[
                {"name": "gpu0", "base_url": "http://localhost:11434",
                 "roles": ["brain"]},
                {"name": "gpu1", "base_url": "http://localhost:11435",
                 "roles": ["worker", "semantic_verifier"]},
            ],
        )
        runtime = Runtime(config, prefer_backend="offline")
        pool = EndpointPool(OllamaBackend("http://localhost:11434"), config)
        pool._backends["gpu0"] = AnsweringBackend("gpu0", fail_first=1)
        pool._backends["gpu1"] = AnsweringBackend("gpu1")
        runtime.endpoint_pool = pool

        result = runtime.run_envelope(_envelope("brain-fo", "B3"))

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["metrics"]["endpoint_failovers"], 1)
        trace = Storage.read_json(result["trace_pack"]["path"])
        reroutes = [e["data"] for e in trace["events"]
                    if e["name"] == "endpoint_failover"]
        self.assertEqual(len(reroutes), 1, "the metric and the trace disagree")
        self.assertEqual(reroutes[0]["role"], "brain")
        self.assertEqual(reroutes[0]["served_by"], "gpu1")
        self.assertIn("endpoint_unhealthy",
                      [e["name"] for e in trace["events"]])

    def test_failover_metric_is_per_run_not_cumulative(self):
        # Hosts are told to reuse one Runtime across tasks, and the health
        # ledger is deliberately session-scoped. The metric must still
        # describe THIS run, or "0 means not degraded" stops being true.
        gpu0 = AnsweringBackend("gpu0", fail_first=1)  # fails exactly once
        runtime, pool = self._runtime_with(gpu0, AnsweringBackend("gpu1"))

        per_run = [
            runtime.run_envelope(_envelope(f"reuse-{i}"))["metrics"][
                "endpoint_failovers"
            ]
            for i in range(4)
        ]
        # Every run is served by the spare (gpu0's circuit stays open), so
        # each run has exactly one reroute -- not one, two, three, four.
        self.assertEqual(per_run, [1, 1, 1, 1])
        # The pool's own tally does keep accumulating; that is the ledger.
        self.assertEqual(pool.failover_count(), 4)

    def test_a_healthy_reused_runtime_keeps_reporting_zero(self):
        runtime, _ = self._runtime_with(
            AnsweringBackend("gpu0"), AnsweringBackend("gpu1"),
        )
        for i in range(3):
            result = runtime.run_envelope(_envelope(f"clean-{i}"))
            self.assertEqual(result["metrics"]["endpoint_failovers"], 0)

    def test_whole_cluster_down_fails_the_run_with_a_host_visible_code(self):
        # Failover must not paper over a genuinely dead cluster -- but the
        # host gets a Result Envelope, not a traceback.
        runtime, _ = self._runtime_with(
            AnsweringBackend("gpu0", fail_first=99),
            AnsweringBackend("gpu1", fail_first=99),
        )
        result = runtime.run_envelope(_envelope("fo-4"))
        self.assertEqual(result["status"], "fail")
        self.assertIn(result["code"], status_codes.BACKEND_CODES)
        # The trace still lands, so the outage is diagnosable afterwards.
        trace = Storage.read_json(result["trace_pack"]["path"])
        names = [e["name"] for e in trace["events"]]
        self.assertIn("backend_failure", names)
        self.assertIn("endpoint_unhealthy", names)

    def test_dead_verifier_endpoint_reroutes_to_the_worker_gpu(self):
        # gpu1 serves only the semantic verifier. When it dies, `auto` should
        # borrow gpu0 rather than fail the verification step.
        config = temp_config(tempfile.mkdtemp())
        config.manifest["ollama"] = dict(
            config.manifest.get("ollama", {}), endpoints=_ENDPOINTS,
        )
        runtime = Runtime(config, prefer_backend="offline",
                          verifier_override="hybrid")
        gpu0 = AnsweringBackend("gpu0")
        gpu1 = AnsweringBackend("gpu1", fail_first=99)
        pool = EndpointPool(OllamaBackend("http://localhost:11434"), config)
        pool._backends["gpu0"], pool._backends["gpu1"] = gpu0, gpu1
        runtime.endpoint_pool = pool

        result = runtime.run_envelope(_envelope("fo-5", "B3"))

        self.assertEqual(result["status"], "pass")
        trace = Storage.read_json(result["trace_pack"]["path"])
        reroutes = [e["data"] for e in trace["events"]
                    if e["name"] == "endpoint_failover"]
        self.assertTrue(
            any(r["role"] == "semantic_verifier" and r["served_by"] == "gpu0"
                for r in reroutes),
            f"expected a semantic_verifier reroute onto gpu0, got {reroutes}",
        )


class HalfOpenGateTests(unittest.TestCase):
    """The post-cooldown trial is one request, not open season."""

    def test_only_one_caller_is_admitted_per_cooldown(self):
        clock = FakeClock()
        health = EndpointHealth(cooldown_seconds=10, clock=clock)
        health.record_failure("srv", OllamaError("down"))
        clock.advance(11)
        # First read claims the trial; the rest keep skipping the endpoint.
        self.assertEqual([health.is_open("srv") for _ in range(4)],
                         [False, True, True, True])

    def test_parallel_samples_do_not_all_pile_into_a_dead_endpoint(self):
        clock = FakeClock()
        health = EndpointHealth(cooldown_seconds=10, clock=clock)
        dead, spare = StubBackend("gpu0", OllamaError("down")), StubBackend("gpu1")
        backend = FailoverBackend("worker", _chain(dead, spare), health)
        backend.generate("first", model="m")          # discovers gpu0 is down
        self.assertEqual(dead.calls, 1)
        clock.advance(11)                             # cooldown elapses
        for _ in range(5):
            backend.generate("batch", model="m")
        # Exactly one trial got through, not five full timeouts.
        self.assertEqual(dead.calls, 2)

    def test_a_failed_trial_re_arms_the_cooldown(self):
        clock = FakeClock()
        health = EndpointHealth(cooldown_seconds=10, clock=clock)
        health.record_failure("srv", OllamaError("down"))
        clock.advance(11)
        self.assertFalse(health.is_open("srv"))       # trial claimed
        health.record_failure("srv", OllamaError("still down"))
        self.assertTrue(health.is_open("srv"))
        clock.advance(11)
        self.assertFalse(health.is_open("srv"))       # a fresh trial

    def test_a_successful_trial_closes_the_circuit(self):
        clock = FakeClock()
        health = EndpointHealth(cooldown_seconds=10, clock=clock)
        health.record_failure("srv", OllamaError("down"))
        clock.advance(11)
        health.is_open("srv")
        health.record_success("srv")
        self.assertEqual([health.is_open("srv") for _ in range(3)],
                         [False, False, False])


class LastResortOrderingTests(unittest.TestCase):
    """An open circuit demotes a candidate; it does not remove it."""

    def test_an_open_circuit_candidate_is_still_tried_when_all_else_fails(self):
        clock = FakeClock()
        health = EndpointHealth(cooldown_seconds=10_000, clock=clock)
        recovered = StubBackend("gpu0")                      # actually fine now
        dead = StubBackend("gpu1", OllamaError("down"))
        health.record_failure(recovered.base_url, OllamaError("earlier blip"))
        backend = FailoverBackend("worker", _chain(recovered, dead), health)
        # gpu1 is "healthy" per the ledger and tried first, but fails; gpu0's
        # circuit is open yet it is the only thing left, so it gets a shot.
        self.assertEqual(backend.generate("hi", model="m").text, "answer from gpu0")
        self.assertEqual(recovered.calls, 1)

    def test_generate_and_is_available_agree_on_the_chain(self):
        health = EndpointHealth(cooldown_seconds=10_000, clock=FakeClock())
        alive = StubBackend("gpu0")
        health.record_failure(alive.base_url, OllamaError("blip"))
        backend = FailoverBackend("worker", _chain(alive), health)
        self.assertTrue(backend.is_available())
        # is_available() said yes, so generate() must not refuse.
        self.assertEqual(backend.generate("hi", model="m").text, "answer from gpu0")

    def test_the_error_names_every_candidate_that_was_tried(self):
        health = EndpointHealth(cooldown_seconds=10_000, clock=FakeClock())
        a = StubBackend("gpu0", OllamaError("down"))
        b = StubBackend("gpu1", OllamaError("down"))
        health.record_failure(a.base_url, OllamaError("earlier"))
        backend = FailoverBackend("worker", _chain(a, b), health)
        with self.assertRaises(OllamaError) as ctx:
            backend.generate("hi", model="m")
        message = str(ctx.exception)
        self.assertIn("2 endpoint(s)", message)
        self.assertIn("gpu0", message)
        self.assertIn("gpu1", message)


class HealthIsTrackedPerServerTests(unittest.TestCase):
    """base_url identifies a server; a name is just a label for one."""

    def test_two_names_for_one_server_share_a_circuit(self):
        shared = StubBackend("localhost")
        a = Candidate("gpu0", shared, True)
        b = Candidate("default", shared, False)
        self.assertEqual(a.health_key, b.health_key)
        health = EndpointHealth(cooldown_seconds=1000, clock=FakeClock())
        health.record_failure(a.health_key, OllamaError("down"))
        # Reaching the same dead server under the other name must not look
        # healthy -- that is the whole point of routing around it.
        self.assertTrue(health.is_open(b.health_key))

    def test_the_aliased_default_endpoint_shares_the_pool_circuit(self):
        # The shipped example has ollama.base_url == gpu0.base_url, so an
        # unmapped role falls through to the same server under the name
        # "default".
        config = temp_config(tempfile.mkdtemp())
        config.manifest["ollama"] = dict(config.manifest.get("ollama", {}),
                                         endpoints=_ENDPOINTS)
        pool = EndpointPool(OllamaBackend("http://localhost:11434"), config)
        pool.health.record_failure("http://localhost:11434", OllamaError("down"))
        coder = pool.candidates_for_role("coder")
        self.assertEqual(coder[0].name, "default")
        self.assertTrue(pool.health.is_open(coder[0].health_key))
