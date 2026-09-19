"""v0.7.0: multi-GPU endpoint pool -- role routing + parallel sampling.

No GPUs or Ollama exist in CI, so these tests prove the *mechanics* with
stub backends: role -> endpoint resolution, backward compatibility with no
endpoints configured, genuine concurrency of parallel sampling (via a
threading.Barrier that deadlocks unless two drafts run simultaneously),
and per-role backend usage inside the Quality Factory.
"""

import contextlib
import io
import json
import tempfile
import threading
import unittest

import yaml

from tests.helpers import temp_config, write_temp_manifest

from heimdal.cli import main
from heimdal.core import quality_factory
from heimdal.core.repro_trace import TraceBuilder
from heimdal.core.role_binding import resolve_role
from heimdal.core.runtime import Runtime
from heimdal.core.task_contract import build_contract
from heimdal.models.base import GenerationResult, ModelBackend
from heimdal.models.endpoint_pool import EndpointPool, parse_endpoints
from heimdal.models.ollama import OllamaBackend
from heimdal.storage import Storage

_ENDPOINTS = [
    {"name": "gpu0", "base_url": "http://localhost:11434",
     "roles": ["worker", "brain"]},
    {"name": "gpu1", "base_url": "http://localhost:11435",
     "roles": ["semantic_verifier", "worker"]},
]


def _envelope(task_id: str, quality_level: str = "B1") -> dict:
    return {
        "host": {"type": "cli", "host_task_id": task_id,
                 "source_agent": None, "callback": {}},
        "role_binding": {"role_id": "general", "risk_mode": "balanced",
                         "privacy_mode": "local_only", "output_profiles": ["markdown"]},
        "task_request": {"task_id": task_id, "title": "EP demo",
                         "instruction": "Explain what a queue is and how it behaves.",
                         "inputs": {}, "constraints": {}, "priority": "P2",
                         "budget": {"quality_level": quality_level},
                         "expected_outputs": ["markdown_response"]},
        "runtime_hints": {},
    }


class RecorderBackend(ModelBackend):
    """Offline stub that records every generate() call's structured dict."""

    name = "recorder"

    def __init__(self, tag: str):
        self.tag = tag
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        return True

    def list_models(self) -> list[str]:
        return ["recorder-model"]

    def generate(self, prompt, *, model, system="", json_mode=False,
                 max_tokens=512, temperature=0.2, structured=None):
        s = structured or {}
        with self._lock:
            self.calls.append(dict(s, _tag=self.tag))
        if s.get("verify_task") == "semantic":
            text = json.dumps({"status": "pass", "score": 0.9,
                               "confidence": 0.8, "defects": [],
                               "rationale_short": "ok"})
        elif s.get("brain_task") == "plan":
            text = "1. Plan step one.\n2. Plan step two."
        else:
            text = ("A queue is a first-in, first-out structure where items "
                    "are added at the back and removed from the front, which "
                    "keeps processing in arrival order.")
        return GenerationResult(text=text, model=model, backend=self.name)


class ParseEndpointsTests(unittest.TestCase):
    def test_absent_endpoints_parse_to_empty(self):
        self.assertEqual(parse_endpoints({"base_url": "http://x"}), [])
        self.assertEqual(parse_endpoints({"endpoints": []}), [])

    def test_well_formed_endpoints_parse(self):
        parsed = parse_endpoints({"endpoints": _ENDPOINTS})
        self.assertEqual([e.name for e in parsed], ["gpu0", "gpu1"])
        self.assertEqual(parsed[0].roles, ["worker", "brain"])

    def test_malformed_entries_are_skipped(self):
        parsed = parse_endpoints({"endpoints": [
            {"name": "no-url"}, "not-a-dict",
            {"base_url": "http://ok:11434"},
        ]})
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].base_url, "http://ok:11434")


class EndpointPoolTests(unittest.TestCase):
    def _pool(self, endpoints=None) -> EndpointPool:
        config = temp_config(tempfile.mkdtemp())
        config.manifest["ollama"] = dict(
            config.manifest.get("ollama", {}), endpoints=endpoints or [],
        )
        default = OllamaBackend("http://localhost:11434")
        return EndpointPool(default, config)

    def test_no_endpoints_resolves_every_role_to_default(self):
        pool = self._pool()
        for role in ("worker", "semantic_verifier", "brain", "coder"):
            self.assertIs(pool.backend_for_role(role), pool.default_backend)
        self.assertEqual(pool.worker_backends(), [pool.default_backend])
        self.assertFalse(pool.has_multiple_worker_endpoints())
        self.assertEqual(
            set(pool.routing_map().values()), {"default"},
        )

    def test_roles_route_to_their_endpoints(self):
        pool = self._pool(_ENDPOINTS)
        self.assertEqual(pool.backend_for_role("worker").base_url,
                         "http://localhost:11434")
        self.assertEqual(pool.backend_for_role("semantic_verifier").base_url,
                         "http://localhost:11435")
        # brain shares gpu0; coder is unmapped, so its first choice is the
        # default backend (with the mapped endpoints behind it as spares).
        self.assertEqual(pool.backend_for_role("brain").base_url,
                         "http://localhost:11434")
        self.assertIs(pool.backend_for_role("coder").primary,
                      pool.default_backend)
        self.assertEqual(pool.routing_map()["semantic_verifier"], "gpu1")

    def test_multiple_worker_endpoints_detected(self):
        pool = self._pool(_ENDPOINTS)
        self.assertTrue(pool.has_multiple_worker_endpoints())
        self.assertEqual(len(pool.worker_backends()), 2)

    def test_backends_are_cached_per_endpoint(self):
        pool = self._pool(_ENDPOINTS)
        # worker and brain both map to gpu0. Each role gets its own failover
        # wrapper (their fallback chains differ), but the wrappers share the
        # one cached backend for that endpoint.
        self.assertIs(pool.backend_for_role("worker").primary,
                      pool.backend_for_role("brain").primary)

    def test_role_wrappers_are_cached(self):
        pool = self._pool(_ENDPOINTS)
        self.assertIs(pool.backend_for_role("worker"),
                      pool.backend_for_role("worker"))

    def test_offline_default_backend_ignores_endpoint_config(self):
        from heimdal.models.offline import OfflineBackend
        config = temp_config(tempfile.mkdtemp())
        config.manifest["ollama"] = dict(
            config.manifest.get("ollama", {}), endpoints=_ENDPOINTS,
        )
        pool = EndpointPool(OfflineBackend(), config)
        self.assertIs(pool.backend_for_role("worker"), pool.default_backend)

    def test_parallel_samples_flag_semantics(self):
        config = temp_config(tempfile.mkdtemp())
        pool_single = self._pool()
        pool_multi = self._pool(_ENDPOINTS)
        # auto (default): only with >1 worker endpoint.
        config.manifest["concurrency"] = {"parallel_samples": "auto"}
        self.assertFalse(pool_single.parallel_samples_enabled(config))
        self.assertTrue(pool_multi.parallel_samples_enabled(config))
        # explicit true / false override.
        config.manifest["concurrency"] = {"parallel_samples": True}
        self.assertTrue(pool_single.parallel_samples_enabled(config))
        config.manifest["concurrency"] = {"parallel_samples": False}
        self.assertFalse(pool_multi.parallel_samples_enabled(config))


class FanOutRouterEndpointTests(unittest.TestCase):
    """A single endpoint that fronts several machines (e.g. NVIDIA PAIR).

    Only one base_url is configured, so counting distinct endpoints would
    wrongly report no room for concurrency; the slot count is what says
    how wide the router can be driven.
    """

    _PAIR = [{
        "name": "pair",
        "base_url": "http://127.0.0.1:11434",
        "roles": ["worker", "brain"],
        "slots": 3,
    }]

    def _pool(self, endpoints) -> EndpointPool:
        config = temp_config(tempfile.mkdtemp())
        config.manifest["ollama"] = dict(
            config.manifest.get("ollama", {}), endpoints=endpoints,
        )
        return EndpointPool(OllamaBackend("http://localhost:11434"), config), config

    def test_slots_parsed_and_defaulted(self):
        parsed = parse_endpoints({"endpoints": [
            {"name": "plain", "base_url": "http://a:11434"},
            {"name": "router", "base_url": "http://b:11434", "slots": 4},
            {"name": "junk", "base_url": "http://c:11434", "slots": "many"},
            {"name": "zero", "base_url": "http://d:11434", "slots": 0},
        ]})
        self.assertEqual([e.slots for e in parsed], [1, 4, 1, 1])

    def test_single_router_endpoint_enables_auto_parallel(self):
        pool, config = self._pool(self._PAIR)
        config.manifest["concurrency"] = {"parallel_samples": "auto"}
        self.assertEqual(pool.worker_slots(), 3)
        self.assertTrue(pool.parallel_samples_enabled(config))
        # One distinct endpoint, so the old distinct-endpoint test is False;
        # the slot count is what carries the concurrency signal now.
        self.assertFalse(pool.has_multiple_worker_endpoints())

    def test_worker_backends_repeat_per_slot_onto_one_backend(self):
        pool, _ = self._pool(self._PAIR)
        backends = pool.worker_backends()
        self.assertEqual(len(backends), 3)
        self.assertEqual({id(b) for b in backends}, {id(backends[0])})
        self.assertEqual(backends[0].base_url, "http://127.0.0.1:11434")

    def test_status_reports_slots(self):
        pool, _ = self._pool(self._PAIR)
        pool._backend_for_endpoint(pool._endpoints[0]).is_available = lambda: False
        self.assertEqual(pool.status()[0]["slots"], 3)

    def test_default_slots_preserve_pre_existing_behavior(self):
        pool, config = self._pool([{
            "name": "gpu0", "base_url": "http://localhost:11434",
            "roles": ["worker"],
        }])
        config.manifest["concurrency"] = {"parallel_samples": "auto"}
        self.assertEqual(pool.worker_slots(), 1)
        self.assertFalse(pool.parallel_samples_enabled(config))
        self.assertEqual(len(pool.worker_backends()), 1)


class QualityFactoryRoutingTests(unittest.TestCase):
    """Per-role backends are actually used inside the pipeline."""

    class FakePool:
        def __init__(self, worker, verifier_b, brain):
            self._worker, self._verifier, self._brain = worker, verifier_b, brain

        def worker_backends(self):
            return [self._worker]

        def backend_for_role(self, role):
            return {"semantic_verifier": self._verifier,
                    "brain": self._brain}.get(role, self._worker)

        def routing_map(self):
            return {"worker": "gpu0", "semantic_verifier": "gpu1",
                    "brain": "gpu0", "coder": "default"}

        failover_mode = "auto"

        def failover_map(self):
            return {"worker": ["gpu0"], "semantic_verifier": ["gpu1"],
                    "brain": ["gpu0"], "coder": ["default"]}

        def parallel_samples_enabled(self, config=None):
            return False

        def has_multiple_worker_endpoints(self):
            return False

    def test_semantic_verifier_and_brain_use_their_role_backends(self):
        config = temp_config(tempfile.mkdtemp())
        storage = Storage(config.storage_root).ensure()
        role = resolve_role({"role_id": "general"})
        envelope = _envelope("route-1", "B3")  # B3: brain step + hybrid-eligible
        contract = build_contract(envelope, role, config)
        trace = TraceBuilder(contract["task_id"])

        worker = RecorderBackend("worker")
        verifier_b = RecorderBackend("verifier")
        brain = RecorderBackend("brain")
        outcome = quality_factory.run_quality_factory(
            contract, role, envelope, worker, storage, config, trace,
            verifier_override="hybrid",
            backend_pool=self.FakePool(worker, verifier_b, brain),
        )
        self.assertEqual(outcome["status"], "pass")
        # Semantic verification ran on the verifier-role backend only.
        self.assertTrue(any(c.get("verify_task") == "semantic"
                            for c in verifier_b.calls))
        self.assertFalse(any(c.get("verify_task") == "semantic"
                             for c in worker.calls))
        # The brain plan ran on the brain-role backend.
        self.assertTrue(any(c.get("brain_task") == "plan" for c in brain.calls))
        # Worker drafts stayed on the worker backend.
        self.assertTrue(any("instruction" in c and "defects" in c
                            for c in worker.calls))
        # The routing decision is recorded in the trace.
        names = [e["name"] for e in trace.events]
        self.assertIn("endpoint_routing", names)


class ParallelSamplingTests(unittest.TestCase):
    class BarrierBackend(RecorderBackend):
        """Worker drafts block on a 2-party barrier: the test only passes if
        two drafts are in flight AT THE SAME TIME. Serial execution would
        block forever on the first draft (BrokenBarrierError after timeout)."""

        def __init__(self):
            super().__init__("barrier")
            self.barrier = threading.Barrier(2, timeout=10)

        def generate(self, prompt, *, model, system="", json_mode=False,
                     max_tokens=512, temperature=0.2, structured=None):
            s = structured or {}
            if "defects" in s and s.get("brain_task") is None \
                    and s.get("verify_task") is None:
                self.barrier.wait()  # worker draft: rendezvous or die
            return super().generate(
                prompt, model=model, system=system, json_mode=json_mode,
                max_tokens=max_tokens, temperature=temperature, structured=s,
            )

    def test_b3_samples_run_concurrently_when_enabled(self):
        config = temp_config(tempfile.mkdtemp())
        config.manifest["concurrency"] = {"parallel_samples": True}
        runtime = Runtime(config, prefer_backend="offline")
        backend = self.BarrierBackend()
        runtime.backend = backend
        runtime.endpoint_pool = EndpointPool(backend, config)
        result = runtime.run_envelope(_envelope("par-1", "B3"))  # samples=2
        self.assertEqual(result["status"], "pass")
        trace = Storage.read_json(result["trace_pack"]["path"])
        parallel_event = next(
            e for e in trace["events"] if e["name"] == "parallel_samples"
        )
        self.assertEqual(parallel_event["data"]["count"], 2)
        # One offline backend stands in for the pool here: one distinct
        # endpoint, one slot.
        self.assertEqual(parallel_event["data"]["worker_endpoints"], 1)
        self.assertEqual(parallel_event["data"]["worker_slots"], 1)

    def test_serial_remains_the_default_without_endpoints(self):
        config = temp_config(tempfile.mkdtemp())  # parallel_samples: auto
        runtime = Runtime(config, prefer_backend="offline")
        result = runtime.run_envelope(_envelope("ser-1", "B3"))
        trace = Storage.read_json(result["trace_pack"]["path"])
        names = [e["name"] for e in trace["events"]]
        self.assertNotIn("parallel_samples", names)

    def test_metrics_include_endpoint_routing(self):
        config = temp_config(tempfile.mkdtemp())
        runtime = Runtime(config, prefer_backend="offline")
        result = runtime.run_envelope(_envelope("met-1", "B1"))
        self.assertIn("endpoint_routing", result["metrics"])
        self.assertEqual(
            set(result["metrics"]["endpoint_routing"].values()), {"default"},
        )


class EndpointsCLITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.manifest = write_temp_manifest(self.tmp, self.tmp)

    def _add_endpoints(self, endpoints) -> None:
        with open(self.manifest, "r", encoding="utf-8") as fh:
            manifest = yaml.safe_load(fh)
        manifest.setdefault("ollama", {})["endpoints"] = endpoints
        with open(self.manifest, "w", encoding="utf-8") as fh:
            yaml.safe_dump(manifest, fh)

    def test_list_without_endpoints_explains_default(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["endpoints", "list", "--manifest", self.manifest])
        self.assertEqual(code, 0)
        self.assertIn("No endpoints configured", buf.getvalue())

    def test_list_shows_configured_endpoints(self):
        self._add_endpoints(_ENDPOINTS)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["endpoints", "list", "--json", "--manifest", self.manifest])
        self.assertEqual(code, 0)
        data = json.loads(buf.getvalue())
        self.assertEqual([e["name"] for e in data["endpoints"]], ["gpu0", "gpu1"])
        self.assertEqual(data["failover"], "auto")
        # gpu1 serves the verifier, and gpu0 backs it up.
        self.assertEqual(data["role_candidates"]["semantic_verifier"],
                         ["gpu1", "gpu0"])

    def test_status_flags_unreachable_endpoint(self):
        # Port 9 (discard) refuses fast; status must mark it DOWN and exit 1.
        self._add_endpoints([
            {"name": "dead", "base_url": "http://127.0.0.1:9",
             "roles": ["worker"]},
        ])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["endpoints", "status", "--manifest", self.manifest])
        self.assertEqual(code, 1)
        self.assertIn("DOWN", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
