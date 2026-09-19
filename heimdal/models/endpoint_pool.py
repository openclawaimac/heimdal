"""Multi-GPU endpoint pool (v0.7.0).

Routes internal roles (worker / semantic_verifier / brain / coder) to
separate Ollama endpoints so a multi-GPU machine can pin one Ollama
instance per GPU and run Heimdal's roles on different devices:

    # GPU 0 -- worker + brain
    CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=127.0.0.1:11434 ollama serve
    # GPU 1 -- semantic verifier
    CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=127.0.0.1:11435 ollama serve

Manifest shape (``ollama.endpoints`` -- optional; absent means everything
runs on the single default endpoint exactly as before v0.7.0)::

    ollama:
      base_url: http://localhost:11434
      endpoints:
        - name: gpu0
          base_url: http://localhost:11434
          roles: [worker, brain]
        - name: gpu1
          base_url: http://localhost:11435
          roles: [semantic_verifier]

A single endpoint may itself be a fan-out router backed by several
machines -- NVIDIA PAIR, for example, presents one Ollama-compatible
proxy that spreads independent requests over every paired node. Such an
endpoint declares how many requests it can absorb at once via ``slots``,
so Heimdal knows to issue concurrent samples even though only one
base_url is configured::

    ollama:
      endpoints:
        - name: pair
          base_url: http://127.0.0.1:11434
          roles: [worker, brain, semantic_verifier]
          slots: 3          # three paired nodes behind the router

Scope guard: this is *role*-level routing across whole Ollama instances.
It is NOT tensor/model parallelism (splitting one model across GPUs --
that is Ollama/llama.cpp's job) and NOT a distributed cluster.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from heimdal.models.base import ModelBackend
from heimdal.models.ollama import OllamaBackend

# Roles the pool understands. Anything else falls through to the default.
POOL_ROLES = ("worker", "semantic_verifier", "brain", "coder")


@dataclass
class Endpoint:
    name: str
    base_url: str
    roles: list[str] = field(default_factory=list)
    # Concurrent requests this endpoint can absorb. 1 for a plain Ollama
    # instance; higher for a fan-out router such as NVIDIA PAIR.
    slots: int = 1


def _parse_slots(raw_value) -> int:
    try:
        slots = int(raw_value)
    except (TypeError, ValueError):
        return 1
    return max(1, slots)


def parse_endpoints(ollama_cfg: dict) -> list[Endpoint]:
    """Read ``ollama.endpoints`` from the manifest; empty when absent."""
    out: list[Endpoint] = []
    for index, raw in enumerate(ollama_cfg.get("endpoints") or []):
        if not isinstance(raw, dict) or not raw.get("base_url"):
            continue
        out.append(Endpoint(
            name=str(raw.get("name") or f"endpoint{index}"),
            base_url=str(raw["base_url"]).rstrip("/"),
            roles=[str(r) for r in (raw.get("roles") or [])],
            slots=_parse_slots(raw.get("slots", 1)),
        ))
    return out


class EndpointPool:
    """Resolves a role to a backend instance.

    With no ``ollama.endpoints`` configured -- or when the session runs on
    the offline backend -- every role resolves to the single default
    backend, which is byte-for-byte the pre-v0.7.0 behavior. Backends are
    constructed once per endpoint and cached, so repeated lookups are cheap
    and each endpoint keeps its own retry/timeout state.
    """

    def __init__(self, default_backend: ModelBackend, config=None):
        self.default_backend = default_backend
        self._config = config
        self._endpoints: list[Endpoint] = []
        self._backends: dict[str, ModelBackend] = {}
        # Endpoint routing only makes sense for the Ollama backend; the
        # offline backend is a single in-process stub.
        if config is not None and default_backend.name == "ollama":
            self._endpoints = parse_endpoints(config.ollama)

    # -- construction -------------------------------------------------------
    def _backend_for_endpoint(self, endpoint: Endpoint) -> ModelBackend:
        if endpoint.name not in self._backends:
            ollama = self._config.ollama if self._config else {}
            self._backends[endpoint.name] = OllamaBackend(
                base_url=endpoint.base_url,
                timeout=ollama.get("timeout_seconds", 120),
                max_retries=ollama.get("max_retries", 2),
                retry_backoff_seconds=ollama.get("retry_backoff_seconds", 1.0),
            )
        return self._backends[endpoint.name]

    # -- lookup --------------------------------------------------------------
    def endpoints_for_role(self, role: str) -> list[Endpoint]:
        return [e for e in self._endpoints if role in e.roles]

    def backend_for_role(self, role: str) -> ModelBackend:
        """The backend a role should use; the default when none is mapped."""
        matches = self.endpoints_for_role(role)
        if not matches:
            return self.default_backend
        return self._backend_for_endpoint(matches[0])

    def worker_backends(self) -> list[ModelBackend]:
        """One entry per concurrent worker slot; the default when none are
        mapped.

        An endpoint with ``slots: n`` appears n times, so the caller's
        round-robin over this list issues n concurrent requests to it. That
        is what lets a fan-out router in front of several machines be
        driven at full width from a single configured base_url.
        """
        matches = self.endpoints_for_role("worker")
        if not matches:
            return [self.default_backend]
        out: list[ModelBackend] = []
        for endpoint in matches:
            backend = self._backend_for_endpoint(endpoint)
            out.extend([backend] * endpoint.slots)
        return out

    def worker_slots(self) -> int:
        """Total concurrent worker requests the pool can sustain."""
        matches = self.endpoints_for_role("worker")
        if not matches:
            return 1
        return sum(e.slots for e in matches)

    def endpoint_name_for_role(self, role: str) -> str:
        matches = self.endpoints_for_role(role)
        return matches[0].name if matches else "default"

    def routing_map(self) -> dict:
        """role -> endpoint name, for the Trace Pack's endpoint_routing event."""
        return {role: self.endpoint_name_for_role(role) for role in POOL_ROLES}

    def has_multiple_worker_endpoints(self) -> bool:
        return len(self.endpoints_for_role("worker")) > 1

    # -- concurrency policy ---------------------------------------------------
    def parallel_samples_enabled(self, config=None) -> bool:
        """Whether B3/B4 multi-sample drafting should run concurrently.

        ``concurrency.parallel_samples`` in the manifest:
          - true  -> always parallel when samples > 1 (works offline; used
                     by CI to exercise the path)
          - false -> never
          - "auto" (default) -> parallel only when the worker role has more
                     than one concurrent slot, i.e. there is real hardware
                     to spread the samples across. Several single-slot
                     endpoints and one multi-slot router endpoint both
                     qualify.
        """
        cfg = (config or self._config)
        flag = "auto"
        if cfg is not None:
            flag = cfg.manifest.get("concurrency", {}).get("parallel_samples", "auto")
        if flag is True:
            return True
        if flag is False:
            return False
        return self.worker_slots() > 1

    # -- health ----------------------------------------------------------------
    def status(self) -> list[dict]:
        """Reachability of every configured endpoint (pings each one)."""
        out: list[dict] = []
        for endpoint in self._endpoints:
            backend = self._backend_for_endpoint(endpoint)
            reachable = backend.is_available()
            out.append({
                "name": endpoint.name,
                "base_url": endpoint.base_url,
                "roles": endpoint.roles,
                "slots": endpoint.slots,
                "reachable": reachable,
                "models": backend.list_models() if reachable else [],
            })
        return out
