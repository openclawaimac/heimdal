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

Pinning a role to one endpoint makes that endpoint a single point of
failure, so roles fail over to other endpoints when theirs stops
answering (``heimdal.models.failover``). ``ollama.failover`` picks the
policy::

    ollama:
      failover: auto        # auto (default) | strict | off
      failover_cooldown_seconds: 60

``auto`` falls back first to any other endpoint mapped to the same role,
then to the rest of the pool as spares. ``strict`` stays within the
role's own endpoints. ``off`` restores the pre-v0.7.1 behavior, where a
downed endpoint fails the run.

Scope guard: this is *role*-level routing across whole Ollama instances.
It is NOT tensor/model parallelism (splitting one model across GPUs --
that is Ollama/llama.cpp's job) and NOT a distributed cluster.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from heimdal.models.base import ModelBackend
from heimdal.models.failover import (
    DEFAULT_COOLDOWN_SECONDS,
    Candidate,
    EndpointHealth,
    FailoverBackend,
)
from heimdal.models.ollama import OllamaBackend

# Roles the pool understands. Anything else falls through to the default.
POOL_ROLES = ("worker", "semantic_verifier", "brain", "coder")

# Name reported for the unconfigured fallback backend (``ollama.base_url``).
DEFAULT_ENDPOINT_NAME = "default"

FAILOVER_MODES = ("auto", "strict", "off")


def _dedupe(candidates: list[Candidate]) -> list[Candidate]:
    """Drop candidates pointing at a URL an earlier candidate already covers.

    Two endpoint names may share a base_url; retrying the same server after
    it just failed buys nothing.
    """
    out: list[Candidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = getattr(candidate.backend, "base_url", None) or candidate.name
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def _parse_failover_mode(raw_value) -> str:
    """Read ``ollama.failover``; anything unrecognized means the default."""
    if raw_value is False:
        return "off"
    if raw_value is True:
        return "auto"
    mode = str(raw_value).strip().lower()
    return mode if mode in FAILOVER_MODES else "auto"


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

    When more than one endpoint could serve a role, the pool hands back a
    :class:`FailoverBackend` instead of a bare backend so a dead endpoint
    degrades the run rather than ending it (see ``ollama.failover``).
    """

    def __init__(self, default_backend: ModelBackend, config=None):
        self.default_backend = default_backend
        self._config = config
        self._endpoints: list[Endpoint] = []
        self._backends: dict[str, ModelBackend] = {}
        self._role_backends: dict[str, ModelBackend] = {}
        # Endpoint routing only makes sense for the Ollama backend; the
        # offline backend is a single in-process stub.
        if config is not None and default_backend.name == "ollama":
            self._endpoints = parse_endpoints(config.ollama)
        ollama_cfg = config.ollama if config is not None else {}
        self.failover_mode = _parse_failover_mode(ollama_cfg.get("failover", "auto"))
        self.health = EndpointHealth(
            cooldown_seconds=float(
                ollama_cfg.get("failover_cooldown_seconds", DEFAULT_COOLDOWN_SECONDS)
            ),
        )

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

    def _default_candidate(self) -> Candidate:
        return Candidate(DEFAULT_ENDPOINT_NAME, self.default_backend, preferred=True)

    def candidates_for_role(self, role: str, primary: Endpoint | None = None) -> list[Candidate]:
        """Endpoints that may serve ``role``, best first.

        Preferred candidates are those the manifest maps to the role (or the
        default backend when it maps none). Under ``failover: auto`` the
        remaining endpoints follow as spares -- running the semantic
        verifier on the worker's GPU is worse than running it on its own,
        but far better than failing the run.
        """
        mapped = self.endpoints_for_role(role)
        if primary is not None:
            mapped = [primary] + [e for e in mapped if e.name != primary.name]

        preferred = [
            Candidate(e.name, self._backend_for_endpoint(e), preferred=True)
            for e in mapped
        ] or [self._default_candidate()]

        if self.failover_mode != "auto":
            return _dedupe(preferred[:1] if self.failover_mode == "off" else preferred)

        spares: list[Candidate] = []
        taken = {c.name for c in preferred}
        for endpoint in self._endpoints:
            if endpoint.name not in taken:
                spares.append(Candidate(
                    endpoint.name, self._backend_for_endpoint(endpoint), preferred=False,
                ))
                taken.add(endpoint.name)
        if DEFAULT_ENDPOINT_NAME not in taken and not self._serves_default_url():
            spares.append(Candidate(
                DEFAULT_ENDPOINT_NAME, self.default_backend, preferred=False,
            ))
        return _dedupe(preferred + spares)

    def _serves_default_url(self) -> bool:
        """True when a configured endpoint already points at ``base_url``."""
        default_url = getattr(self.default_backend, "base_url", None)
        if default_url is None:
            return True
        return any(e.base_url == default_url.rstrip("/") for e in self._endpoints)

    def _wrap(self, role: str, candidates: list[Candidate]) -> ModelBackend:
        """A bare backend when there is nothing to fall back to."""
        if len(candidates) == 1:
            return candidates[0].backend
        return FailoverBackend(role, candidates, self.health)

    def backend_for_role(self, role: str) -> ModelBackend:
        """The backend a role should use; the default when none is mapped."""
        if role not in self._role_backends:
            self._role_backends[role] = self._wrap(
                role, self.candidates_for_role(role),
            )
        return self._role_backends[role]

    def worker_backends(self) -> list[ModelBackend]:
        """One entry per concurrent worker slot; the default when none are
        mapped.

        An endpoint with ``slots: n`` appears n times, so the caller's
        round-robin over this list issues n concurrent requests to it. That
        is what lets a fan-out router in front of several machines be
        driven at full width from a single configured base_url. Each slot
        keeps its own endpoint as first choice and falls back to the rest.
        """
        matches = self.endpoints_for_role("worker")
        if not matches:
            return [self.backend_for_role("worker")]
        out: list[ModelBackend] = []
        for endpoint in matches:
            key = f"worker@{endpoint.name}"
            if key not in self._role_backends:
                self._role_backends[key] = self._wrap(
                    "worker", self.candidates_for_role("worker", primary=endpoint),
                )
            out.extend([self._role_backends[key]] * endpoint.slots)
        return out

    def worker_slots(self) -> int:
        """Total concurrent worker requests the pool can sustain."""
        matches = self.endpoints_for_role("worker")
        if not matches:
            return 1
        return sum(e.slots for e in matches)

    def endpoint_name_for_role(self, role: str) -> str:
        matches = self.endpoints_for_role(role)
        return matches[0].name if matches else DEFAULT_ENDPOINT_NAME

    def routing_map(self) -> dict:
        """role -> endpoint name, for the Trace Pack's endpoint_routing event."""
        return {role: self.endpoint_name_for_role(role) for role in POOL_ROLES}

    def failover_map(self) -> dict:
        """role -> ordered candidate names, for the Trace Pack."""
        return {
            role: [c.name for c in self.candidates_for_role(role)]
            for role in POOL_ROLES
        }

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
    def failover_count(self) -> int:
        """How many requests this session served from a fallback endpoint."""
        return self.health.failover_count

    def health_snapshot(self) -> list[dict]:
        """Observed success/failure per endpoint; empty before any request."""
        return self.health.snapshot()

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
                "circuit_open": self.health.is_open(endpoint.name),
                "models": backend.list_models() if reachable else [],
            })
        return out
