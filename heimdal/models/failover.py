"""Health-aware endpoint failover (v0.7.1).

Endpoint routing (``heimdal.models.endpoint_pool``) pins roles to Ollama
instances, but a pinned endpoint that dies takes the whole run with it:
the semantic verifier mapped to GPU 1 fails outright even while GPU 0 sits
idle and able to serve it. This module adds the missing degraded mode.

Two pieces:

``EndpointHealth``
    A session-scoped circuit breaker. An endpoint that raises is marked
    down and skipped by later requests until a cooldown elapses, at which
    point one trial request is allowed through (half-open). The ledger is
    shared by every role and is thread-safe, because B3/B4 parallel
    sampling drives several endpoints at once.

``FailoverBackend``
    Wraps an ordered list of candidate endpoints for one role and tries
    them in turn. The first healthy candidate that answers wins; failures
    are recorded and the next candidate is tried.

Both are transparent to the pipeline: ``FailoverBackend`` is a
``ModelBackend``, so the Quality Factory keeps calling ``generate()``
without knowing an endpoint moved under it. Every fallback is traced, so
a degraded run is visible after the fact rather than silently slower.

Scope guard: this reroutes *whole requests* between endpoints. It does not
retry inside an endpoint (``OllamaBackend`` already does that), and it
never resumes a partially generated response elsewhere.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from heimdal.models.base import GenerationResult, ModelBackend
from heimdal.models.ollama import OllamaError

# One OllamaError already means the backend exhausted its own retries, so a
# single failure is enough to take an endpoint out of rotation.
DEFAULT_FAILURE_THRESHOLD = 1
# How long a downed endpoint stays out before one trial request is allowed.
DEFAULT_COOLDOWN_SECONDS = 60.0

# Failures worth rerouting on: the endpoint is unreachable, timed out, or
# does not have the model. A different endpoint may well satisfy any of
# them. Errors raised anywhere else (bad prompt, bug) are not caught, so
# they surface instead of being masked by a retry storm.
FAILOVER_ERRORS = (OllamaError, OSError)


@dataclass
class _EndpointState:
    consecutive_failures: int = 0
    total_failures: int = 0
    total_successes: int = 0
    opened_at: float | None = None
    last_error: str = ""


@dataclass
class Candidate:
    """One endpoint a role may be served by, in preference order."""

    name: str
    backend: ModelBackend
    # True when this candidate is mapped to the role in the manifest, False
    # when it is a spare borrowed from elsewhere in the pool.
    preferred: bool = True


class EndpointHealth:
    """Thread-safe circuit breaker ledger, shared across roles."""

    def __init__(
        self,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        clock=time.monotonic,
    ):
        self.failure_threshold = max(1, failure_threshold)
        self.cooldown_seconds = max(0.0, cooldown_seconds)
        self._clock = clock
        self._lock = threading.Lock()
        self._states: dict[str, _EndpointState] = {}
        self.failover_count = 0

    def _state(self, name: str) -> _EndpointState:
        state = self._states.get(name)
        if state is None:
            state = self._states[name] = _EndpointState()
        return state

    def record_success(self, name: str) -> None:
        with self._lock:
            state = self._state(name)
            state.consecutive_failures = 0
            state.opened_at = None
            state.total_successes += 1

    def record_failure(self, name: str, error: BaseException) -> None:
        with self._lock:
            state = self._state(name)
            state.consecutive_failures += 1
            state.total_failures += 1
            state.last_error = str(error)[:300]
            if state.consecutive_failures >= self.failure_threshold:
                state.opened_at = self._clock()

    def record_failover(self) -> None:
        with self._lock:
            self.failover_count += 1

    def is_open(self, name: str) -> bool:
        """True when ``name`` is currently out of rotation.

        The circuit re-closes optimistically once the cooldown elapses; the
        next request is the trial that decides whether it opens again.
        """
        with self._lock:
            state = self._states.get(name)
            if state is None or state.opened_at is None:
                return False
            if self._clock() - state.opened_at >= self.cooldown_seconds:
                state.opened_at = None
                state.consecutive_failures = 0
                return False
            return True

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "name": name,
                    "healthy": state.opened_at is None,
                    "successes": state.total_successes,
                    "failures": state.total_failures,
                    "last_error": state.last_error,
                }
                for name, state in sorted(self._states.items())
            ]


class FailoverBackend(ModelBackend):
    """Serves one role from the first candidate endpoint that answers."""

    def __init__(self, role: str, candidates: list[Candidate], health: EndpointHealth):
        if not candidates:
            raise ValueError("FailoverBackend needs at least one candidate")
        self.role = role
        self.candidates = candidates
        self.health = health
        # Present as the underlying backend so metrics and Repro Packs keep
        # reporting the real backend rather than this wrapper.
        self.name = candidates[0].backend.name
        self._event_sink = None

    # -- delegation ----------------------------------------------------------
    @property
    def primary(self) -> ModelBackend:
        """The first-choice backend, ignoring current health."""
        return self.candidates[0].backend

    @property
    def base_url(self) -> str:
        """The first-choice endpoint's URL, for introspection and logging."""
        return getattr(self.primary, "base_url", "")

    # -- event plumbing -----------------------------------------------------
    # The Quality Factory assigns ``event_sink`` to route backend events into
    # the Trace Pack. Fan it out so wrapped endpoints keep reporting.
    @property
    def event_sink(self):
        return self._event_sink

    @event_sink.setter
    def event_sink(self, sink) -> None:
        self._event_sink = sink
        for candidate in self.candidates:
            candidate.backend.event_sink = sink

    # -- candidate ordering --------------------------------------------------
    def _ordered(self) -> tuple[list[Candidate], list[Candidate]]:
        """Split candidates into healthy ones and those with an open circuit."""
        healthy, down = [], []
        for candidate in self.candidates:
            (down if self.health.is_open(candidate.name) else healthy).append(candidate)
        return healthy, down

    def _attempt_order(self) -> list[Candidate]:
        healthy, down = self._ordered()
        # Every endpoint is marked down: try them all anyway rather than
        # refusing outright. The cooldown may simply not have elapsed, and a
        # long-shot attempt beats a guaranteed failure.
        return healthy + down

    # -- interface -----------------------------------------------------------
    def is_available(self) -> bool:
        return any(c.backend.is_available() for c in self._attempt_order())

    def list_models(self) -> list[str]:
        for candidate in self._attempt_order():
            models = candidate.backend.list_models()
            if models:
                return models
        return []

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        healthy, skipped = self._ordered()
        # Everything is marked down: attempt it all anyway, in preference
        # order, rather than refusing outright.
        order = healthy or skipped
        skipped_names = [c.name for c in skipped] if healthy else []
        primary = self.candidates[0].name

        failed: list[str] = []
        last_error: BaseException | None = None
        for candidate in order:
            try:
                result = candidate.backend.generate(prompt, **kwargs)
            except FAILOVER_ERRORS as exc:
                last_error = exc
                failed.append(candidate.name)
                self.health.record_failure(candidate.name, exc)
                self._emit(
                    "endpoint_unhealthy",
                    role=self.role,
                    endpoint=candidate.name,
                    error_type=type(exc).__name__,
                    error=str(exc)[:200],
                    remaining_candidates=len(order) - len(failed),
                )
                continue
            self.health.record_success(candidate.name)
            # Anything other than the configured first choice is a fallback,
            # including the case where the primary was skipped outright
            # because its circuit was still open from an earlier request.
            if candidate.name != primary:
                self.health.record_failover()
                self._emit(
                    "endpoint_failover",
                    role=self.role,
                    failed=failed,
                    skipped=skipped_names,
                    served_by=candidate.name,
                    borrowed=not candidate.preferred,
                )
            return result

        raise OllamaError(
            f"All {len(order)} endpoint(s) for role '{self.role}' failed "
            f"({', '.join(c.name for c in order)}). Last error: {last_error}"
        )
