"""Ollama model backend.

Talks to an Ollama server over HTTP using the standard library only. The base
URL, timeout, and retry policy are configurable (manifest / OLLAMA_HOST).
Errors are surfaced as :class:`OllamaError` with actionable messages, and the
request lifecycle is reported through the backend ``event_sink``.
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request

from heimdal.models.base import GenerationResult, ModelBackend


class OllamaError(RuntimeError):
    """Raised when an Ollama request fails in a non-recoverable way."""


def _is_missing_model(exc: urllib.error.HTTPError) -> bool:
    return exc.code == 404


def _describe_error(exc: Exception, base_url: str, model: str, timeout: float) -> str:
    """Build an actionable message for an Ollama failure."""
    if isinstance(exc, urllib.error.HTTPError):
        body = ""
        try:
            body = exc.read().decode("utf-8", "ignore")[:300]
        except OSError:
            pass
        if exc.code == 404:
            return (
                f"Ollama model '{model}' is not installed (HTTP 404). "
                f"Run: ollama pull {model}"
            )
        return f"Ollama returned HTTP {exc.code} for model '{model}': {body or exc.reason}"
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return (
            f"Ollama request for model '{model}' timed out after {timeout}s. "
            "Increase ollama.timeout_seconds or use a smaller model."
        )
    if isinstance(exc, urllib.error.URLError):
        return (
            f"Ollama is not reachable at {base_url} ({exc.reason}). "
            "Is the Ollama server running?"
        )
    return f"Ollama request for model '{model}' failed: {exc}"


class OllamaBackend(ModelBackend):
    name = "ollama"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.retry_backoff_seconds = retry_backoff_seconds

    @classmethod
    def from_config(cls, config) -> "OllamaBackend":
        ollama = config.ollama
        return cls(
            base_url=ollama.get("base_url", "http://localhost:11434"),
            timeout=ollama.get("timeout_seconds", 120),
            max_retries=ollama.get("max_retries", 2),
            retry_backoff_seconds=ollama.get("retry_backoff_seconds", 1.0),
        )

    # -- low level ---------------------------------------------------------
    def _get(self, path: str, timeout: float | None = None):
        req = urllib.request.Request(f"{self.base_url}{path}", method="GET")
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _post(self, path: str, payload: dict, timeout: float | None = None):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # -- interface ---------------------------------------------------------
    def is_available(self) -> bool:
        try:
            self._get("/api/tags", timeout=3)
            return True
        except (urllib.error.URLError, OSError, ValueError, TimeoutError):
            return False

    def list_models(self) -> list[str]:
        try:
            data = self._get("/api/tags", timeout=5)
        except (urllib.error.URLError, OSError, ValueError, TimeoutError):
            return []
        return [m.get("name", "") for m in data.get("models", []) if m.get("name")]

    def generate(
        self,
        prompt: str,
        *,
        model: str,
        system: str = "",
        json_mode: bool = False,
        max_tokens: int = 512,
        temperature: float = 0.2,
        structured: dict | None = None,
    ) -> GenerationResult:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            start = time.time()
            self._emit(
                "ollama_request_start",
                endpoint="/api/generate",
                model=model,
                attempt=attempt,
                timeout=self.timeout,
            )
            try:
                data = self._post("/api/generate", payload)
                latency = round((time.time() - start) * 1000, 2)
                self._emit(
                    "ollama_request_success",
                    endpoint="/api/generate",
                    model=model,
                    attempt=attempt,
                    latency_ms=latency,
                )
                return GenerationResult(
                    text=data.get("response", ""),
                    model=model,
                    backend=self.name,
                    latency_ms=latency,
                    raw={k: v for k, v in data.items() if k != "context"},
                )
            except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
                last_error = exc
                status = getattr(exc, "code", None)
                self._emit(
                    "ollama_request_error",
                    endpoint="/api/generate",
                    model=model,
                    attempt=attempt,
                    error_type=type(exc).__name__,
                    status=status,
                )
                # A missing model will never succeed on retry — fail fast.
                if isinstance(exc, urllib.error.HTTPError) and _is_missing_model(exc):
                    break
                if attempt < self.max_retries:
                    backoff = self.retry_backoff_seconds * (2 ** attempt)
                    self._emit(
                        "ollama_retry", model=model, attempt=attempt + 1, backoff_seconds=backoff
                    )
                    time.sleep(backoff)

        raise OllamaError(
            _describe_error(last_error, self.base_url, model, self.timeout)
        )
