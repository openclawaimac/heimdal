"""Ollama model backend.

Talks to an Ollama server over HTTP using the standard library only. The base
URL is configurable (manifest / OLLAMA_HOST). Failures degrade gracefully so
the runtime can fall back to the offline backend.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from heimdal.models.base import GenerationResult, ModelBackend


class OllamaBackend(ModelBackend):
    name = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # -- low level ---------------------------------------------------------
    def _get(self, path: str, timeout: float | None = None):
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _post(self, path: str, payload: dict, timeout: float | None = None):
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, method="POST", headers={"Content-Type": "application/json"}
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

        start = time.time()
        last_error: Exception | None = None
        for attempt in range(2):  # one retry
            try:
                data = self._post("/api/generate", payload)
                latency = (time.time() - start) * 1000
                return GenerationResult(
                    text=data.get("response", ""),
                    model=model,
                    backend=self.name,
                    latency_ms=latency,
                    raw={k: v for k, v in data.items() if k != "context"},
                )
            except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
                last_error = exc
                time.sleep(1 + attempt)
        raise RuntimeError(f"Ollama generation failed: {last_error}")
