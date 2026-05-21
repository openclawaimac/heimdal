"""Model backend interface and selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationResult:
    text: str
    model: str
    backend: str
    latency_ms: float = 0.0
    raw: dict = field(default_factory=dict)


class ModelBackend:
    """Abstract model backend.

    Subclasses must be safe to construct even when the backend is unreachable;
    availability is reported by :meth:`is_available`.
    """

    name = "base"

    def is_available(self) -> bool:
        raise NotImplementedError

    def list_models(self) -> list[str]:
        raise NotImplementedError

    def generate(
        self,
        prompt: str,
        *,
        model: str,
        system: str = "",
        json_mode: bool = False,
        max_tokens: int = 512,
        temperature: float = 0.2,
        structured: dict[str, Any] | None = None,
    ) -> GenerationResult:
        raise NotImplementedError


def select_backend(config, prefer: str | None = None) -> ModelBackend:
    """Pick a backend: Ollama when reachable, otherwise the offline backend.

    ``prefer`` may be 'ollama' or 'offline' to force a choice.
    """
    from heimdal.models.offline import OfflineBackend
    from heimdal.models.ollama import OllamaBackend

    if prefer == "offline":
        return OfflineBackend()

    ollama = OllamaBackend(
        base_url=config.ollama.get("base_url", "http://localhost:11434"),
        timeout=config.ollama.get("timeout_seconds", 120),
    )
    if prefer == "ollama":
        return ollama
    if ollama.is_available() and ollama.list_models():
        return ollama
    return OfflineBackend()
