"""Model backend interface and selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


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

    ``event_sink``, when set, receives backend events as ``(name, **data)`` so
    a run's Trace Pack can record request lifecycle without coupling the
    backend to the runtime.
    """

    name = "base"
    event_sink: Callable[..., None] | None = None

    def _emit(self, name: str, **data) -> None:
        if self.event_sink is not None:
            self.event_sink(name, **data)

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


def is_embedding_model(name: str) -> bool:
    """Heuristic: embedding models cannot be used for text generation."""
    lowered = name.lower()
    return "embed" in lowered or "bge" in lowered


def select_generative_model(
    config,
    installed: list[str],
    profiles: tuple[str, ...] = ("worker", "verifier", "brain", "coder"),
) -> str | None:
    """Pick an installed text-generation model.

    Prefers a manifest profile candidate that is installed; otherwise falls
    back to the first installed non-embedding model. Returns None if none fit.
    """
    installed_set = set(installed)
    for profile in profiles:
        for candidate in config.model_profiles.get(profile, {}).get("candidates", []) or []:
            if candidate in installed_set:
                return candidate
    for model in installed:
        if not is_embedding_model(model):
            return model
    return None


def select_backend(config, prefer: str | None = None) -> ModelBackend:
    """Pick a backend: Ollama when reachable, otherwise the offline backend.

    ``prefer`` may be 'ollama' or 'offline' to force a choice.
    """
    from heimdal.models.offline import OfflineBackend
    from heimdal.models.ollama import OllamaBackend

    if prefer == "offline":
        return OfflineBackend()

    ollama = OllamaBackend.from_config(config)
    if prefer == "ollama":
        return ollama
    if ollama.is_available() and ollama.list_models():
        return ollama
    return OfflineBackend()
