"""Offline deterministic model backend.

Lets the full Quality Factory pipeline run (demo, run, eval, tests) without a
model server. Output is deterministic and composed from the structured context
the worker passes in, so it is honest about what it is: a stub, not a model.
"""

from __future__ import annotations

import json
import re

from heimdal.models.base import GenerationResult, ModelBackend

OFFLINE_MODEL = "heimdal-offline-stub"


def _clean(text: str) -> str:
    """Drop markdown headings/bullets and collapse whitespace from a snippet."""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(re.sub(r"^[-*]\s+", "", stripped))
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", _clean(text))
    return [p.strip() for p in parts if len(p.strip()) > 3]


def _trim_words(text: str, max_words: int | None) -> str:
    if not max_words:
        return text
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[: max_words])


class OfflineBackend(ModelBackend):
    name = "offline"

    def is_available(self) -> bool:
        return True

    def list_models(self) -> list[str]:
        return [OFFLINE_MODEL]

    def generate(
        self,
        prompt: str,
        *,
        model: str = OFFLINE_MODEL,
        system: str = "",
        json_mode: bool = False,
        max_tokens: int = 512,
        temperature: float = 0.2,
        structured: dict | None = None,
    ) -> GenerationResult:
        s = structured or {}
        instruction = (s.get("instruction") or prompt or "").strip()
        title = s.get("title") or "Heimdal Response"
        truth: list[str] = s.get("truth") or []
        max_words = s.get("max_words")

        if json_mode or s.get("output_profile") == "json":
            text = self._compose_json(instruction, truth)
        else:
            text = self._compose_markdown(title, instruction, truth, max_words)

        return GenerationResult(
            text=text,
            model=OFFLINE_MODEL,
            backend=self.name,
            latency_ms=0.0,
            raw={"deterministic": True},
        )

    # -- composition -------------------------------------------------------
    def _compose_markdown(
        self, title: str, instruction: str, truth: list[str], max_words: int | None
    ) -> str:
        body_words_budget = (max_words - 8) if max_words else None
        if truth:
            sentences: list[str] = []
            for snippet in truth:
                sentences.extend(_sentences(snippet))
            body = " ".join(sentences) if sentences else _clean(" ".join(truth))
            body = _trim_words(body, body_words_budget)
        else:
            body = (
                f"This response addresses the request: {instruction} "
                "No grounding sources were supplied, so this is a structured "
                "answer produced by the Heimdal offline backend."
            )
            body = _trim_words(body, body_words_budget)
        return f"## {title}\n\n{body}".strip()

    def _compose_json(self, instruction: str, truth: list[str]) -> str:
        return json.dumps(
            {
                "instruction": instruction,
                "answer": (_clean(truth[0]) if truth else instruction),
                "sources_used": len(truth),
                "generated_by": OFFLINE_MODEL,
            },
            indent=2,
        )
