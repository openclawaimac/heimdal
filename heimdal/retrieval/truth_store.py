"""Truth Vault retrieval.

Heimdal is truth-first: factual answers must be grounded in the local Truth
Vault (storage/truth) rather than guessed. This module does keyword retrieval
over that directory with no external dependency.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

_STOPWORDS = {
    "the", "a", "an", "of", "to", "and", "or", "is", "are", "for", "in", "on",
    "what", "how", "why", "with", "as", "be", "this", "that", "it", "its",
    "write", "explain", "describe", "state", "exact", "short", "by", "from",
    "at", "into", "named", "called", "using", "use", "about", "give", "list",
    "summarize", "provide", "precise", "guaranteed", "reported", "offered",
}

_TEXT_EXTENSIONS = (".md", ".txt", ".json", ".yml", ".yaml")

# A single shared word is too weak to count as a grounding source; require at
# least this many overlapping content words before a document is considered a
# match. This keeps the No-Guess Gate honest.
MIN_OVERLAP = 2


@dataclass
class TruthSnippet:
    ref: str
    path: str
    text: str
    score: float


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 1]


class TruthStore:
    """Keyword retrieval over the local Truth Vault."""

    def __init__(self, truth_dir: str):
        self.truth_dir = truth_dir

    def _iter_files(self):
        if not os.path.isdir(self.truth_dir):
            return
        for root, _dirs, files in os.walk(self.truth_dir):
            for name in sorted(files):
                if name.lower().endswith(_TEXT_EXTENSIONS):
                    yield os.path.join(root, name)

    def retrieve(self, query: str, k: int = 3, min_score: float = 0.12) -> list[TruthSnippet]:
        """Return the top-k truth snippets relevant to ``query``."""
        query_tokens = set(_tokens(query))
        if not query_tokens:
            return []

        results: list[TruthSnippet] = []
        for path in self._iter_files():
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    content = fh.read().strip()
            except OSError:
                continue
            if not content:
                continue
            doc_tokens = _tokens(content)
            if not doc_tokens:
                continue
            overlap = query_tokens & set(doc_tokens)
            score = len(overlap) / len(query_tokens)
            if len(overlap) >= MIN_OVERLAP and score >= min_score:
                results.append(
                    TruthSnippet(
                        ref=os.path.relpath(path, self.truth_dir),
                        path=path,
                        text=content,
                        score=round(score, 3),
                    )
                )

        results.sort(key=lambda s: s.score, reverse=True)
        return results[:k]
