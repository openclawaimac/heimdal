"""Truth Vault retrieval.

Heimdal is truth-first: factual answers must be grounded in the local Truth
Vault (storage/truth) rather than guessed. This module does BM25-style keyword
retrieval over plain-text and markdown files in that directory, with no
external dependency.

A document must share at least ``MIN_OVERLAP`` distinct content words with the
query before it is considered a match. That gate keeps the No-Guess Gate
honest: an irrelevant file in the vault must not let a sourced task answer.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass

_STOPWORDS = {
    "the", "a", "an", "of", "to", "and", "or", "is", "are", "for", "in", "on",
    "what", "how", "why", "with", "as", "be", "this", "that", "it", "its",
    "write", "explain", "describe", "state", "exact", "short", "by", "from",
    "at", "into", "named", "called", "using", "use", "about", "give", "list",
    "summarize", "provide", "precise", "guaranteed", "reported", "offered",
    "only",
}

# Plain-text and markdown only (v0.2.2 Truth Vault scope).
_TEXT_EXTENSIONS = (".md", ".txt")

MIN_OVERLAP = 2

# Standard BM25 parameters.
_BM25_K1 = 1.5
_BM25_B = 0.75


@dataclass
class TruthSnippet:
    ref: str
    path: str
    text: str
    score: float


@dataclass
class _Doc:
    ref: str
    path: str
    text: str
    tf: dict[str, int]
    length: int


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 1]


def content_terms(text: str) -> set[str]:
    """The distinct content tokens of ``text`` (stopwords/1-char tokens removed)."""
    return set(_tokens(text))


def grounding_coverage(objective: str, truth_context) -> float:
    """How well retrieved Truth Vault snippets cover a task's content terms.

    Returns the fraction of the objective's distinct content terms that the
    single best-matching retrieved snippet shares, in [0.0, 1.0]; 0.0 when
    nothing relevant was retrieved. The No-Guess Gate uses this to tell genuine
    grounding from an incidental keyword overlap -- e.g. a vault document that
    merely shares a generic word like "local" with the instruction must not
    count as a source for an unrelated factual question.
    """
    query_terms = content_terms(objective)
    if not query_terms:
        return 0.0
    best = 0
    for snippet in truth_context or []:
        overlap = query_terms & content_terms(snippet.get("text", ""))
        best = max(best, len(overlap))
    return best / len(query_terms)


class TruthStore:
    """BM25-style keyword retrieval over the local Truth Vault."""

    def __init__(self, truth_dir: str):
        self.truth_dir = truth_dir

    def _iter_files(self):
        if not os.path.isdir(self.truth_dir):
            return
        for root, _dirs, files in os.walk(self.truth_dir):
            for name in sorted(files):
                if name.lower().endswith(_TEXT_EXTENSIONS):
                    yield os.path.join(root, name)

    def list_sources(self) -> list[dict]:
        """List Truth Vault files with their ref and size."""
        sources = []
        for path in self._iter_files():
            sources.append(
                {
                    "ref": os.path.relpath(path, self.truth_dir),
                    "path": path,
                    "size_bytes": os.path.getsize(path),
                }
            )
        return sources

    def _load_corpus(self) -> list[_Doc]:
        docs: list[_Doc] = []
        for path in self._iter_files():
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    content = fh.read().strip()
            except OSError:
                continue
            tokens = _tokens(content)
            if not content or not tokens:
                continue
            tf: dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            docs.append(
                _Doc(
                    ref=os.path.relpath(path, self.truth_dir),
                    path=path,
                    text=content,
                    tf=tf,
                    length=len(tokens),
                )
            )
        return docs

    def retrieve(
        self, query: str, k: int = 3, min_score: float = 0.0
    ) -> list[TruthSnippet]:
        """Return the top-k Truth Vault snippets relevant to ``query``.

        A document must share at least ``MIN_OVERLAP`` query terms and score at
        least ``min_score`` (BM25) to be returned.
        """
        query_terms = set(_tokens(query))
        if not query_terms:
            return []

        docs = self._load_corpus()
        if not docs:
            return []

        total = len(docs)
        avgdl = sum(d.length for d in docs) / total
        doc_freq: dict[str, int] = {}
        for doc in docs:
            for term in doc.tf:
                doc_freq[term] = doc_freq.get(term, 0) + 1

        results: list[TruthSnippet] = []
        for doc in docs:
            overlap = query_terms & set(doc.tf)
            if len(overlap) < MIN_OVERLAP:
                continue  # relevance gate: too weak to count as a source
            score = 0.0
            for term in overlap:
                idf = math.log(
                    1 + (total - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5)
                )
                freq = doc.tf[term]
                denom = freq + _BM25_K1 * (
                    1 - _BM25_B + _BM25_B * doc.length / avgdl
                )
                score += idf * (freq * (_BM25_K1 + 1)) / denom
            if score < min_score:
                continue
            results.append(
                TruthSnippet(ref=doc.ref, path=doc.path, text=doc.text, score=round(score, 4))
            )

        results.sort(key=lambda s: s.score, reverse=True)
        return results[:k]
