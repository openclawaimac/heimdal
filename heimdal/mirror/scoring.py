"""Deterministic per-dimension scoring for local + teacher outputs.

The Frontier Diff Engine relies on heuristics rather than a model so the
diff comparison stays cheap, reproducible, and runnable in CI without any
cloud or local LLM call. The heuristics are intentionally simple and
documented; the goal is to score the same answer the same way every run,
not to recreate a human grader.

Each ``score_output(text, task)`` call returns a dict keyed by the 14
documented diff dimensions. All scores are 0..1 floats rounded to three
decimals -- including ``hallucination_risk``, which is inverted so 1.0
means low risk and 0.0 means high risk (consistent with the other
dimensions: higher is always better).
"""

from __future__ import annotations

import re

DIMENSIONS = (
    "task_adherence",
    "factuality",
    "source_grounding",
    "completeness",
    "reasoning_depth",
    "actionability",
    "structure_format",
    "conciseness",
    "uncertainty_handling",
    "hallucination_risk",
    "missing_caveats",
    "tool_or_source_use",
    "no_guess_behavior",
    "semantic_quality",
)

_STOPWORDS = {
    "the", "a", "an", "is", "are", "to", "of", "in", "for", "and", "or",
    "on", "at", "by", "with", "as", "be", "this", "that", "it", "from",
}
_REASONING_WORDS = {"because", "since", "therefore", "thus", "if", "then", "however", "while"}
_UNCERTAINTY_WORDS = {
    "may", "might", "could", "uncertain", "verify", "approximately",
    "assumption", "likely", "perhaps",
}
_ACTION_VERBS = {
    "run", "set", "configure", "update", "review", "check", "install",
    "remove", "deploy", "audit", "enable", "disable",
}
_HEDGE_PHRASES = (
    "i don't know", "no source", "cannot find", "please provide",
    "need more information", "need a source",
)

_NUMBER_RE = re.compile(r"(?<![\w])\$?\d[\d,]*(?:\.\d+)?(?:%|/\w|\b)")
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")
_SOURCE_REF_RE = re.compile(r"\([A-Za-z0-9_\-]+\.(?:md|txt|json|yaml|yml)\)|\[\d+\]")


def _content_tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def _has_source_refs(text: str) -> bool:
    return bool(_SOURCE_REF_RE.search(text or ""))


def score_output(text: str, task: dict, *, truth_refs: list | None = None) -> dict:
    """Return a per-dimension score dict for ``text`` against ``task``.

    ``hallucination_risk`` is inverted (higher = lower risk) so the diff
    engine can naively compare "higher score is better" across all
    dimensions.
    """
    text = text or ""
    task = task or {}
    objective = task.get("objective", "")
    constraints = task.get("constraints", {}) or {}
    requires_sources = bool(constraints.get("requires_sources"))

    task_terms = _content_tokens(objective)
    answer_terms = _content_tokens(text)
    overlap = len(task_terms & answer_terms)
    word_count = len(text.split())

    refs_present = _has_source_refs(text) or bool(truth_refs)
    numbers = len(_NUMBER_RE.findall(text))
    dates = len(_DATE_RE.findall(text))
    specific_claims = numbers + dates
    ref_count = len(_SOURCE_REF_RE.findall(text))

    scores: dict[str, float] = {}

    # 1. task_adherence -- coverage of task content terms.
    scores["task_adherence"] = (
        min(1.0, overlap / max(1, len(task_terms))) if task_terms else 0.5
    )

    # 2. factuality -- specific unsupported claims drag this down; cited
    #    claims (or no claims at all) are fine.
    if specific_claims == 0:
        scores["factuality"] = 0.85
    elif refs_present:
        scores["factuality"] = max(0.0, 0.85 - 0.05 * specific_claims)
    else:
        scores["factuality"] = max(0.0, 0.55 - 0.15 * specific_claims)

    # 3. source_grounding -- explicit source refs vs requirement.
    if requires_sources:
        scores["source_grounding"] = min(1.0, ref_count / 2.0)
    else:
        scores["source_grounding"] = min(1.0, 0.5 + ref_count * 0.25)

    # 4. completeness -- coverage of task terms + adequate length.
    coverage = overlap / max(1, len(task_terms)) if task_terms else 0.5
    if word_count < 10:
        coverage *= 0.5
    scores["completeness"] = min(1.0, coverage + (0.1 if word_count > 30 else 0.0))

    # 5. reasoning_depth -- presence of connective words.
    reasoning = sum(1 for w in _REASONING_WORDS if w in text.lower())
    scores["reasoning_depth"] = min(1.0, 0.3 + 0.15 * reasoning)

    # 6. actionability.
    actions = sum(1 for v in _ACTION_VERBS if v in text.lower())
    scores["actionability"] = min(1.0, 0.4 + 0.15 * actions)

    # 7. structure_format -- markdown headers + bullets.
    headers = len(re.findall(r"(?m)^#+\s", text))
    bullets = len(re.findall(r"(?m)^[-*]\s", text))
    scores["structure_format"] = min(1.0, 0.3 + 0.2 * headers + 0.1 * bullets)

    # 8. conciseness -- respects max_words (or a 200-word floor).
    max_words = constraints.get("max_words")
    if max_words:
        scores["conciseness"] = (
            1.0 if word_count <= max_words
            else max(0.0, 1.0 - (word_count - max_words) / max_words)
        )
    elif word_count > 200:
        scores["conciseness"] = max(0.0, 1.0 - (word_count - 200) / 400)
    else:
        scores["conciseness"] = 0.9

    # 9. uncertainty_handling -- hedge words present where appropriate.
    uncertainty = sum(1 for w in _UNCERTAINTY_WORDS if w in text.lower())
    scores["uncertainty_handling"] = min(1.0, 0.4 + 0.2 * uncertainty)

    # 10. hallucination_risk -- INVERTED. Specific claims without source
    #     refs are the strongest signal of risk.
    risk = 0.0
    if specific_claims > 0 and not refs_present:
        risk += 0.5 * min(1.0, specific_claims / 3.0)
    if dates and not refs_present:
        risk += 0.3
    scores["hallucination_risk"] = max(0.0, 1.0 - risk)

    # 11. missing_caveats -- when the task touches policy/legal/compliance
    #     a caveat is expected.
    needs_caveat = any(
        kw in objective.lower()
        for kw in ("policy", "legal", "rule", "regulation", "compliance",
                   "license", "guarantee")
    )
    if needs_caveat:
        scores["missing_caveats"] = 0.85 if uncertainty else 0.3
    else:
        scores["missing_caveats"] = 0.75

    # 12. tool_or_source_use -- explicit source references.
    scores["tool_or_source_use"] = min(1.0, 0.4 + 0.25 * ref_count)

    # 13. no_guess_behavior -- for source-required tasks, expect a hedge
    #     when no source is cited.
    if requires_sources:
        if any(p in text.lower() for p in _HEDGE_PHRASES):
            scores["no_guess_behavior"] = 0.95
        elif refs_present:
            scores["no_guess_behavior"] = 0.85
        else:
            scores["no_guess_behavior"] = 0.35
    else:
        scores["no_guess_behavior"] = 0.7

    # 14. semantic_quality -- text exists and addresses the task.
    if not text.strip():
        scores["semantic_quality"] = 0.0
    elif word_count < 5 and "?" in text:
        # A short question-back is exactly what semantic verifier rejects.
        scores["semantic_quality"] = 0.15
    else:
        scores["semantic_quality"] = round(
            (scores["task_adherence"] + scores["completeness"]) / 2, 3
        )

    return {k: round(v, 3) for k, v in scores.items()}
