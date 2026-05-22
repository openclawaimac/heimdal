"""Verifier, No-Guess Gate, and optional hybrid semantic verification.

The verifier judges a worker draft and returns a structured Verification
Result (docs/builder_pack/03_schemas/verification_result.schema.json); it never
rewrites the answer.

Gate order (v0.2.3):

  1. deterministic rule-based verifier
  2. optional model-based semantic verifier (hybrid mode, B2-B4 only)
  3. final deterministic check

A deterministic hard fail short-circuits before the model verifier runs, so a
model-based verifier can never override a deterministic fail. The semantic
verifier is an extra quality layer, not a replacement.
"""

from __future__ import annotations

import json
import re

from heimdal import jsonschema_min
from heimdal.core.constants import FAIL, HYBRID, LENIENT, PASS, RULE_BASED, STANDARD, STRICT
from heimdal.core.task_contract import requires_grounding

_SEVERITY_PENALTY = {"low": 0.05, "medium": 0.15, "high": 0.35, "critical": 0.6}
_PASS_THRESHOLD = {LENIENT: 0.5, STANDARD: 0.65, STRICT: 0.8}
_SEVERITIES = ("low", "medium", "high", "critical")

# The semantic verifier only runs for these budget levels.
_SEMANTIC_BUDGETS = {"B2", "B3", "B4"}

# Heuristic markers of factual claims that need a source.
_NUMBER_CLAIM = re.compile(r"(?<![\w])(?:\$|€|£)?\d[\d,]*(?:\.\d+)?\s*(?:%|usd|dollars)?", re.I)

_SEMANTIC_SYSTEM = (
    "You are a strict semantic verifier. Decide whether the RESPONSE genuinely "
    "fulfils the TASK. A response that is empty, asks a question back, or "
    "ignores the task does NOT fulfil it. Return ONLY JSON of the form: "
    '{"status": "pass"|"fail", "score": 0.0-1.0, "confidence": 0.0-1.0, '
    '"defects": [{"severity": "low|medium|high|critical", "message": "..."}], '
    '"rationale_short": "<one short sentence>"}.'
)


def _defect(severity: str, message: str, fix: str | None = None) -> dict:
    defect = {"severity": severity, "message": message}
    if fix:
        defect["suggested_fix"] = fix
    return defect


def _clamp01(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _score(defects: list[dict]) -> float:
    score = 1.0
    for defect in defects:
        score -= _SEVERITY_PENALTY[defect["severity"]]
    return max(0.0, round(score, 3))


def _grade(defects: list[dict], strictness: str) -> str:
    """Deterministic pass/fail decision from a defect set."""
    if any(d["severity"] == "critical" for d in defects):
        return FAIL
    if any(d["severity"] == "high" for d in defects) and strictness != LENIENT:
        return FAIL
    if _score(defects) < _PASS_THRESHOLD[strictness]:
        return FAIL
    return PASS


# -- Gate 1: deterministic rule-based checks -------------------------------
def _deterministic_checks(
    output_text: str, contract: dict, packet: dict, strictness: str
) -> tuple[list[dict], list[str], list[str]]:
    verification = contract.get("verification", {})
    constraints = contract.get("constraints", {}) or {}
    truth_context = packet.get("truth_context", []) or []

    defects: list[dict] = []
    missing_sources: list[str] = []
    schema_errors: list[str] = []

    text = (output_text or "").strip()
    word_count = len(text.split())

    if not text:
        defects.append(_defect("critical", "Worker produced an empty response."))

    max_words = constraints.get("max_words")
    if max_words and word_count > int(max_words):
        defects.append(
            _defect(
                "high",
                f"Response has {word_count} words, exceeds max_words={max_words}.",
                "Shorten the response.",
            )
        )

    if verification.get("requires_schema_validation"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            parsed = None
            schema_errors.append(f"output is not valid JSON: {exc}")
            defects.append(_defect("high", "Output must be valid JSON but failed to parse."))
        expected_schema = constraints.get("output_schema")
        if parsed is not None and isinstance(expected_schema, dict):
            errors = jsonschema_min.validate(parsed, expected_schema)
            schema_errors.extend(errors)
            if errors:
                defects.append(
                    _defect("high", "Output JSON does not match the required schema.")
                )

    if verification.get("no_guess_gate") and requires_grounding(verification):
        if not truth_context:
            missing_sources.append("no truth sources retrieved for a source-required task")
            defects.append(
                _defect(
                    "critical",
                    "Source-required task answered without any retrieved source.",
                    "Retrieve a source or return need_input.",
                )
            )
        elif _NUMBER_CLAIM.search(text) and strictness == STRICT:
            refs = [s.get("ref", "") for s in truth_context]
            if not any(ref and ref in text for ref in refs):
                defects.append(
                    _defect(
                        "medium",
                        "Numeric claim present without an explicit source reference.",
                        "Cite the source ref next to the figure.",
                    )
                )

    return defects, missing_sources, schema_errors


# -- Gate 2: model-based semantic verifier ---------------------------------
def _normalize_semantic(raw, model: str) -> dict:
    """Coerce a model's reply into a schema-valid semantic result.

    A malformed reply is non-fatal: it normalises to a pass with confidence 0
    so the deterministic gate (which already passed) stays decisive.
    """
    if not isinstance(raw, dict):
        return {
            "status": PASS,
            "score": 1.0,
            "confidence": 0.0,
            "defects": [],
            "rationale_short": "semantic verification unavailable",
            "model": model,
        }
    status = raw.get("status") if raw.get("status") in (PASS, FAIL) else PASS
    defects = []
    for item in raw.get("defects", []) or []:
        if isinstance(item, dict) and item.get("message"):
            severity = item.get("severity")
            defects.append(
                {
                    "severity": severity if severity in _SEVERITIES else "medium",
                    "message": str(item["message"]),
                }
            )
    rationale = str(raw.get("rationale_short") or "").strip()[:300]
    return {
        "status": status,
        "score": _clamp01(raw.get("score", 1.0 if status == PASS else 0.0)),
        "confidence": _clamp01(raw.get("confidence", 0.5)),
        "defects": defects,
        "rationale_short": rationale or "no rationale provided",
        "model": model,
    }


def _semantic_verify(output_text, contract, routing, backend, config) -> dict:
    """Run the model-based semantic verifier and return a schema-valid result."""
    model = routing.get("semantic_verifier_model") or "unknown"
    objective = contract.get("objective", "")
    prompt = f"TASK:\n{objective}\n\nRESPONSE:\n{output_text}\n"
    raw = None
    try:
        gen = backend.generate(
            prompt,
            model=model,
            system=_SEMANTIC_SYSTEM,
            json_mode=True,
            max_tokens=300,
            temperature=0.0,
            structured={
                "verify_task": "semantic",
                "objective": objective,
                "answer": output_text,
            },
        )
        raw = json.loads(gen.text)
    except (RuntimeError, OSError, ValueError, json.JSONDecodeError):
        raw = None
    result = _normalize_semantic(raw, model)
    jsonschema_min.validate_or_raise(
        result,
        config.schema_path("semantic_verification.schema.json"),
        "Semantic Verification Result",
    )
    return result


# -- orchestration ---------------------------------------------------------
def verify(
    output_text: str, contract: dict, packet: dict, routing: dict, config, backend=None
) -> dict:
    """Return a schema-valid Verification Result for a worker draft."""
    strictness = routing.get("verifier_strictness", STANDARD)
    verifier_backend = routing.get("verifier_backend", RULE_BASED)

    # Gate 1: deterministic rule-based verifier.
    det_defects, missing_sources, schema_errors = _deterministic_checks(
        output_text, contract, packet, strictness
    )
    semantic_result = None

    if _grade(det_defects, strictness) == FAIL:
        # Deterministic hard fail: short-circuit, the model verifier never runs.
        final_defects = det_defects
    else:
        # Gate 2: optional model-based semantic verifier (hybrid, B2-B4 only).
        run_semantic = (
            verifier_backend == HYBRID
            and routing.get("quality_level") in _SEMANTIC_BUDGETS
            and backend is not None
        )
        if run_semantic:
            semantic_result = _semantic_verify(
                output_text, contract, routing, backend, config
            )
        # Gate 3: final deterministic check. Re-grade over the deterministic
        # defects plus any semantic failure; deterministic results stay
        # authoritative, the semantic verifier can only ADD a defect.
        final_defects = list(det_defects)
        if semantic_result is not None and semantic_result["status"] == FAIL:
            final_defects.append(
                _defect(
                    "high",
                    f"Semantic verifier: {semantic_result['rationale_short']}",
                    "Make the answer genuinely fulfil the task.",
                )
            )

    result = {
        "status": _grade(final_defects, strictness),
        "score": _score(final_defects),
        "defects": final_defects,
        "missing_sources": missing_sources,
        "schema_errors": schema_errors,
        "verifier_backend": verifier_backend,
        "semantic": semantic_result,
    }
    jsonschema_min.validate_or_raise(
        result,
        config.schema_path("verification_result.schema.json"),
        "Verification Result",
    )
    return result
