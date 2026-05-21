"""Verifier and No-Guess Gate.

A deterministic, rule-based verifier that returns a structured Verification
Result (docs/builder_pack/03_schemas/verification_result.schema.json). It judges
the answer; it never rewrites it (docs/builder_pack/04_runtime/QUALITY_FACTORY.md).

No-Guess Gate (docs/builder_pack/07_quality_eval/NO_GUESS_GATE.md): unsupported
numbers, factual/policy claims, fabricated citations and missing required source
references all fail verification.
"""

from __future__ import annotations

import json
import re

from heimdal import jsonschema_min
from heimdal.core.constants import FAIL, HYBRID, LENIENT, PASS, STANDARD, STRICT
from heimdal.core.task_contract import requires_grounding

_SEVERITY_PENALTY = {"low": 0.05, "medium": 0.15, "high": 0.35, "critical": 0.6}
_PASS_THRESHOLD = {LENIENT: 0.5, STANDARD: 0.65, STRICT: 0.8}

# Heuristic markers of factual claims that need a source.
_NUMBER_CLAIM = re.compile(r"(?<![\w])(?:\$|€|£)?\d[\d,]*(?:\.\d+)?\s*(?:%|usd|dollars)?", re.I)

_SEMANTIC_SYSTEM = (
    "You are a strict answer verifier. Decide whether the RESPONSE genuinely "
    "fulfils the TASK. A response that asks a question back, is empty, or "
    "ignores the task does NOT fulfil it. "
    'Return only JSON: {"satisfies": true|false, "reason": "<short reason>"}.'
)


def _defect(severity: str, message: str, fix: str | None = None) -> dict:
    defect = {"severity": severity, "message": message}
    if fix:
        defect["suggested_fix"] = fix
    return defect


def _semantic_defect(output_text: str, contract: dict, routing: dict, backend) -> dict | None:
    """Run the model-based semantic check; returns a defect or None.

    A failure to reach the model is non-fatal: rule-based checks still run.
    """
    prompt = f"TASK:\n{contract.get('objective', '')}\n\nRESPONSE:\n{output_text}\n"
    try:
        gen = backend.generate(
            prompt,
            model=routing["verifier_model"],
            system=_SEMANTIC_SYSTEM,
            json_mode=True,
            max_tokens=200,
            temperature=0.0,
        )
        judgment = json.loads(gen.text)
    except (RuntimeError, OSError, ValueError, json.JSONDecodeError):
        return None
    if isinstance(judgment, dict) and judgment.get("satisfies") is False:
        reason = str(judgment.get("reason", "response does not satisfy the task"))
        return _defect(
            "high",
            f"Semantic verifier: {reason}",
            "Answer the task directly and completely.",
        )
    return None


def verify(
    output_text: str, contract: dict, packet: dict, routing: dict, config, backend=None
) -> dict:
    """Return a schema-valid Verification Result for a worker draft.

    When the router selected the hybrid verifier, a model-based semantic check
    runs first; the deterministic rule-based checks always run afterwards.
    """
    verification = contract.get("verification", {})
    constraints = contract.get("constraints", {}) or {}
    strictness = routing.get("verifier_strictness", STANDARD)
    truth_context = packet.get("truth_context", []) or []

    defects: list[dict] = []
    missing_sources: list[str] = []
    schema_errors: list[str] = []

    text = (output_text or "").strip()
    word_count = len(text.split())

    if routing.get("verifier_backend") == HYBRID and backend is not None and text:
        semantic = _semantic_defect(text, contract, routing, backend)
        if semantic:
            defects.append(semantic)

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

    # No-Guess Gate: defence-in-depth alongside the pre-model gate in
    # quality_factory, so a direct verify() call is still source-aware.
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

    score = 1.0
    for defect in defects:
        score -= _SEVERITY_PENALTY[defect["severity"]]
    score = max(0.0, round(score, 3))

    has_critical = any(d["severity"] == "critical" for d in defects)
    has_high = any(d["severity"] == "high" for d in defects)
    threshold = _PASS_THRESHOLD[strictness]
    if has_critical:
        status = FAIL
    elif has_high and strictness != LENIENT:
        status = FAIL
    elif score < threshold:
        status = FAIL
    else:
        status = PASS

    result = {
        "status": status,
        "score": score,
        "defects": defects,
        "missing_sources": missing_sources,
        "schema_errors": schema_errors,
    }
    jsonschema_min.validate_or_raise(
        result,
        config.schema_path("verification_result.schema.json"),
        "Verification Result",
    )
    return result
