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

_SEVERITY_PENALTY = {"low": 0.05, "medium": 0.15, "high": 0.35, "critical": 0.6}
_PASS_THRESHOLD = {"lenient": 0.5, "standard": 0.65, "strict": 0.8}

# Heuristic markers of factual claims that need a source.
_NUMBER_CLAIM = re.compile(r"(?<![\w])(?:\$|€|£)?\d[\d,]*(?:\.\d+)?\s*(?:%|usd|dollars)?", re.I)


def _defect(severity: str, message: str, fix: str | None = None) -> dict:
    defect = {"severity": severity, "message": message}
    if fix:
        defect["suggested_fix"] = fix
    return defect


def verify(output_text: str, contract: dict, packet: dict, routing: dict, config) -> dict:
    """Return a schema-valid Verification Result for a worker draft."""
    verification = contract.get("verification", {})
    constraints = contract.get("constraints", {}) or {}
    strictness = routing.get("verifier_strictness", "standard")
    truth_context = packet.get("truth_context", []) or []

    defects: list[dict] = []
    missing_sources: list[str] = []
    schema_errors: list[str] = []

    text = (output_text or "").strip()

    if not text:
        defects.append(_defect("critical", "Worker produced an empty response."))

    # -- constraint checks -------------------------------------------------
    max_words = constraints.get("max_words")
    if max_words and len(text.split()) > int(max_words):
        defects.append(
            _defect(
                "high",
                f"Response has {len(text.split())} words, exceeds max_words={max_words}.",
                "Shorten the response.",
            )
        )

    # -- schema validation -------------------------------------------------
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

    # -- No-Guess Gate -----------------------------------------------------
    requires_sources = verification.get("requires_sources") or verification.get(
        "requires_citations"
    )
    if verification.get("no_guess_gate") and requires_sources:
        if not truth_context:
            missing_sources.append("no truth sources retrieved for a source-required task")
            defects.append(
                _defect(
                    "critical",
                    "Source-required task answered without any retrieved source.",
                    "Retrieve a source or return need_input.",
                )
            )
        elif _NUMBER_CLAIM.search(text) and strictness == "strict":
            refs = [s.get("ref", "") for s in truth_context]
            if not any(ref and ref in text for ref in refs):
                defects.append(
                    _defect(
                        "medium",
                        "Numeric claim present without an explicit source reference.",
                        "Cite the source ref next to the figure.",
                    )
                )

    # -- scoring -----------------------------------------------------------
    score = 1.0
    for defect in defects:
        score -= _SEVERITY_PENALTY.get(defect["severity"], 0.1)
    score = max(0.0, round(score, 3))

    has_critical = any(d["severity"] == "critical" for d in defects)
    has_high = any(d["severity"] == "high" for d in defects)
    threshold = _PASS_THRESHOLD.get(strictness, 0.65)
    status = "pass"
    if has_critical:
        status = "fail"
    elif has_high and strictness != "lenient":
        status = "fail"
    elif score < threshold:
        status = "fail"

    result = {
        "status": status,
        "score": score,
        "defects": defects,
        "missing_sources": missing_sources,
        "schema_errors": schema_errors,
    }

    schema = jsonschema_min.load_schema(
        config.schema_path("verification_result.schema.json")
    )
    errors = jsonschema_min.validate(result, schema)
    if errors:  # pragma: no cover - defensive
        raise ValueError("Verifier produced an invalid result: " + "; ".join(errors))
    return result
