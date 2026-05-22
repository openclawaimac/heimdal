"""Machine-readable status / error codes for host integrations.

A host (Hermes, OpenClaw) gets a stable ``code`` alongside the human-readable
``message`` so it can branch programmatically instead of parsing prose. Codes
are deliberately coarse and stable: they name *why* a result is what it is.
"""

from __future__ import annotations

# Pass.
OK = "OK"

# need_input -- the task cannot be answered without more grounding.
SOURCE_MISSING = "SOURCE_MISSING"
SOURCE_SUPPORT_INSUFFICIENT = "SOURCE_SUPPORT_INSUFFICIENT"

# fail -- the answer was produced but did not pass verification.
VERIFIER_SEMANTIC_FAIL = "VERIFIER_SEMANTIC_FAIL"
VERIFIER_RULE_FAIL = "VERIFIER_RULE_FAIL"
SCHEMA_INVALID = "SCHEMA_INVALID"

# Backend / model availability.
OLLAMA_MODEL_MISSING = "OLLAMA_MODEL_MISSING"
OLLAMA_UNREACHABLE = "OLLAMA_UNREACHABLE"

# Callback delivery.
CALLBACK_DELIVERY_FAILED = "CALLBACK_DELIVERY_FAILED"

# Local file bridge.
JOB_SCHEMA_INVALID = "JOB_SCHEMA_INVALID"
ADAPTER_UNSUPPORTED = "ADAPTER_UNSUPPORTED"
INTERNAL_ERROR = "INTERNAL_ERROR"

ALL_CODES = [
    OK,
    SOURCE_MISSING,
    SOURCE_SUPPORT_INSUFFICIENT,
    VERIFIER_SEMANTIC_FAIL,
    VERIFIER_RULE_FAIL,
    SCHEMA_INVALID,
    OLLAMA_MODEL_MISSING,
    OLLAMA_UNREACHABLE,
    CALLBACK_DELIVERY_FAILED,
    JOB_SCHEMA_INVALID,
    ADAPTER_UNSUPPORTED,
    INTERNAL_ERROR,
]


def fail_code(verification: dict) -> str:
    """Classify a FAIL Verification Result into a machine-readable code.

    Schema failures rank first, then a semantic-verifier defect, then a plain
    rule-based failure.
    """
    if verification.get("schema_errors"):
        return SCHEMA_INVALID
    for defect in verification.get("defects", []) or []:
        if str(defect.get("message", "")).startswith("Semantic verifier:"):
            return VERIFIER_SEMANTIC_FAIL
    return VERIFIER_RULE_FAIL
