"""Redaction for outbound teacher payloads.

Mirror Mode applies this before any cloud call. The goal is honest, not
perfect -- we cover the obvious cases (API keys, env-var assignments,
private-key markers, .ssh paths, Authorization headers) and document what
we do NOT cover. Anything sensitive in the user's truth vault must be
flagged with the existing privacy_mode gate, not relied upon here.

Each redaction replaces the match with ``[REDACTED:<kind>]`` so the
operator can audit what was stripped from the manifest of redactions
returned alongside the cleaned text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# (kind, pattern) -- order matters: longer / more specific patterns first.
_PATTERNS: list[tuple[str, "re.Pattern[str]"]] = [
    (
        "private_key_block",
        re.compile(
            r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"
            r".*?-----END (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----",
            re.DOTALL,
        ),
    ),
    ("openai_api_key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}\b")),
    ("anthropic_api_key", re.compile(r"\bsk-ant-(?:api|admin)[A-Za-z0-9_\-]{20,}\b")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{30,}\b")),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("bearer_header", re.compile(r"(?i)\b(?:authorization|bearer)[:=]\s*[A-Za-z0-9._\-]{16,}")),
    (
        "env_secret_assignment",
        re.compile(
            r"(?i)\b("
            r"password|passwd|secret|token|api[_\-]?key|access[_\-]?key|"
            r"private[_\-]?key|session[_\-]?key|client[_\-]?secret"
            r")\s*[:=]\s*[\"']?[A-Za-z0-9_\-+/=.]{6,}[\"']?"
        ),
    ),
    ("ssh_path", re.compile(r"~?(?:/[A-Za-z0-9_.\-]+)*/\.ssh(?:/[A-Za-z0-9_.\-/]+)?")),
]


@dataclass
class RedactionResult:
    text: str
    redactions: list[dict]


def redact(text: str) -> RedactionResult:
    """Return a copy of ``text`` with obvious secrets removed."""
    if not text:
        return RedactionResult(text="", redactions=[])
    redactions: list[dict] = []
    cleaned = text
    for kind, pattern in _PATTERNS:
        def _replace(match: "re.Match[str]") -> str:
            sample = match.group(0)
            redactions.append({
                "kind": kind,
                "length": len(sample),
            })
            return f"[REDACTED:{kind}]"
        cleaned = pattern.sub(_replace, cleaned)
    return RedactionResult(text=cleaned, redactions=redactions)


def redact_payload(payload: dict) -> tuple[dict, list[dict]]:
    """Walk ``payload`` and redact string values; return (cleaned, redactions)."""
    redactions: list[dict] = []

    def _walk(node):
        if isinstance(node, str):
            result = redact(node)
            redactions.extend(result.redactions)
            return result.text
        if isinstance(node, dict):
            return {k: _walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_walk(v) for v in node]
        return node

    return _walk(payload), redactions
