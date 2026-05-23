"""Teacher provider interface and shared types.

The provider abstraction lets Mirror Mode call deterministic stubs, manual
file-fed teachers, or real cloud providers behind the same surface. CI uses
the stub; real cloud providers stay opt-in via env vars and never run in
the test suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class TeacherInput:
    """The cleaned, redacted payload sent to a teacher provider."""

    case_id: str
    task: dict
    local_output: str
    context_summary: str = ""
    truth_refs: list = field(default_factory=list)
    constraints: dict = field(default_factory=dict)
    rubric: dict = field(default_factory=dict)


@dataclass
class TeacherResult:
    """What a teacher provider returns. ``status`` lets a provider report
    ``skipped`` (e.g. budget exhausted) without raising."""

    provider: str
    model: str
    status: str  # "pass" | "fail" | "skipped"
    output: str
    usage: dict = field(default_factory=lambda: {
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": None,
    })
    metadata: dict = field(default_factory=dict)


class TeacherProvider(Protocol):
    """A teacher provider exposes a single ``generate`` method.

    Providers MUST NOT echo API keys / env vars / .ssh paths into the
    ``output`` or ``metadata`` -- redaction happens before the call but
    providers should also avoid logging anything sensitive themselves.
    """

    name: str

    def generate(self, input_: TeacherInput) -> TeacherResult: ...
