"""Deterministic stub teacher.

The stub composes a teacher answer from the local output by adding light
structure, a couple of caveats, and an explicit "verify before use" note.
It is *intentionally* better-structured than a typical local answer so the
diff engine has something to score on -- but it never invents facts, so
tests that assert "teacher beats local on structure" don't have to mock a
real model.

For hallucination tests, use the alternative ``HallucinatingStub`` which
injects a fabricated specific claim so the diff engine can flag it.
"""

from __future__ import annotations

from heimdal.mirror.provider import TeacherInput, TeacherProvider, TeacherResult

STUB_MODEL = "heimdal-mirror-stub"


def _bullet_lines(text: str, *, max_bullets: int = 4) -> list[str]:
    """Split ``text`` into a few short bullet-friendly lines."""
    raw = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    return [s.rstrip(".") for s in raw[:max_bullets]]


class StubTeacher(TeacherProvider):
    """Adds structure + a caveat to the local output -- no new facts."""

    name = "stub"

    def __init__(self, model: str = STUB_MODEL):
        self.model = model

    def generate(self, input_: TeacherInput) -> TeacherResult:
        bullets = _bullet_lines(input_.local_output or input_.task.get("objective", ""))
        body_lines = [f"# {input_.task.get('title') or 'Teacher response'}", ""]
        if bullets:
            body_lines.append("## Key points")
            body_lines.extend(f"- {line}" for line in bullets)
        body_lines.extend([
            "",
            "## Caveat",
            (
                "Verify each claim against the cited Truth Vault source "
                "before relying on it."
            ),
        ])
        output = "\n".join(body_lines)
        return TeacherResult(
            provider=self.name,
            model=self.model,
            status="pass",
            output=output,
            usage={
                "input_tokens": len((input_.local_output or "").split()),
                "output_tokens": len(output.split()),
                "estimated_cost": 0.0,
            },
            metadata={"deterministic": True},
        )


class HallucinatingStub(TeacherProvider):
    """Stub that injects a fabricated specific claim -- for testing the
    diff engine's hallucination detection."""

    name = "stub_hallucinator"

    def __init__(self, model: str = "heimdal-mirror-stub-hallucinator"):
        self.model = model

    def generate(self, input_: TeacherInput) -> TeacherResult:
        title = input_.task.get("title") or "Teacher response"
        # The "$249.99 effective 2024-03-15" is a deliberately specific
        # number + date combo with no source -- the diff engine should flag
        # this as a hallucination_risk regardless of the rest of the output.
        output = (
            f"# {title}\n\n"
            "The price is $249.99 effective 2024-03-15 per official guidance.\n\n"
            "Verified across multiple internal records.\n"
        )
        return TeacherResult(
            provider=self.name,
            model=self.model,
            status="pass",
            output=output,
            usage={"input_tokens": 0, "output_tokens": len(output.split()),
                   "estimated_cost": 0.0},
            metadata={"deterministic": True, "hallucinator": True},
        )
