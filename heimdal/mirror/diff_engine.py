"""Frontier Diff Engine -- structured local vs teacher comparison.

Compares per-dimension scores and emits findings + a high-level verdict.
The teacher is not assumed correct: a hallucinating teacher gets marked
``teacher_better=False`` regardless of its other dimensions, and a task
where local correctly refused to guess and the teacher fabricated an
answer flips to ``local_better=True``.
"""

from __future__ import annotations

from heimdal.ids import new_id
from heimdal.mirror import scoring

# Minimum per-dimension score gap before we count a dimension as a real
# difference; below this both sides are "noise-equivalent."
_DIMENSION_NOISE_FLOOR = 0.1

# Hallucination_risk is *inverted* (1.0 = low risk). A teacher score
# below this threshold means the teacher is risky enough to disqualify
# its other wins on this case.
_TEACHER_HALLUCINATION_FLOOR = 0.5


def _recommendation_for(dimension: str, winner: str) -> str:
    """Short, actionable hint the proposal builder picks up."""
    if winner == "teacher_better":
        return {
            "task_adherence": "Tighten the worker prompt to anchor on the task objective.",
            "factuality": "Tighten rubric to penalize unsupported specifics.",
            "source_grounding": "Add a rubric/skill that requires inline source refs.",
            "completeness": "Add a skill that enumerates what the answer must cover.",
            "reasoning_depth": "Add a skill that requires explicit reasoning steps.",
            "actionability": "Add a skill that requires concrete next steps.",
            "structure_format": "Adopt the teacher's structure as a skill template.",
            "conciseness": "Add a skill that enforces the answer's word budget.",
            "uncertainty_handling": "Add a skill that explicitly flags uncertainty.",
            "hallucination_risk": "Tighten rubric to reject specifics without sources.",
            "missing_caveats": "Add a rubric clause requiring policy caveats.",
            "tool_or_source_use": "Add a skill that lists source refs.",
            "no_guess_behavior": "Reinforce No-Guess via a rubric or skill patch.",
            "semantic_quality": "Add a verifier rubric that rejects question-back answers.",
        }.get(dimension, "Review teacher's approach for a possible skill.")
    return {
        "factuality": "Teacher invented specifics; not a useful template.",
        "hallucination_risk": "Teacher hallucinated; keep local behavior.",
        "no_guess_behavior": "Local correctly refused to guess; reinforce existing rubric.",
    }.get(
        dimension,
        "Local outperformed the teacher; no patch needed for this dimension.",
    )


def _severity(delta: float) -> str:
    if abs(delta) >= 0.4:
        return "high"
    if abs(delta) >= 0.2:
        return "medium"
    return "low"


def compare(
    *,
    case_id: str,
    local_output: str,
    teacher_output: str,
    task: dict,
    truth_refs: list | None = None,
    mirror_run_id: str = "",
) -> dict:
    """Return a Mirror Diff dict comparing local against teacher.

    The diff captures per-dimension scores, per-dimension findings (only
    where the gap is above the noise floor), and a single verdict --
    ``teacher_better`` / ``local_better`` / ``mixed`` -- that downstream
    code uses to decide whether to emit proposals.
    """
    local_scores = scoring.score_output(local_output, task, truth_refs=truth_refs)
    teacher_scores = scoring.score_output(teacher_output, task, truth_refs=truth_refs)

    teacher_wins = 0
    local_wins = 0
    findings: list[dict] = []
    for dim in scoring.DIMENSIONS:
        delta = teacher_scores[dim] - local_scores[dim]
        if abs(delta) < _DIMENSION_NOISE_FLOOR:
            continue
        winner = "teacher_better" if delta > 0 else "local_better"
        if winner == "teacher_better":
            teacher_wins += 1
        else:
            local_wins += 1
        findings.append({
            "dimension": dim,
            "severity": _severity(delta),
            # Explicit winner per dimension. Downstream code MUST use this,
            # not string-match the finding text -- a local-win finding reads
            # "local=X vs teacher=Y", which also contains "teacher=".
            "winner": winner,
            "finding": (
                f"teacher={teacher_scores[dim]} vs local={local_scores[dim]}"
                if winner == "teacher_better"
                else f"local={local_scores[dim]} vs teacher={teacher_scores[dim]}"
            ),
            "evidence": f"score delta = {delta:+.2f}",
            "recommendation": _recommendation_for(dim, winner),
        })

    teacher_hallucinated = (
        teacher_scores["hallucination_risk"] < _TEACHER_HALLUCINATION_FLOOR
    )
    # A teacher that invented facts doesn't get credit for its other wins
    # on this case -- the diff flips to "not better" so the proposal
    # builder doesn't propagate the teacher's bad behavior.
    teacher_better = (teacher_wins > local_wins) and not teacher_hallucinated
    local_better = (local_wins > teacher_wins) or teacher_hallucinated
    mixed = not teacher_better and not local_better

    return {
        "diff_id": new_id("diff"),
        "case_id": case_id,
        "mirror_run_id": mirror_run_id,
        "scores": {"local": local_scores, "teacher": teacher_scores},
        "teacher_better": teacher_better,
        "local_better": local_better,
        "mixed": mixed,
        "teacher_hallucinated": teacher_hallucinated,
        "findings": findings,
        "proposal_candidates": [],
    }
