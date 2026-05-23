"""Turn frontier diff findings into improvement proposals.

The proposal builder is conservative: it never emits anything when the
diff says the teacher is not better or the teacher hallucinated. When the
teacher genuinely beat local on a dimension, the builder maps it to one
of the existing proposal kinds (patch_proposal / skill_proposal /
eval_case_proposal) so the patch lifecycle and Skill Library 2.0 already
know what to do with it.

Each emitted patch carries the v0.4.1-required metadata (``intent``,
``rollback`` review-only marker, ``risk_level``, ``created_by:
mirror_mode``) so ``heimdal patch review/eval/promote --to beta`` works
on it without manual cleanup.
"""

from __future__ import annotations

from heimdal.ids import new_id, now_iso

# Dimension -> proposal kind. Dimensions not in this table do not produce
# proposals (some are diagnostic only).
_DIMENSION_PROPOSAL = {
    # patch_proposal (rubric / prompt) ---------------------------------
    "task_adherence": ("patch_proposal", "prompt_patch"),
    "factuality": ("patch_proposal", "rubric_patch"),
    "source_grounding": ("patch_proposal", "rubric_patch"),
    "missing_caveats": ("patch_proposal", "rubric_patch"),
    "no_guess_behavior": ("patch_proposal", "rubric_patch"),
    # skill_proposal ---------------------------------------------------
    "structure_format": ("skill_proposal", None),
    "completeness": ("skill_proposal", None),
    "reasoning_depth": ("skill_proposal", None),
    "actionability": ("skill_proposal", None),
    "conciseness": ("skill_proposal", None),
    "uncertainty_handling": ("skill_proposal", None),
}

_DIMENSION_SKILL_TEMPLATE = {
    "structure_format": (
        "skill.mirror.structured_answer",
        ["structure", "format", "outline"],
        ["Lead with a one-sentence answer.",
         "Use short headers + bullets for the body.",
         "Close with a single caveat or next step."],
    ),
    "completeness": (
        "skill.mirror.complete_coverage",
        ["coverage", "missing", "section"],
        ["Enumerate the parts of the task before answering.",
         "Mark any part you cannot answer rather than skipping it."],
    ),
    "reasoning_depth": (
        "skill.mirror.explicit_reasoning",
        ["because", "therefore", "since"],
        ["Explain why each step follows from the previous.",
         "When a claim depends on an assumption, name the assumption."],
    ),
    "actionability": (
        "skill.mirror.action_oriented",
        ["next step", "configure", "run", "check"],
        ["End with one concrete next step the user can take.",
         "Prefer verbs ('run', 'set', 'check') over abstract advice."],
    ),
    "conciseness": (
        "skill.mirror.concise_mode",
        ["short", "brief", "tight"],
        ["Cut filler and qualifiers.",
         "Respect the task's max_words constraint."],
    ),
    "uncertainty_handling": (
        "skill.mirror.uncertainty_flag",
        ["may", "verify", "assumption"],
        ["Mark any assumption you make.",
         "Suggest one specific check the user can run to verify."],
    ),
}


def _patch(*, target: str, change: dict, patch_type: str, intent: str,
           risk_level: str) -> dict:
    return {
        "id": new_id("patch"),
        "type": patch_type,
        "channel": "experimental",
        "target": target,
        "change": change,
        "rationale": "Frontier Diff Engine proposal -- review before promotion.",
        "intent": intent,
        "risk_level": risk_level,
        "created_by": "mirror_mode",
        "created_at": now_iso(),
        "eval_run": None,
        "source": "mirror_mode",
        "rollback": {
            "note": (
                "Mirror-generated; revert by removing the change or "
                "reversing the keys under 'change'."
            ),
            "review_only": True,
        },
    }


def _skill_payload(skill_id: str, role: str, triggers: list[str],
                   instructions: list[str], *, title: str) -> dict:
    return {
        "id": skill_id,
        "version": "0.1.0",
        "role": role,
        "title": title,
        "description": instructions[0] if instructions else title,
        "triggers": triggers,
        "instructions": instructions,
        "performance": {"uses": 0, "passes": 0, "fails": 0, "last_used": None},
    }


def _wrap(kind: str, *, intent: str, rationale: str, risk_level: str,
          finding: dict, **inner) -> dict:
    proposal = {
        "id": new_id("proposal"),
        "kind": kind,
        "intent": intent,
        "rationale": rationale,
        "created_by": "mirror_mode",
        "created_at": now_iso(),
        "risk_level": risk_level,
        "evidence": [finding],
        "status": "experimental",
    }
    proposal.update(inner)
    return proposal


def build_proposals(diff: dict, *, case: dict) -> list[dict]:
    """Convert findings into experimental proposals; empty when not applicable."""
    if diff.get("teacher_hallucinated"):
        return []
    if not diff.get("teacher_better"):
        return []

    role = (case.get("task") or {}).get("role_id", "general")
    proposals: list[dict] = []
    requires_sources = bool(
        (case.get("task") or {}).get("constraints", {}).get("requires_sources")
    )

    for finding in diff["findings"]:
        dim = finding["dimension"]
        # Only act on teacher_better findings (severity in {medium, high}).
        if "teacher=" not in finding["finding"]:
            continue
        if finding.get("severity") == "low":
            continue
        kind_patch_type = _DIMENSION_PROPOSAL.get(dim)
        if kind_patch_type is None:
            continue
        kind, patch_type = kind_patch_type

        risk = "high" if finding["severity"] == "high" else "medium"
        intent = finding["recommendation"]

        if kind == "patch_proposal":
            target = f"role_pack:{role}:{dim}"
            change = {"note": intent}
            proposal = _wrap(
                kind, intent=intent,
                rationale=f"Diff finding on {dim}: {finding['finding']}",
                risk_level=risk, finding=finding,
                patch=_patch(target=target, change=change,
                             patch_type=patch_type, intent=intent,
                             risk_level=risk),
            )
            proposals.append(proposal)
        elif kind == "skill_proposal":
            template = _DIMENSION_SKILL_TEMPLATE.get(dim)
            if template is None:
                continue
            skill_id, triggers, instructions = template
            proposal = _wrap(
                kind, intent=intent,
                rationale=f"Diff finding on {dim}: {finding['finding']}",
                risk_level=risk, finding=finding,
                skill=_skill_payload(
                    skill_id, role, triggers, instructions, title=intent,
                ),
            )
            proposals.append(proposal)

    # If the case is source-required and the teacher beat local on
    # source_grounding, also propose an eval case so the regression
    # stays locked in even after the rubric patch is reviewed.
    if requires_sources and any(
        f["dimension"] == "source_grounding" and "teacher=" in f["finding"]
        for f in diff["findings"]
    ):
        case_id = case.get("case_id", "case")
        objective = (case.get("task") or {}).get("objective", "")
        proposals.append(_wrap(
            "eval_case_proposal",
            intent="Lock the source_grounding gap with a regression case.",
            rationale=(
                "Diff showed teacher cited sources where local did not on "
                f"case {case_id!r}."
            ),
            risk_level="low",
            finding={"dimension": "source_grounding", "severity": "medium",
                     "finding": "regression case proposed"},
            eval_case={
                "id": f"mirror_grounding_{new_id('case').rsplit('_', 1)[-1][:6]}",
                "instruction": objective,
                "role_id": role,
                "quality_level": "B2",
                "constraints": {"requires_sources": True},
                "expect_status": "need_input",
            },
        ))
    return proposals
