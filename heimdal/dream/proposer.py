"""Turn failure patterns into structured improvement proposals.

Each proposal is a review-only artifact that the patch promotion pipeline can
later evaluate. The proposer never mutates stable behavior; it only writes
JSON files into ``storage/dream/proposals/``.
"""

from __future__ import annotations

from heimdal.ids import new_id, now_iso


def _proposal(kind: str, *, intent: str, rationale: str, risk_level: str,
              evidence: list[dict], **inner) -> dict:
    proposal = {
        "id": new_id("proposal"),
        "kind": kind,
        "intent": intent,
        "rationale": rationale,
        "created_by": "dream_mode",
        "created_at": now_iso(),
        "risk_level": risk_level,
        "evidence": evidence,
        "status": "experimental",
    }
    proposal.update(inner)
    return proposal


def _patch(target: str, change: dict, patch_type: str) -> dict:
    return {
        "id": new_id("patch"),
        "type": patch_type,
        "channel": "experimental",
        "target": target,
        "change": change,
        "rationale": "Dream Mode proposal -- review before promotion.",
        "created_at": now_iso(),
        "eval_run": None,
        "source": "dream_mode",
    }


def _skill_proposal_payload(skill_id: str, role: str, title: str,
                            instructions: list[str], triggers: list[str]) -> dict:
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


def _eval_case_payload(case_id: str, role_id: str, instruction: str, *,
                       quality_level: str = "B2", expect_status: str = "need_input",
                       constraints: dict | None = None) -> dict:
    case: dict = {
        "id": case_id,
        "instruction": instruction,
        "role_id": role_id,
        "quality_level": quality_level,
        "expect_status": expect_status,
    }
    if constraints:
        case["constraints"] = constraints
    return case


# -- per-category builders -------------------------------------------------
def _propose_for_missing_source(pattern: dict) -> list[dict]:
    examples = pattern.get("examples", [])
    proposals: list[dict] = []
    proposals.append(_proposal(
        "eval_case_proposal",
        intent="Add a no-guess regression case for a recurring missing-source topic.",
        rationale=(
            f"Mined {pattern['count']} task(s) hitting the No-Guess Gate with no "
            "usable source. A fresh eval case keeps this regression visible."
        ),
        risk_level="low",
        evidence=examples,
        eval_case=_eval_case_payload(
            f"dream_no_guess_{new_id('case').rsplit('_', 1)[-1][:6]}",
            role_id="research",
            instruction=(
                "State the documented policy for the topic described in "
                f"{examples[0]['task_id']!r}." if examples
                else "State an undocumented vendor policy not in the Truth Vault."
            ),
            constraints={"requires_sources": True},
        ),
    ))
    return proposals


def _propose_for_weak_retrieval(pattern: dict) -> list[dict]:
    return [_proposal(
        "patch_proposal",
        intent="Tighten the No-Guess grounding-coverage threshold.",
        rationale=(
            f"{pattern['count']} task(s) retrieved a snippet that shared only a "
            "couple of generic words with the objective. Consider raising "
            "retrieval.min_grounding_coverage so weak overlaps fail closed."
        ),
        risk_level="medium",
        evidence=pattern.get("examples", []),
        patch=_patch(
            target="config.retrieval.min_grounding_coverage",
            change={"set": 0.6, "previous": 0.5},
            patch_type="retrieval_patch",
        ),
    )]


def _propose_for_semantic_miss(pattern: dict) -> list[dict]:
    proposals: list[dict] = []
    proposals.append(_proposal(
        "skill_proposal",
        intent="Sharpen the semantic-quality-check skill for short hedge answers.",
        rationale=(
            f"{pattern['count']} answer(s) were flagged as semantic misses. A "
            "skill that explicitly bans hedge phrasing on B2+ tasks could "
            "raise pass rate without weakening the gate."
        ),
        risk_level="low",
        evidence=pattern.get("examples", []),
        skill=_skill_proposal_payload(
            "verifier.no_hedge_answer",
            role="verifier",
            title="Reject hedge answers",
            triggers=["hedge", "i think", "may", "might", "uncertain"],
            instructions=[
                "If the answer is a question back to the user, mark it as not "
                "fulfilling the task.",
                "Hedge words without a cited source count as unsupported claims.",
            ],
        ),
    ))
    return proposals


def _propose_for_schema_failure(pattern: dict) -> list[dict]:
    return [_proposal(
        "patch_proposal",
        intent="Add an explicit JSON-shape reminder to schema-required prompts.",
        rationale=(
            f"{pattern['count']} task(s) failed schema validation. A prompt "
            "patch that restates the required JSON shape close to the answer "
            "boundary is a low-risk way to lift schema pass rate."
        ),
        risk_level="low",
        evidence=pattern.get("examples", []),
        patch=_patch(
            target="role_pack:general:system_context",
            change={"append": "When asked for JSON, return ONLY a JSON object matching the schema."},
            patch_type="prompt_patch",
        ),
    )]


def _propose_for_adapter_issue(pattern: dict) -> list[dict]:
    return [_proposal(
        "patch_proposal",
        intent="Surface unsupported-adapter errors with a clearer host-side hint.",
        rationale=(
            f"{pattern['count']} bridge job(s) named an unsupported adapter. A "
            "small message improvement helps host authors recover faster."
        ),
        risk_level="low",
        evidence=pattern.get("examples", []),
        patch=_patch(
            target="heimdal.bridge:SUPPORTED_ADAPTERS",
            change={"note": "Add adapter to documentation; no code change."},
            patch_type="prompt_patch",
        ),
    )]


def _propose_for_backend_error(pattern: dict) -> list[dict]:
    return [_proposal(
        "patch_proposal",
        intent="Surface Ollama outages earlier in the bridge cycle.",
        rationale=(
            f"{pattern['count']} bridge run(s) failed at backend setup. A "
            "scheduler patch could throttle retries while Ollama is down."
        ),
        risk_level="medium",
        evidence=pattern.get("examples", []),
        patch=_patch(
            target="config.scheduler.work_preempts_background",
            change={"note": "Consider adding ollama-backoff scheduler entry."},
            patch_type="scheduler_patch",
        ),
    )]


# Dispatcher: pattern category -> proposal builders.
_BUILDERS = {
    "missing_source": _propose_for_missing_source,
    "weak_retrieval": _propose_for_weak_retrieval,
    "semantic_miss": _propose_for_semantic_miss,
    "schema_failure": _propose_for_schema_failure,
    "adapter_mapping_issue": _propose_for_adapter_issue,
    "timeout_or_backend_error": _propose_for_backend_error,
}


def generate_proposals(patterns: list[dict]) -> list[dict]:
    """Produce one or more structured proposals per mined pattern."""
    proposals: list[dict] = []
    for pattern in patterns:
        builder = _BUILDERS.get(pattern["category"])
        if builder:
            proposals.extend(builder(pattern))
    return proposals


def synthetic_proposal() -> dict:
    """A baseline exploratory proposal used when no failures are mined.

    Keeps Dream Mode honest: every run produces at least one artifact so the
    user can see Dream Mode actually executed.
    """
    return _proposal(
        "skill_proposal",
        intent="Synthetic baseline -- explore adding a follow-up question skill.",
        rationale=(
            "No failure patterns were mined in this run. A skill that prompts "
            "the worker to surface a one-line follow-up question when "
            "confidence is low is a safe, low-risk addition to explore."
        ),
        risk_level="low",
        evidence=[],
        skill=_skill_proposal_payload(
            "general.surface_followup",
            role="general",
            title="Surface one follow-up question on low confidence",
            triggers=["uncertain", "low confidence", "maybe", "unsure"],
            instructions=[
                "When the answer leans on assumption, end with one concise "
                "question the host can answer to remove the assumption.",
            ],
        ),
    )


def categorize_actions(patterns: list[dict], proposals: list[dict]) -> tuple[list[str], list[str]]:
    """Build human-facing recommended-actions and risk-notes lists."""
    actions: list[str] = []
    risks: list[str] = []
    if not patterns:
        actions.append(
            "No failure patterns in the mined window; consider running with "
            "--source eval or letting more workloads accumulate."
        )
    for pattern in patterns:
        if pattern["category"] == "missing_source":
            actions.append(
                "Add the missing Truth Vault sources for the highlighted topics."
            )
        if pattern["category"] == "weak_retrieval":
            actions.append(
                "Review retrieval.min_grounding_coverage in heimdal.manifest.yml."
            )
        if pattern["category"] == "semantic_miss":
            actions.append(
                "Review semantic verifier model and prompt; a hybrid model "
                "switch may help."
            )
    for proposal in proposals:
        if proposal.get("risk_level") in ("medium", "high"):
            risks.append(
                f"{proposal['id']} is {proposal['risk_level']}-risk -- evaluate "
                "before promotion."
            )
    return actions, risks
