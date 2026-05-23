"""Quality Factory.

The internal pipeline that turns a Task Contract into a verified result:

    Contract -> Context Packet -> Router -> Worker Draft -> Verifier
             -> Repair Loop (on FAIL) -> Final Result

See docs/builder_pack/04_runtime/QUALITY_FACTORY.md.
"""

from __future__ import annotations

import re

from heimdal.core import context_os, model_router, status_codes, verifier
from heimdal.core.constants import FAIL, HYBRID, NEED_INPUT, PASS
from heimdal.core.task_contract import requires_grounding
from heimdal.retrieval import truth_store


_TOPIC_LEADING_VERBS = {
    "state", "explain", "describe", "summarize", "give", "list", "write",
    "provide", "report", "tell", "specify", "outline", "define", "detail",
    "quote", "cite",
}
_TOPIC_LEADING_FILLER = {
    "the", "a", "an", "exact", "precise", "published", "guaranteed",
    "reported", "offered", "quarterly", "official",
}
_TOPIC_META_HEAD = {"using", "with", "from", "given", "per", "according"}

_TOPIC_PIVOT_RE = re.compile(r"^(.*?)\s+(?:for|of)\s+(.+)$", re.IGNORECASE)


def _missing_topic(instruction: str) -> str:
    """Distill a concise topic from a source-required instruction.

    Deterministic, no model: strips a leading retrieval-meta clause ("Using
    the Truth Vault, ..."), the leading action verb, generic filler, and
    rotates a "<thing> for/of <X>" sentence into "<X> <thing>" so the topic
    leads with the entity the host needs to supply. Returns the cleaned
    instruction unchanged when no shortening is possible.
    """
    text = instruction.strip().rstrip(".").strip()
    if not text:
        return ""
    if "," in text:
        head, _, tail = text.partition(",")
        first = (head.strip().split() or [""])[0].lower()
        if first in _TOPIC_META_HEAD:
            text = tail.strip()
    words = text.split()
    if words and words[0].lower() in _TOPIC_LEADING_VERBS:
        words = words[1:]
    while words and words[0].lower() in _TOPIC_LEADING_FILLER:
        words = words[1:]
    text = " ".join(words)
    match = _TOPIC_PIVOT_RE.match(text)
    if match:
        head, tail = match.group(1).strip(), match.group(2).strip()
        tail_words = tail.split()
        while tail_words and tail_words[0].lower() in _TOPIC_LEADING_FILLER:
            tail_words = tail_words[1:]
        text = f"{' '.join(tail_words)} {head}".strip()
    return text or instruction.strip().rstrip(".")


def _needed_input(objective: str, reason: str) -> dict:
    """A single structured needed-input entry for a need_input result."""
    return {
        "type": "source_document",
        "reason": reason,
        "missing_topic": _missing_topic(objective),
        "suggested_action": "Provide a source document or allow retrieval.",
    }


def _worker_prompt_with_defects(prompt: str, defects: list[dict]) -> str:
    if not defects:
        return prompt
    lines = ["", "# PRIOR DEFECTS TO FIX"]
    for defect in defects:
        fix = defect.get("suggested_fix", "")
        lines.append(f"- ({defect['severity']}) {defect['message']} {fix}".rstrip())
    return prompt + "\n" + "\n".join(lines)


def run_quality_factory(
    contract,
    role,
    envelope,
    backend,
    storage,
    config,
    trace,
    model_override=None,
    verifier_override=None,
) -> dict:
    """Execute the Quality Factory pipeline for one Work Mode task."""
    trace.event("contract_ready", contract_id=contract["contract_id"])

    packet = context_os.build_packet(contract, role, envelope, storage, config)
    selected_skills_refs = [
        {
            "skill_id": s["skill_id"],
            "source": s.get("source", "registry"),
            "role": role.get("role_id"),
        }
        for s in packet["skills_context"]
    ]
    trace.event(
        "context_packet_ready",
        packet_id=packet["packet_id"],
        truth_refs=context_os.retrieval_refs(packet),
        skills=[s["skill_id"] for s in packet["skills_context"]],
    )
    # Dedicated event so external tools can find selected skills without
    # parsing the packet-ready event. Only the skill IDs / source / role go
    # in the trace -- full guidance content stays in the Context Packet
    # artifact (which is internal-only).
    trace.event("skill_selection", selected_skills=selected_skills_refs)

    routing = model_router.route(
        contract, role, backend, config, model_override, verifier_override
    )
    trace.event("routing", **routing)

    verification = contract.get("verification", {})

    # No-Guess Gate: stop before the model call when a source-required task
    # lacks sufficient grounding, and return need_input rather than guessing.
    # "Sufficient" means more than an incidental keyword overlap -- a retrieved
    # snippet must cover a real share of the task's content terms -- so a task
    # whose source is missing returns need_input here, before the semantic
    # verifier could mistake the model's honest hedge for a verification fail.
    needs_sources = verification.get("no_guess_gate") and requires_grounding(verification)
    if needs_sources:
        min_coverage = config.retrieval.get("min_grounding_coverage", 0.5)
        coverage = truth_store.grounding_coverage(
            contract["objective"], packet["truth_context"]
        )
        if not packet["truth_context"] or coverage < min_coverage:
            if not packet["truth_context"]:
                reason = "no truth sources retrieved for a source-required task"
                code = status_codes.SOURCE_MISSING
            else:
                reason = (
                    f"retrieved sources cover only {coverage:.0%} of the task's "
                    f"key terms (minimum {min_coverage:.0%})"
                )
                code = status_codes.SOURCE_SUPPORT_INSUFFICIENT
            trace.event(
                "no_guess_gate",
                outcome=NEED_INPUT,
                code=code,
                reason=reason,
                retrieval_refs=context_os.retrieval_refs(packet),
            )
            question = (
                "This task requires grounded sources, but the local Truth Vault "
                f"has no sufficient source for: {contract['objective']!r}. "
                "Provide the source document or reference so Heimdal can answer "
                "without guessing."
            )
            return {
                "status": NEED_INPUT,
                "code": code,
                "needed_inputs": [_needed_input(contract["objective"], reason)],
                "output_text": "",
                "questions": [question],
                "packet": packet,
                "routing": routing,
                "verification": {
                    "status": FAIL,
                    "score": 0.0,
                    "defects": [
                        {
                            "severity": "critical",
                            "message": "No-Guess Gate: source-required task lacks "
                            "sufficient grounding.",
                        }
                    ],
                    "missing_sources": [reason],
                    "schema_errors": [],
                },
                "repair_iterations": 0,
                "models_used": [],
            }

    base_prompt, structured = context_os.render_worker_input(packet, role)
    json_mode = structured.get("output_profile") == "json" or verification.get(
        "requires_schema_validation", False
    )
    max_output = contract.get("budget", {}).get("max_output_tokens", 2000)
    models_used: list[dict] = []

    def draft(prompt: str, defects: list[dict]):
        result = backend.generate(
            _worker_prompt_with_defects(prompt, defects),
            model=routing["worker_model"],
            system=role.get("system_context", ""),
            json_mode=json_mode,
            max_tokens=max_output,
            temperature=0.2,
            structured={**structured, "defects": defects},
        )
        models_used.append(
            {"role": "worker", "model": result.model, "backend": result.backend}
        )
        return result

    def check(text: str, **trace_kw):
        result = verifier.verify(text, contract, packet, routing, config, backend)
        semantic = result.get("semantic")
        if semantic is not None:
            trace.event(
                "semantic_verify",
                semantic_verifier_model=semantic["model"],
                semantic_verifier_status=semantic["status"],
                semantic_verifier_score=semantic["score"],
                semantic_verifier_confidence=semantic["confidence"],
            )
        trace.event("verify", status=result["status"], score=result["score"], **trace_kw)
        return result

    # Route the backend's request events into this run's Trace Pack.
    backend.event_sink = trace.event
    try:
        # Initial draft; high budgets (B3/B4) take the best of multiple samples.
        best_text = ""
        best_verification = None
        for sample in range(routing["samples"]):
            result = draft(base_prompt, [])
            trace.event(
                "worker_draft", sample=sample, model=result.model, latency_ms=result.latency_ms
            )
            candidate = check(result.text, sample=sample)
            if best_verification is None or candidate["score"] > best_verification["score"]:
                best_text, best_verification = result.text, candidate
            if candidate["status"] == PASS:
                break

        repair_iterations = 0
        while (
            best_verification["status"] == FAIL
            and repair_iterations < routing["max_repair_iterations"]
        ):
            repair_iterations += 1
            result = draft(base_prompt, best_verification["defects"])
            trace.event("repair", iteration=repair_iterations, model=result.model)
            repaired = check(result.text, repair=repair_iterations)
            if repaired["score"] >= best_verification["score"]:
                best_text, best_verification = result.text, repaired
            if repaired["status"] == PASS:
                break
    finally:
        backend.event_sink = None

    if routing["verifier_backend"] == HYBRID:
        models_used.append(
            {
                "role": "semantic_verifier",
                "model": routing["semantic_verifier_model"],
                "backend": backend.name,
            }
        )

    status = PASS if best_verification["status"] == PASS else FAIL
    if status == FAIL and best_verification.get("missing_sources"):
        status = NEED_INPUT

    if status == PASS:
        code, questions, needed_inputs = status_codes.OK, [], []
    elif status == NEED_INPUT:
        reason = "; ".join(best_verification.get("missing_sources", [])) or (
            "source-required task lacks sufficient grounding"
        )
        code = status_codes.SOURCE_MISSING
        questions = [
            "This task requires grounded sources that Heimdal could not "
            f"retrieve: {reason}. Provide the source document or reference."
        ]
        needed_inputs = [_needed_input(contract["objective"], reason)]
    else:
        code, questions, needed_inputs = status_codes.fail_code(best_verification), [], []

    return {
        "status": status,
        "code": code,
        "needed_inputs": needed_inputs,
        "output_text": best_text,
        "questions": questions,
        "packet": packet,
        "routing": routing,
        "verification": best_verification,
        "repair_iterations": repair_iterations,
        "models_used": models_used,
    }
