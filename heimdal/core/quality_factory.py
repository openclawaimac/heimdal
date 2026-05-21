"""Quality Factory.

The internal pipeline that turns a Task Contract into a verified result:

    Contract -> Context Packet -> Router -> Worker Draft -> Verifier
             -> Repair Loop (on FAIL) -> Final Result

See docs/builder_pack/04_runtime/QUALITY_FACTORY.md.
"""

from __future__ import annotations

from heimdal.core import context_os, model_router, verifier


def _worker_prompt_with_defects(prompt: str, defects: list[dict]) -> str:
    if not defects:
        return prompt
    lines = ["", "# PRIOR DEFECTS TO FIX"]
    for defect in defects:
        fix = defect.get("suggested_fix", "")
        lines.append(f"- ({defect['severity']}) {defect['message']} {fix}".rstrip())
    return prompt + "\n" + "\n".join(lines)


def run_quality_factory(contract, role, envelope, backend, storage, config, trace) -> dict:
    """Execute the Quality Factory pipeline for one Work Mode task."""
    trace.event("contract_ready", contract_id=contract["contract_id"])

    packet = context_os.build_packet(contract, role, envelope, storage, config)
    truth_refs = [s["ref"] for s in packet["truth_context"]]
    trace.event(
        "context_packet_ready",
        packet_id=packet["packet_id"],
        truth_refs=truth_refs,
        skills=[s["skill_id"] for s in packet["skills_context"]],
    )

    routing = model_router.route(contract, role, backend, config)
    trace.event("routing", **routing)

    verification = contract.get("verification", {})
    requires_sources = verification.get("requires_sources") or verification.get(
        "requires_citations"
    )

    # -- No-Guess Gate (pre-model) ----------------------------------------
    if verification.get("no_guess_gate") and requires_sources and not packet["truth_context"]:
        trace.event("no_guess_gate", outcome="need_input")
        question = (
            "This task requires grounded sources, but none were found in the local "
            f"Truth Vault for: {contract['objective']!r}. Provide the source "
            "document or reference so Heimdal can answer without guessing."
        )
        return {
            "status": "need_input",
            "output_text": "",
            "questions": [question],
            "packet": packet,
            "routing": routing,
            "verification": {
                "status": "fail",
                "score": 0.0,
                "defects": [
                    {
                        "severity": "critical",
                        "message": "No-Guess Gate: source-required task lacks sources.",
                    }
                ],
                "missing_sources": ["local truth vault has no matching source"],
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

    # -- initial draft (multi-sample for high budgets) --------------------
    best_text = ""
    best_verification = None
    for sample in range(routing["samples"]):
        result = draft(base_prompt, [])
        trace.event(
            "worker_draft", sample=sample, model=result.model, latency_ms=result.latency_ms
        )
        candidate = verifier.verify(result.text, contract, packet, routing, config)
        trace.event(
            "verify", sample=sample, status=candidate["status"], score=candidate["score"]
        )
        if best_verification is None or candidate["score"] > best_verification["score"]:
            best_text, best_verification = result.text, candidate
        if candidate["status"] == "pass":
            break

    # -- repair loop ------------------------------------------------------
    repair_iterations = 0
    while (
        best_verification["status"] == "fail"
        and repair_iterations < routing["max_repair_iterations"]
    ):
        repair_iterations += 1
        result = draft(base_prompt, best_verification["defects"])
        trace.event("repair", iteration=repair_iterations, model=result.model)
        repaired = verifier.verify(result.text, contract, packet, routing, config)
        trace.event(
            "verify", repair=repair_iterations, status=repaired["status"], score=repaired["score"]
        )
        if repaired["score"] >= best_verification["score"]:
            best_text, best_verification = result.text, repaired
        if repaired["status"] == "pass":
            break

    status = "pass" if best_verification["status"] == "pass" else "fail"
    if status == "fail" and best_verification.get("missing_sources"):
        status = "need_input"

    return {
        "status": status,
        "output_text": best_text,
        "questions": [],
        "packet": packet,
        "routing": routing,
        "verification": best_verification,
        "repair_iterations": repair_iterations,
        "models_used": models_used,
    }
