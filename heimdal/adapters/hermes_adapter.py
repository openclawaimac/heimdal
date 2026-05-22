"""Hermes adapter (translation only).

Heimdal is usable with NousResearch/Hermes-style persistent agents without
depending on Hermes. Hermes owns the long-term social memory and interaction
loop; it hands a task to Heimdal as a single external skill/sub-agent, and
Heimdal runs the full Quality Factory internally
(docs/builder_pack/02_contracts/HERMES_ADAPTER_SPEC.md).

This adapter only translates a Hermes payload into a Heimdal Host Task Envelope
and a Heimdal Result Envelope back into a Hermes-style result. It never
orchestrates; orchestration lives in heimdal/adapters/hermes_host.py.
"""

from __future__ import annotations

from heimdal.adapters.base import HostAdapter
from heimdal.ids import new_id


class HermesAdapter(HostAdapter):
    host_type = "hermes"

    def to_host_task_envelope(self, raw_input: dict) -> dict:
        """Map a Hermes skill-invocation payload into a Host Task Envelope.

        Expected Hermes-style shape::

            {
              "hermes_session_id": "...",
              "invocation_id": "...",
              "from_agent": "Hermes",
              "role": "research",
              "request": {"id": "...", "title": "...", "instruction": "...",
                          "constraints": {...}, "budget": {...},
                          "expected_outputs": [...]},
              "policy": {"privacy_mode": "local_only", "risk_mode": "balanced"},
              "callback": {"file": "..."}
            }
        """
        if not isinstance(raw_input, dict):
            raise ValueError("HermesAdapter expects a dict payload.")

        request = raw_input.get("request", {}) or {}
        session_id = raw_input.get("hermes_session_id") or new_id("hermes")
        invocation_id = raw_input.get("invocation_id") or session_id
        task_id = request.get("id") or invocation_id
        policy = raw_input.get("policy", {}) or {}

        return {
            "host": {
                "type": "hermes",
                "host_task_id": invocation_id,
                "source_agent": raw_input.get("from_agent", "Hermes"),
                "callback": raw_input.get("callback", {}) or {},
            },
            "role_binding": {
                "role_id": raw_input.get("role", "general"),
                "risk_mode": policy.get("risk_mode", "balanced"),
                "privacy_mode": policy.get("privacy_mode", "local_only"),
                "output_profiles": request.get("output_profiles", ["markdown"]),
            },
            "task_request": {
                "task_id": task_id,
                "title": request.get("title", "Hermes task"),
                "instruction": request.get("instruction", request.get("prompt", "")),
                "inputs": request.get("inputs", {}) or {},
                "constraints": request.get("constraints", {}) or {},
                "priority": request.get("priority", "P2"),
                "budget": request.get("budget", {"quality_level": "B1"}),
                "expected_outputs": request.get("expected_outputs", ["markdown_response"]),
            },
            "runtime_hints": {"hermes_session_id": session_id},
        }

    def from_heimdal_result(self, result: dict) -> dict:
        """Map a Heimdal Result Envelope into a Hermes-style result.

        Presents Heimdal as one agent: status, message, artifacts, pack
        references and verifier metadata only -- never the internal sub-agent
        graph (router/worker/repair).
        """
        metrics = result.get("metrics", {}) or {}
        return {
            "invocation_id": result.get("host_task_id", ""),
            "heimdal_task_id": result.get("task_id", ""),
            "status": result.get("status", "fail"),
            "message": result.get("message", ""),
            "questions": result.get("questions", []),
            "artifacts": result.get("artifacts", []),
            "repro_pack_ref": result.get("repro_pack", {}).get("path"),
            "trace_pack_ref": result.get("trace_pack", {}).get("path"),
            "verifier": {
                "backend": metrics.get("verifier_backend"),
                "semantic_model": metrics.get("semantic_verifier_model"),
            },
            "metrics": metrics,
        }
