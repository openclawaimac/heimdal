"""OpenClaw adapter (stub).

OpenClaw sees Heimdal as a single agent and may assign it any role. This
adapter maps an OpenClaw-style task payload into a Host Task Envelope and maps
the Heimdal Result Envelope back into an OpenClaw-style result. It translates
only; it never addresses Heimdal's Router/Worker/Verifier directly
(docs/builder_pack/02_contracts/OPENCLAW_ADAPTER_SPEC.md).
"""

from __future__ import annotations

from heimdal.adapters.base import HostAdapter
from heimdal.ids import new_id


class OpenClawAdapter(HostAdapter):
    host_type = "openclaw"

    def to_host_task_envelope(self, raw_input: dict) -> dict:
        """Map an OpenClaw task payload into a Heimdal Host Task Envelope.

        Expected OpenClaw-style shape::

            {
              "openclaw_task_id": "...",
              "assigned_role": "research",
              "from_agent": "planner",
              "task": {"title": "...", "prompt": "...", "constraints": {...}},
              "policy": {"privacy_mode": "local_only"}
            }
        """
        if not isinstance(raw_input, dict):
            raise ValueError("OpenClawAdapter expects a dict payload.")

        task = raw_input.get("task", {}) or {}
        host_task_id = raw_input.get("openclaw_task_id") or new_id("oc")
        task_id = task.get("id") or host_task_id
        policy = raw_input.get("policy", {}) or {}

        return {
            "host": {
                "type": "openclaw",
                "host_task_id": host_task_id,
                "source_agent": raw_input.get("from_agent"),
                "callback": raw_input.get("callback", {}) or {},
            },
            "role_binding": {
                "role_id": raw_input.get("assigned_role", "general"),
                "risk_mode": policy.get("risk_mode", "balanced"),
                "privacy_mode": policy.get("privacy_mode", "local_only"),
                "output_profiles": task.get("output_profiles", ["markdown"]),
            },
            "task_request": {
                "task_id": task_id,
                "title": task.get("title", "OpenClaw task"),
                "instruction": task.get("prompt", task.get("instruction", "")),
                "inputs": task.get("inputs", {}) or {},
                "constraints": task.get("constraints", {}) or {},
                "priority": task.get("priority", "P2"),
                "budget": task.get("budget", {"quality_level": "B1"}),
                "expected_outputs": task.get("expected_outputs", ["markdown_response"]),
            },
            "runtime_hints": raw_input.get("runtime_hints", {}) or {},
        }

    def from_heimdal_result(self, result: dict) -> dict:
        """Map a Heimdal Result Envelope back into an OpenClaw-style result."""
        return {
            "openclaw_task_id": result.get("host_task_id", ""),
            "heimdal_task_id": result.get("task_id", ""),
            "outcome": result.get("status", "fail"),
            "summary": result.get("message", ""),
            "questions": result.get("questions", []),
            "artifacts": result.get("artifacts", []),
            "repro_pack_ref": result.get("repro_pack", {}).get("path"),
            "trace_pack_ref": result.get("trace_pack", {}).get("path"),
            "metrics": result.get("metrics", {}),
        }
