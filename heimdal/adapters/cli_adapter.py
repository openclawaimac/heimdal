"""CLI adapter.

The simplest host adapter: it accepts either a full Host Task Envelope (loaded
from a JSON file) or a plain instruction string, and renders results for a
terminal (docs/builder_pack/05_adapters/CLI_ADAPTER.md).
"""

from __future__ import annotations

from heimdal.adapters.base import HostAdapter
from heimdal.ids import new_id


class CLIAdapter(HostAdapter):
    host_type = "cli"

    def to_host_task_envelope(self, raw_input, *, role: str | None = None) -> dict:
        """Build a Host Task Envelope from a CLI input.

        ``role`` (when given alongside an instruction string) sets the role
        binding's ``role_id``, so ``heimdal run --instruction "..." --role
        research`` reaches the research role pack and its skill candidates.
        """
        if isinstance(raw_input, dict) and "host" in raw_input and "task_request" in raw_input:
            return raw_input  # already a Host Task Envelope

        if isinstance(raw_input, str):
            task_id = new_id("task")
            role_id = role or "general"
            return {
                "host": {
                    "type": "cli",
                    "host_task_id": task_id,
                    "source_agent": None,
                    "callback": {},
                },
                "role_binding": {
                    "role_id": role_id,
                    "risk_mode": "balanced",
                    "privacy_mode": "local_only",
                    "output_profiles": ["markdown"],
                },
                "task_request": {
                    "task_id": task_id,
                    "title": "CLI task",
                    "instruction": raw_input,
                    "inputs": {},
                    "constraints": {},
                    "priority": "P2",
                    "budget": {"quality_level": "B1"},
                    "expected_outputs": ["markdown_response"],
                },
                "runtime_hints": {},
            }
        raise ValueError("CLIAdapter expects a Host Task Envelope dict or an instruction string.")

    def from_heimdal_result(self, result: dict) -> str:
        lines = [
            f"status : {result['status']}",
            f"task   : {result['task_id']}",
            f"message: {result['message']}",
        ]
        if result.get("questions"):
            lines.append("questions:")
            lines += [f"  - {q}" for q in result["questions"]]
        if result.get("artifacts"):
            lines.append("artifacts:")
            lines += [f"  - {a['type']}: {a['path']}" for a in result["artifacts"]]
        repro = result.get("repro_pack", {})
        trace = result.get("trace_pack", {})
        if repro.get("path"):
            lines.append(f"repro_pack: {repro['path']}")
        if trace.get("path"):
            lines.append(f"trace_pack: {trace['path']}")
        return "\n".join(lines)
