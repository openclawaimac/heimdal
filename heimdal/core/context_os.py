"""Context OS.

Builds the Context Packet: the exact, budgeted context assembly handed to an
internal model call. It enforces token budgets and prevents context bloat
(docs/builder_pack/02_contracts/CONTEXT_PACKET_SPEC.md and
docs/builder_pack/09_storage_context/CONTEXT_OS_REQUIREMENTS.md).

Priority order when trimming to budget: task instruction, truth context,
working state, role context, selected skills, experience examples.
"""

from __future__ import annotations

import json
import os

from heimdal import jsonschema_min
from heimdal.ids import estimate_tokens, new_id, sha256_obj
from heimdal.retrieval.truth_store import TruthStore
from heimdal.skills.selector import SkillSelector

MAX_EXPERIENCE = 2


class ContextError(ValueError):
    """Raised when a built Context Packet fails schema validation."""


def _load_working_state(storage, task_id: str) -> dict:
    path = storage.path("working_state", f"{task_id}.json")
    if os.path.exists(path):
        try:
            return storage.read_json(path)
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _load_experience(storage, query_tokens: set[str]) -> list[dict]:
    directory = storage.path("experience")
    if not os.path.isdir(directory):
        return []
    items: list[dict] = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json"):
            continue
        try:
            record = storage.read_json(os.path.join(directory, name))
        except (OSError, json.JSONDecodeError):
            continue
        tags = set(str(t).lower() for t in record.get("tags", []))
        if tags & query_tokens:
            items.append(record)
    return items[:MAX_EXPERIENCE]


def _packet_tokens(packet: dict) -> int:
    return estimate_tokens(json.dumps(packet, default=str))


def build_packet(contract: dict, role: dict, envelope: dict, storage, config) -> dict:
    """Assemble, budget-trim, hash, and validate a Context Packet."""
    task_request = envelope.get("task_request", {}) or {}
    instruction = contract.get("objective", "")

    truth_store = TruthStore(storage.path("truth"))
    truth_hits = truth_store.retrieve(instruction)
    truth_context = [
        {"ref": hit.ref, "text": hit.text, "score": hit.score} for hit in truth_hits
    ]

    selector = SkillSelector(storage.path("skills"))
    skill_cards = selector.select(role.get("skills", []), instruction)
    skills_context = [
        {"skill_id": card.skill_id, "guidance": card.guidance} for card in skill_cards
    ]

    query_tokens = set(instruction.lower().split())
    experience_context = _load_experience(storage, query_tokens)

    packet = {
        "packet_id": new_id("packet"),
        "contract_id": contract["contract_id"],
        "role_context": {
            "role_id": role.get("role_id"),
            "system_context": role.get("system_context"),
            "risk_mode": role.get("risk_mode"),
            "output_profiles": role.get("output_profiles", []),
        },
        "truth_context": truth_context,
        "working_state": _load_working_state(storage, contract["task_id"]),
        "task_context": {
            "instruction": instruction,
            "title": task_request.get("title", ""),
            "constraints": contract.get("constraints", {}),
            "expected_outputs": contract.get("expected_outputs", []),
        },
        "experience_context": experience_context,
        "skills_context": skills_context,
        "budget": dict(contract.get("budget", {})),
        "hashes": {},
    }

    _enforce_budget(packet)

    packet["hashes"] = {
        "role_context": sha256_obj(packet["role_context"]),
        "truth_context": sha256_obj(packet["truth_context"]),
        "task_context": sha256_obj(packet["task_context"]),
        "skills_context": sha256_obj(packet["skills_context"]),
        "packet": "",
    }
    packet["hashes"]["packet"] = sha256_obj(
        {k: v for k, v in packet.items() if k != "hashes"}
    )

    jsonschema_min.validate_or_raise(
        packet,
        config.schema_path("context_packet.schema.json"),
        "Context Packet",
        ContextError,
    )
    return packet


def _enforce_budget(packet: dict) -> None:
    """Trim low-priority sections until the packet fits max_input_tokens."""
    max_tokens = packet["budget"].get("max_input_tokens", 8000)

    # Priority (lowest first to drop): experience, extra skills, truncate truth.
    while _packet_tokens(packet) > max_tokens and packet["experience_context"]:
        packet["experience_context"].pop()
    while _packet_tokens(packet) > max_tokens and len(packet["skills_context"]) > 1:
        packet["skills_context"].pop()
    if _packet_tokens(packet) > max_tokens:
        for snippet in packet["truth_context"]:
            words = snippet["text"].split()
            if len(words) > 80:
                snippet["text"] = " ".join(words[:80]) + " ..."
    while _packet_tokens(packet) > max_tokens and len(packet["truth_context"]) > 1:
        packet["truth_context"].pop()


def render_worker_input(packet: dict, role: dict) -> tuple[str, dict]:
    """Render a packet into a (prompt, structured) pair for a model backend."""
    task = packet["task_context"]
    lines = [
        "# ROLE",
        packet["role_context"].get("system_context", ""),
        "",
        "# TASK",
        task.get("instruction", ""),
    ]
    if task.get("constraints"):
        lines += ["", "# CONSTRAINTS", json.dumps(task["constraints"])]
    if packet["truth_context"]:
        lines += ["", "# TRUTH CONTEXT"]
        for index, snippet in enumerate(packet["truth_context"], start=1):
            lines.append(f"[{index}] ({snippet['ref']}) {snippet['text']}")
    if packet["skills_context"]:
        lines += ["", "# SKILLS"]
        lines += [f"- {card['guidance']}" for card in packet["skills_context"]]
    profiles = packet["role_context"].get("output_profiles", ["markdown"])
    lines += ["", "# OUTPUT", f"Produce a response as: {', '.join(profiles)}."]

    constraints = task.get("constraints", {}) or {}
    structured = {
        "instruction": task.get("instruction", ""),
        "title": task.get("title") or "Heimdal Response",
        "truth": [s["text"] for s in packet["truth_context"]],
        "output_profile": profiles[0] if profiles else "markdown",
        "max_words": constraints.get("max_words"),
    }
    return "\n".join(lines), structured
