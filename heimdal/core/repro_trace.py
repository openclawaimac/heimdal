"""Repro Pack and Trace Pack logging.

Every run writes a Repro Pack (what it took to reproduce the run) and a Trace
Pack (the ordered events of the run). Schemas live in docs/builder_pack/03_schemas.
"""

from __future__ import annotations

import platform

from heimdal import __version__, jsonschema_min
from heimdal.ids import new_id, now_iso


class TraceBuilder:
    """Accumulates ordered events for a single run."""

    def __init__(self, task_id: str):
        self.task_id = task_id
        self.trace_id = new_id("trace")
        self.events: list[dict] = []

    def event(self, name: str, **data) -> None:
        self.events.append({"ts": now_iso(), "name": name, "data": data})

    def build(self, status: str, metrics: dict) -> dict:
        return {
            "id": self.trace_id,
            "task_id": self.task_id,
            "events": self.events,
            "metrics": metrics,
            "status": status,
        }


def build_repro_pack(
    *,
    models: list[dict],
    params: dict,
    hashes: dict,
    hardware_profile: dict | None = None,
    retrieval_refs: list[str] | None = None,
) -> dict:
    return {
        "id": new_id("repro"),
        "timestamp": now_iso(),
        "models": models,
        "params": params,
        "hashes": hashes,
        "versions": {
            "heimdal": __version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "hardware_profile": hardware_profile or {},
        "retrieval_refs": retrieval_refs or [],
    }


def write_packs(storage, config, repro: dict, trace: dict) -> dict:
    """Validate and persist a Repro Pack and Trace Pack; return their paths."""
    jsonschema_min.validate_or_raise(
        repro, config.schema_path("repro_pack.schema.json"), "Repro Pack"
    )
    jsonschema_min.validate_or_raise(
        trace, config.schema_path("trace_pack.schema.json"), "Trace Pack"
    )
    repro_path = storage.write_json(
        f"logs/repro_packs/{repro['id']}.json", repro
    )
    trace_path = storage.write_json(
        f"logs/trace_packs/{trace['id']}.json", trace
    )
    return {"repro_pack": repro_path, "trace_pack": trace_path}
