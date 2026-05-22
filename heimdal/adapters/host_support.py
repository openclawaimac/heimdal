"""Shared helpers for runnable host integrations (OpenClaw, Hermes).

Host modules orchestrate translate -> Runtime -> translate back; these helpers
cover the parts common to every host: reading the answer artifact, reducing
internal paths to host-safe references, and delivering a result to a sandboxed
file callback.
"""

from __future__ import annotations

import json
import os

from heimdal.core import repro_trace, status_codes


def read_answer(result: dict) -> str:
    """Return the response artifact's text, if the run produced one."""
    for artifact in result.get("artifacts", []):
        if artifact.get("type") == "response":
            try:
                with open(artifact["path"], "r", encoding="utf-8") as fh:
                    return fh.read()
            except OSError:
                return ""
    return ""


def host_safe_ref(path) -> str:
    """Reduce an internal filesystem path to a host-safe relative reference.

    External hosts must never receive absolute paths: they leak the local
    filesystem layout and user names. Only the trailing components that
    identify the artifact within storage/ are kept.
    """
    if not path:
        return ""
    parts = [p for p in str(path).replace("\\", "/").split("/") if p]
    return "/".join(parts[-3:])


def host_safe_artifacts(
    artifacts, omit=("context_packet", "task_contract")
) -> list[dict]:
    """Translate runtime artifacts into host-safe ``{type, ref}`` entries.

    Absolute paths are reduced to relative refs; internal-only artifacts (the
    Context Packet and Task Contract by default) are dropped from the external
    result so the host sees only the response and the verification summary.
    """
    safe: list[dict] = []
    for artifact in artifacts or []:
        if artifact.get("type") in omit:
            continue
        safe.append(
            {"type": artifact.get("type"), "ref": host_safe_ref(artifact.get("path"))}
        )
    return safe


def deliver_callback(payload: dict, host_result: dict, runtime):
    """Write a host result to a file callback under storage/workspace.

    Returns ``(callback_delivery, events)``: ``callback_delivery`` is a
    host-safe dict -- ``{status, target_ref}`` (plus ``code`` on failure) -- or
    ``None`` when no callback was requested. ``events`` are trace events
    recording the delivery for folding back into the run's Trace Pack. Only the
    sanitized relative target ref is exposed -- never an absolute path.

    Directory components in the requested name are stripped, so an external
    payload cannot write outside the sandboxed workspace.
    """
    requested = (payload.get("callback") or {}).get("file")
    if not requested:
        return None, []
    safe_name = os.path.basename(str(requested)) or "result.json"
    target_ref = f"workspace/{safe_name}"
    events = [repro_trace.trace_event("callback_delivery_start", target=target_ref)]
    path = runtime.storage.path("workspace", safe_name)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(host_result, fh, indent=2, sort_keys=True, default=str)
    except OSError as exc:
        events.append(
            repro_trace.trace_event(
                "callback_delivery_error", target=target_ref, error=str(exc)
            )
        )
        return (
            {
                "status": "failed",
                "target_ref": target_ref,
                "code": status_codes.CALLBACK_DELIVERY_FAILED,
            },
            events,
        )
    events.append(
        repro_trace.trace_event("callback_delivery_success", target=target_ref)
    )
    return {"status": "success", "target_ref": target_ref}, events
