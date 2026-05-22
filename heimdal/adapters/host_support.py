"""Shared helpers for runnable host integrations (OpenClaw, Hermes).

Host modules orchestrate translate -> Runtime -> translate back; these helpers
cover the parts common to every host: reading the answer artifact and
delivering a result to a sandboxed file callback.
"""

from __future__ import annotations

import json
import os


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


def deliver_callback(payload: dict, host_result: dict, runtime) -> str | None:
    """Write a host result to a file callback under storage/workspace.

    Directory components in the requested name are stripped, so an external
    payload cannot write outside the sandboxed workspace.
    """
    requested = (payload.get("callback") or {}).get("file")
    if not requested:
        return None
    safe_name = os.path.basename(str(requested)) or "result.json"
    path = runtime.storage.path("workspace", safe_name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(host_result, fh, indent=2, sort_keys=True, default=str)
    return path
