"""OpenClaw host integration.

The runnable surface for OpenClaw. An OpenClaw process drives Heimdal by
importing :func:`handle` and calling it with an OpenClaw-style payload, or via
``heimdal openclaw run``. Heimdal appears to OpenClaw as a single agent.

The :class:`~heimdal.adapters.openclaw_adapter.OpenClawAdapter` only translates
payloads; this module orchestrates: translate -> Heimdal Runtime -> translate
back, and delivers the result to a file callback when one is requested
(docs/builder_pack/02_contracts/OPENCLAW_ADAPTER_SPEC.md).
"""

from __future__ import annotations

import json
import os

from heimdal.adapters.openclaw_adapter import OpenClawAdapter
from heimdal.core.runtime import Runtime


def _read_answer(result: dict) -> str:
    """Return the response artifact's text, if the run produced one."""
    for artifact in result.get("artifacts", []):
        if artifact.get("type") == "response":
            try:
                with open(artifact["path"], "r", encoding="utf-8") as fh:
                    return fh.read()
            except OSError:
                return ""
    return ""


def _deliver_callback(payload: dict, oc_result: dict, runtime: Runtime) -> str | None:
    """Write the OpenClaw result to a file callback, if one is requested.

    Callback files are always delivered under ``storage/workspace``; directory
    components in the requested name are stripped, so an external payload
    cannot write outside the sandboxed workspace.
    """
    requested = (payload.get("callback") or {}).get("file")
    if not requested:
        return None
    safe_name = os.path.basename(str(requested)) or "openclaw_result.json"
    path = runtime.storage.path("workspace", safe_name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(oc_result, fh, indent=2, sort_keys=True, default=str)
    return path


def handle(payload: dict, runtime: Runtime | None = None) -> dict:
    """Run one OpenClaw task end to end; return an OpenClaw-style result.

    OpenClaw integrates by importing this function and calling it. Repro and
    Trace packs are written by the runtime. A ``callback.file`` entry, when
    present, receives the result under storage/workspace. Pass a reused
    ``runtime`` to avoid re-selecting the backend on every call.
    """
    adapter = OpenClawAdapter()
    envelope = adapter.to_host_task_envelope(payload)
    runtime = runtime or Runtime()
    result = runtime.run_envelope(envelope)
    oc_result = adapter.from_heimdal_result(result)
    oc_result["answer"] = _read_answer(result)
    oc_result["callback_delivered"] = _deliver_callback(payload, oc_result, runtime)
    return oc_result
