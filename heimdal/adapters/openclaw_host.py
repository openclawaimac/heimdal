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

from heimdal.adapters.host_support import deliver_callback, read_answer
from heimdal.adapters.openclaw_adapter import OpenClawAdapter
from heimdal.core.runtime import Runtime


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
    oc_result["answer"] = read_answer(result)
    oc_result["callback_delivered"] = deliver_callback(payload, oc_result, runtime)
    return oc_result
