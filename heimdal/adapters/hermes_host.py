"""Hermes host integration.

The runnable surface for Hermes. A Hermes agent drives Heimdal by importing
:func:`handle` and calling it with a Hermes skill-invocation payload, or via
``heimdal hermes run``. Heimdal appears to Hermes as a single skill/sub-agent.

The :class:`~heimdal.adapters.hermes_adapter.HermesAdapter` only translates
payloads; this module orchestrates translate -> Heimdal Runtime -> translate
back, and delivers the result to a file callback when one is requested.
"""

from __future__ import annotations

from heimdal.adapters.hermes_adapter import HermesAdapter
from heimdal.adapters.host_support import deliver_callback, read_answer
from heimdal.core.runtime import Runtime


def handle(payload: dict, runtime: Runtime | None = None) -> dict:
    """Run one Hermes task end to end; return a Hermes-style result.

    Hermes integrates by importing this function and calling it. Repro and
    Trace packs are written by the runtime. A ``callback.file`` entry, when
    present, receives the result under storage/workspace. Pass a reused
    ``runtime`` to avoid re-selecting the backend on every call.
    """
    adapter = HermesAdapter()
    envelope = adapter.to_host_task_envelope(payload)
    runtime = runtime or Runtime()
    result = runtime.run_envelope(envelope)
    hermes_result = adapter.from_heimdal_result(result)
    # The result envelope does not carry the Hermes session id; correlate it
    # back here from the original payload.
    hermes_result["hermes_session_id"] = payload.get("hermes_session_id", "")
    hermes_result["answer"] = read_answer(result)
    hermes_result["callback_delivered"] = deliver_callback(payload, hermes_result, runtime)
    return hermes_result
