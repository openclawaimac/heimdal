"""Dream Mode -- Heimdal's local, eval-gated self-improvement loop.

Dream Mode is *invocation-driven* (``heimdal dream run``): it scans past Trace
Packs, Repro Packs, eval summaries, and bridge failures for recurring failure
patterns, then emits structured *proposals* (patches, skills, eval cases) into
``storage/dream/``. Dream Mode never mutates stable behavior; only the patch
promotion pipeline does that, and only after explicit human approval.
"""

from heimdal.dream.runner import (
    SOURCES,
    list_dream_runs,
    load_dream_report,
    run_dream,
)

__all__ = ["SOURCES", "run_dream", "list_dream_runs", "load_dream_report"]
