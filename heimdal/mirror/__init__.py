"""Mirror Mode -- optional cloud-teacher comparison and frontier diff.

Mirror Mode is OFF by default. When the operator enables it (manifest +
explicit CLI flags), Heimdal compares recent local outputs against a
teacher provider and emits structured improvement proposals. It never
mutates stable state -- proposals flow through the existing patch
lifecycle.
"""

from heimdal.mirror.provider import (
    TeacherInput,
    TeacherProvider,
    TeacherResult,
)
from heimdal.mirror.runner import (
    SOURCES,
    list_mirror_runs,
    load_mirror_report,
    run_mirror,
)

__all__ = [
    "SOURCES",
    "TeacherInput",
    "TeacherProvider",
    "TeacherResult",
    "list_mirror_runs",
    "load_mirror_report",
    "run_mirror",
]
