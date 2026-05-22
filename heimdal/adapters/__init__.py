"""Host adapters.

Adapters are translators between a host framework and the standard Heimdal Host
Task Envelope / Result Envelope. They must not own orchestration logic
(docs/builder_pack/01_architecture/HOST_AGNOSTIC_DESIGN.md).
"""

from heimdal.adapters.base import HostAdapter
from heimdal.adapters.cli_adapter import CLIAdapter
from heimdal.adapters.hermes_adapter import HermesAdapter
from heimdal.adapters.openclaw_adapter import OpenClawAdapter

__all__ = ["HostAdapter", "CLIAdapter", "HermesAdapter", "OpenClawAdapter"]
