"""Host adapter interface."""

from __future__ import annotations


class HostAdapter:
    """A host adapter translates between a host and Heimdal envelopes.

    Adapters translate only: they do not build Context Packets, modify
    patch/eval policy, or bypass the sandbox.
    """

    host_type = "base"

    def to_host_task_envelope(self, raw_input) -> dict:
        """Map host-native input into a Heimdal Host Task Envelope."""
        raise NotImplementedError

    def from_heimdal_result(self, result: dict):
        """Map a Heimdal Result Envelope back into host-native output."""
        raise NotImplementedError
