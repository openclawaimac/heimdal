"""Intake: validate an incoming Host Task Envelope.

The envelope is the single standard request shape every host adapter must
produce (docs/builder_pack/02_contracts/HOST_ADAPTER_CONTRACT.md).
"""

from __future__ import annotations

from heimdal import jsonschema_min


class IntakeError(ValueError):
    """Raised when a Host Task Envelope is malformed."""


def validate_envelope(envelope: dict, config) -> list[str]:
    """Return schema + semantic errors for a Host Task Envelope."""
    schema = jsonschema_min.load_schema(config.schema_path("host_task_envelope.schema.json"))
    errors = jsonschema_min.validate(envelope, schema)

    task_request = envelope.get("task_request", {})
    if isinstance(task_request, dict):
        if not task_request.get("task_id"):
            errors.append("task_request.task_id: required and must be non-empty")
        if not task_request.get("instruction"):
            errors.append("task_request.instruction: required and must be non-empty")
    return errors


def intake(envelope: dict, config) -> dict:
    """Validate the envelope and return it, or raise :class:`IntakeError`."""
    errors = validate_envelope(envelope, config)
    if errors:
        raise IntakeError("Invalid Host Task Envelope: " + "; ".join(errors))
    return envelope
