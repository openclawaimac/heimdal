"""A minimal JSON Schema validator.

Supports the subset of draft 2020-12 used by the Heimdal schemas: type,
required, enum, properties, items, minimum, maximum, additionalProperties.
Kept in-repo to avoid an external dependency.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

_TYPE_MAP = {
    "object": dict,
    "array": list,
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "null": type(None),
}


def _type_matches(value: Any, type_name: str) -> bool:
    expected = _TYPE_MAP.get(type_name)
    if expected is None:
        return True
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "boolean":
        return isinstance(value, bool)
    return isinstance(value, expected)


def _validate(value: Any, schema: dict, path: str, errors: list[str]) -> None:
    if not isinstance(schema, dict):
        return

    type_spec = schema.get("type")
    if type_spec is not None:
        candidates = type_spec if isinstance(type_spec, list) else [type_spec]
        if not any(_type_matches(value, t) for t in candidates):
            errors.append(f"{path or '<root>'}: expected type {type_spec}, got {type(value).__name__}")
            return

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path or '<root>'}: value {value!r} not in enum {schema['enum']}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path or '<root>'}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path or '<root>'}: {value} > maximum {schema['maximum']}")

    if isinstance(value, dict):
        for required in schema.get("required", []):
            if required not in value:
                errors.append(f"{path or '<root>'}: missing required property '{required}'")
        properties = schema.get("properties", {})
        for key, sub_value in value.items():
            child_path = f"{path}.{key}" if path else key
            if key in properties:
                _validate(sub_value, properties[key], child_path, errors)
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path or '<root>'}: additional property '{key}' is not allowed")

    if isinstance(value, list) and "items" in schema:
        item_schema = schema["items"]
        for index, item in enumerate(value):
            _validate(item, item_schema, f"{path}[{index}]", errors)


def validate(instance: Any, schema: dict) -> list[str]:
    """Return a list of human-readable errors; empty means valid."""
    errors: list[str] = []
    _validate(instance, schema, "", errors)
    return errors


def is_valid(instance: Any, schema: dict) -> bool:
    return not validate(instance, schema)


@lru_cache(maxsize=None)
def load_schema(path: str) -> dict:
    """Load a schema file. Cached: schema files are immutable at runtime and
    callers treat the result as read-only."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def validate_or_raise(instance: Any, schema_path: str, label: str, exc=ValueError) -> None:
    """Validate ``instance`` against the schema at ``schema_path`` or raise."""
    errors = validate(instance, load_schema(schema_path))
    if errors:
        raise exc(f"Invalid {label}: " + "; ".join(errors))
