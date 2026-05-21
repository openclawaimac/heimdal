"""Identifier, hashing, and timestamp helpers used across the runtime."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone


def now_iso() -> str:
    """UTC timestamp in ISO-8601 with a trailing Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def now_compact() -> str:
    """Filesystem-safe UTC timestamp, e.g. 20260521T131400Z."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def new_id(prefix: str) -> str:
    """Time-ordered, collision-resistant id with a human-readable prefix."""
    stamp = format(int(time.time() * 1000), "x")
    rand = uuid.uuid4().hex[:8]
    return f"{prefix}_{stamp}_{rand}"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_obj(obj) -> str:
    """Stable hash of any JSON-serialisable object."""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_text(payload)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) used for budget enforcement."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def repo_root() -> str:
    """Repository root, resolved relative to this file."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
