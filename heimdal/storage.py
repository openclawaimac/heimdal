"""Storage layout management.

Creates and exposes the on-disk directory tree described in
docs/builder_pack/09_storage_context/STORAGE_LAYOUT.md.
"""

from __future__ import annotations

import json
import os

LAYOUT = [
    "truth",
    "working_state",
    "experience",
    "skills",
    "skills/general",
    "skills/research",
    "skills/coding",
    "skills/ops",
    "skills/business",
    "skills/writing",
    "skills/verifier",
    "skills/archived",
    "patches/stable",
    "patches/beta",
    "patches/experimental",
    "patches/rejected",
    "patches/archived",
    "patches/reviews",
    "patches/evals",
    "eval",
    "eval_runs",
    "artifacts",
    "logs/repro_packs",
    "logs/trace_packs",
    "logs/hardware_profiles",
    "workspace",
    "dream/runs",
    "dream/reports",
    "dream/proposals",
    "dream/synthetic_tasks",
    "dream/failure_mining",
    "dream/logs",
    "mirror/runs",
    "mirror/reports",
    "mirror/comparisons",
    "mirror/teacher_outputs",
    "mirror/redacted_inputs",
    "mirror/diffs",
    "mirror/proposals",
    "mirror/logs",
]


class Storage:
    """Resolved storage tree rooted at ``root``."""

    def __init__(self, root: str):
        self.root = root

    def ensure(self) -> "Storage":
        for sub in LAYOUT:
            os.makedirs(os.path.join(self.root, sub), exist_ok=True)
        return self

    def path(self, *parts: str) -> str:
        return os.path.join(self.root, *parts)

    def write_json(self, rel_path: str, obj) -> str:
        full = self.path(rel_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, sort_keys=True, default=str)
        return full

    @staticmethod
    def read_json(path: str):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def latest(self, rel_dir: str, suffix: str = ".json") -> str | None:
        directory = self.path(rel_dir)
        if not os.path.isdir(directory):
            return None
        candidates = [
            os.path.join(directory, name)
            for name in os.listdir(directory)
            if name.endswith(suffix)
        ]
        if not candidates:
            return None
        return max(candidates, key=os.path.getmtime)
