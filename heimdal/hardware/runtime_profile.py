"""Runtime Profiles.

A runtime profile is the small budget vocabulary Heimdal uses to adapt to
the machine it's installed on: max context window, default quality
level, repair budget, and Dream/Mirror defaults. Profiles are layered:

    BUILTIN_PROFILES   <-  hardcoded sane defaults shipped with Heimdal
    manifest.runtime_profiles  <-  per-installation overrides (optional)
    storage/runtime/runtime_profile.json  <-  the active profile + source

The Runtime reads the active profile at startup and threads its limits
into task contract construction, so a freshly-installed Heimdal on a
weak machine automatically downsizes its budgets without the operator
having to edit the manifest.
"""

from __future__ import annotations

import json
import os
from typing import Iterable

from heimdal.hardware.capability_matrix import recommend_profile
from heimdal.ids import now_iso

# Public profile names. v0.6.2 keeps the five short names from the spec;
# internal sub-classifications (single_gpu_low / single_gpu_high) are
# applied via builtin overrides without changing the public name.
PROFILES = ("cpu_only", "dev", "single_gpu", "pipeline", "factory")

# Conservative defaults. The cpu_only profile is intentionally tight so
# Heimdal stays usable on a laptop without a GPU.
BUILTIN_PROFILES: dict[str, dict] = {
    "cpu_only": {
        "max_context_tokens": 2048,
        "default_quality_level": "B1",
        "max_repair_iterations": 1,
        "dream_enabled_default": False,
        "mirror_enabled_default": False,
    },
    "dev": {
        "max_context_tokens": 4096,
        "default_quality_level": "B1",
        "max_repair_iterations": 2,
        "dream_enabled_default": False,
        "mirror_enabled_default": False,
    },
    "single_gpu": {
        "max_context_tokens": 8192,
        "default_quality_level": "B2",
        "max_repair_iterations": 2,
        "dream_enabled_default": "manual",
        "mirror_enabled_default": False,
    },
    "pipeline": {
        "max_context_tokens": 12000,
        "default_quality_level": "B2",
        "max_repair_iterations": 3,
        "dream_enabled_default": "idle",
        "mirror_enabled_default": False,
    },
    "factory": {
        "max_context_tokens": 16000,
        "default_quality_level": "B3",
        "max_repair_iterations": 3,
        "dream_enabled_default": "idle",
        "mirror_enabled_default": "budgeted",
    },
}

# Source tags for the active-profile artifact.
SOURCE_AUTO = "auto"
SOURCE_MANUAL = "manual"
SOURCE_MANIFEST = "manifest"


def limits_for(name: str, *, manifest_profiles: dict | None = None) -> dict:
    """Return the limits dict for ``name`` -- manifest overrides win.

    Falls back to ``dev`` for an unknown name so a typo in the manifest
    can never crash the runtime.
    """
    if name not in BUILTIN_PROFILES:
        name = "dev"
    merged = dict(BUILTIN_PROFILES[name])
    overrides = (manifest_profiles or {}).get(name) or {}
    if isinstance(overrides, dict):
        merged.update(overrides)
    return merged


def detect(config, matrix: dict | None = None) -> str:
    """Pick a profile name from the latest capability matrix (or rebuild)."""
    if matrix is None:
        from heimdal.hardware import capability_matrix as cm
        matrix = cm.build_matrix(config, run_capability_tests=False)
    return recommend_profile(matrix.get("hardware", {}))


def active(storage, config, *, manifest_profiles: dict | None = None,
           matrix: dict | None = None) -> dict:
    """Return the currently-active profile (name + source + limits).

    Reads ``storage/runtime/runtime_profile.json`` when it exists;
    otherwise auto-detects from the capability matrix. Source tagging
    lets trace/repro tell which way the choice was made.
    """
    path = storage.path("runtime", "runtime_profile.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                stored = json.load(fh)
            name = stored.get("name", "dev")
            source = stored.get("source", SOURCE_MANUAL)
            limits = limits_for(name, manifest_profiles=manifest_profiles)
            return {"name": name, "source": source, "limits": limits}
        except (OSError, ValueError):
            pass
    # No stored profile -> auto-detect.
    name = detect(config, matrix=matrix)
    return {
        "name": name,
        "source": SOURCE_AUTO,
        "limits": limits_for(name, manifest_profiles=manifest_profiles),
    }


def write(storage, name: str, *, source: str = SOURCE_MANUAL,
          manifest_profiles: dict | None = None) -> dict:
    """Persist the active profile to ``storage/runtime/runtime_profile.json``."""
    if name not in PROFILES:
        raise ValueError(
            f"Unknown profile name: {name!r}. Use one of {PROFILES}."
        )
    record = {
        "name": name,
        "source": source,
        "limits": limits_for(name, manifest_profiles=manifest_profiles),
        "written_at": now_iso(),
    }
    storage.write_json("runtime/runtime_profile.json", record)
    return record


def explain(name: str, *, manifest_profiles: dict | None = None) -> str:
    """Human-readable summary of what a profile changes."""
    limits = limits_for(name, manifest_profiles=manifest_profiles)
    lines = [f"profile: {name}"]
    for key in ("max_context_tokens", "default_quality_level",
                "max_repair_iterations", "dream_enabled_default",
                "mirror_enabled_default"):
        lines.append(f"  {key:<26}: {limits.get(key)}")
    return "\n".join(lines)


def task_contract_overrides(limits: dict) -> dict:
    """Slice of ``limits`` that build_contract should consult."""
    return {
        "default_quality_level": limits.get("default_quality_level"),
        "max_context_tokens": limits.get("max_context_tokens"),
        "max_repair_iterations": limits.get("max_repair_iterations"),
    }
