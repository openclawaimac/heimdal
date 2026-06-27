"""Configuration loading: the Heimdal manifest and the sandbox policy.

Supports ``${VAR:-default}`` environment substitution inside the manifest so
the Ollama base URL can be overridden via OLLAMA_HOST.
"""

from __future__ import annotations

import os
import re
from typing import Any

import yaml

from heimdal.ids import repo_root

_ENV_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")

DEFAULT_MANIFEST = "config/heimdal.manifest.yml"


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def repl(match: re.Match) -> str:
            name, default = match.group(1), match.group(2) or ""
            return os.environ.get(name, default)

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _abspath(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(repo_root(), path))


class Config:
    """Resolved Heimdal configuration."""

    def __init__(self, manifest: dict, manifest_path: str):
        self.manifest = manifest
        self.manifest_path = manifest_path
        self.storage_root = _abspath(
            manifest.get("runtime", {}).get("storage_root", "./storage")
        )
        self._sandbox: dict | None = None

    # -- accessors ---------------------------------------------------------
    @property
    def runtime(self) -> dict:
        return self.manifest.get("runtime", {})

    @property
    def ollama(self) -> dict:
        return self.manifest.get("ollama", {})

    @property
    def model_profiles(self) -> dict:
        return self.manifest.get("model_profiles", {})

    @property
    def model_roles(self) -> dict:
        return self.manifest.get("model_roles", {})

    @property
    def runtime_profiles(self) -> dict:
        return self.manifest.get("runtime_profiles", {})

    @property
    def scheduler(self) -> dict:
        return self.manifest.get("scheduler", {})

    @property
    def budgets(self) -> dict:
        return self.manifest.get("budgets", {})

    @property
    def verifier(self) -> dict:
        return self.manifest.get("verifier", {})

    @property
    def retrieval(self) -> dict:
        return self.manifest.get("retrieval", {})

    @property
    def mirror(self) -> dict:
        return self.manifest.get("mirror", {})

    @property
    def privacy_mode(self) -> str:
        return self.runtime.get("privacy_mode", "local_only")

    def schema_path(self, name: str) -> str:
        return _abspath(os.path.join("schemas", name))

    def sandbox_policy(self) -> dict:
        if self._sandbox is None:
            policy_file = self.manifest.get("sandbox", {}).get(
                "policy_file", "config/sandbox_policy.yml"
            )
            path = _abspath(policy_file)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as fh:
                    self._sandbox = _expand_env(yaml.safe_load(fh) or {})
            else:
                self._sandbox = {}
        return self._sandbox


class ConfigError(ValueError):
    """Raised when the Heimdal manifest is missing, unreadable, or malformed."""


def load_config(manifest_path: str | None = None) -> Config:
    """Load and resolve the Heimdal manifest.

    Surfaces a clear, actionable ConfigError instead of a raw traceback when
    the manifest is missing, unparseable YAML, or not a mapping.
    """
    path = _abspath(manifest_path or DEFAULT_MANIFEST)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(
            f"Heimdal manifest not found: {path}. Pass --manifest <file> or "
            f"create config/heimdal.manifest.yml."
        ) from exc
    except yaml.YAMLError as exc:
        raise ConfigError(
            f"Heimdal manifest at {path} is not valid YAML: {exc}"
        ) from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(
            f"Heimdal manifest at {path} must be a mapping at the top level, "
            f"got {type(raw).__name__}."
        )
    return Config(_expand_env(raw), path)
