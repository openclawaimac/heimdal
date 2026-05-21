"""Capability Sandbox Policy.

Default beta policy is read-only: no shell, no network, blocked secret/system
paths (docs/builder_pack/08_security_sandbox/SANDBOX_POLICY.md). This module
loads the policy and provides minimal enforcement checks.
"""

from __future__ import annotations

import os


def _expand(path: str) -> str:
    return os.path.realpath(os.path.expanduser(path))


class SandboxPolicy:
    def __init__(self, policy: dict):
        sandbox = policy.get("tool_sandbox", {}) if policy else {}
        self.default_mode = sandbox.get("default_mode", "read_only")
        self.allow_shell = bool(sandbox.get("allow_shell", False))
        self.allowed_paths = [_expand(p) for p in sandbox.get("allowed_paths", [])]
        self.blocked_paths = [_expand(p) for p in sandbox.get("blocked_paths", [])]
        self.max_runtime_seconds = sandbox.get("max_runtime_seconds", 60)
        self.max_output_bytes = sandbox.get("max_output_bytes", 200000)
        network = sandbox.get("network", {}) or {}
        self.network_default = str(network.get("default", "off")).lower()
        self.network_allowlist = list(network.get("allowlist", []) or [])

    @classmethod
    def from_config(cls, config) -> "SandboxPolicy":
        return cls(config.sandbox_policy())

    # -- checks ------------------------------------------------------------
    def is_path_allowed(self, path: str) -> bool:
        target = _expand(path)
        for blocked in self.blocked_paths:
            if target == blocked or target.startswith(blocked + os.sep):
                return False
        if not self.allowed_paths:
            return True
        for allowed in self.allowed_paths:
            if target == allowed or target.startswith(allowed + os.sep):
                return True
        return False

    def check_path(self, path: str) -> None:
        if not self.is_path_allowed(path):
            raise PermissionError(f"Sandbox policy denies access to: {path}")

    def shell_allowed(self) -> bool:
        return self.allow_shell

    def network_allowed(self, host: str | None = None) -> bool:
        if self.network_default in ("on", "true"):
            return True
        if host is None:
            return False
        return host in self.network_allowlist

    def summary(self) -> dict:
        return {
            "default_mode": self.default_mode,
            "allow_shell": self.allow_shell,
            "network_default": self.network_default,
            "allowed_paths": self.allowed_paths,
            "blocked_paths": self.blocked_paths,
        }
