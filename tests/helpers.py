"""Shared test helpers."""

from __future__ import annotations

import os

import yaml

from heimdal.config import load_config
from heimdal.ids import repo_root


def temp_config(storage_dir: str):
    """A Config that writes its storage tree into ``storage_dir``."""
    config = load_config()
    config.storage_root = storage_dir
    return config


def write_temp_manifest(directory: str, storage_dir: str) -> str:
    """Write a manifest whose storage_root points at a temp directory."""
    config = load_config()
    manifest = {k: v for k, v in config.manifest.items()}
    runtime = dict(manifest.get("runtime", {}))
    runtime["storage_root"] = storage_dir
    manifest["runtime"] = runtime
    path = os.path.join(directory, "manifest.yml")
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(manifest, fh)
    return path


def repo_path(*parts: str) -> str:
    return os.path.join(repo_root(), *parts)
