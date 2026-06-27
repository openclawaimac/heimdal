"""Hardware Profiler.

Backs ``heimdal doctor``: detects OS, native Linux vs WSL2, CPU, RAM, disk
class, Ollama reachability and installed models, GPUs/VRAM, CUDA, and Apple
Silicon (docs/builder_pack/06_model_hardware/HARDWARE_PROFILER_REQUIREMENTS.md).
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess

from heimdal.ids import now_iso
from heimdal.models.base import select_generative_model
from heimdal.models.ollama import OllamaBackend


def is_wsl2() -> bool:
    for path in ("/proc/version", "/proc/sys/kernel/osrelease"):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                content = fh.read().lower()
        except OSError:
            continue
        if "microsoft" in content or "wsl" in content:
            return True
    return False


def detect_os() -> dict:
    system = platform.system()
    flavour = system
    if system == "Linux":
        flavour = "wsl2" if is_wsl2() else "linux_native"
    elif system == "Darwin":
        flavour = "macos"
    return {
        "system": system,
        "flavour": flavour,
        "release": platform.release(),
        "machine": platform.machine(),
    }


def detect_cpu() -> dict:
    info = {"logical_cores": os.cpu_count() or 0, "model": platform.processor() or "unknown"}
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    info["model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return info


def detect_ram_gb() -> float:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    return round(kb / (1024 * 1024), 1)
    except OSError:
        pass
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return round(total / (1024 ** 3), 1)
    except (ValueError, OSError, AttributeError):
        return 0.0


def detect_disk_class(path: str) -> str:
    """Best-effort SSD/HDD detection via the rotational flag on Linux."""
    sys_block = "/sys/block"
    if not os.path.isdir(sys_block):
        return "unknown"
    rotational = False
    found = False
    for device in os.listdir(sys_block):
        flag = os.path.join(sys_block, device, "queue", "rotational")
        try:
            with open(flag, "r", encoding="utf-8") as fh:
                found = True
                if fh.read().strip() == "1":
                    rotational = True
        except OSError:
            continue
    if not found:
        return "unknown"
    return "hdd" if rotational else "ssd_or_nvme"


def detect_gpus() -> dict:
    result = {"count": 0, "cuda": False, "metal": False, "rocm": False, "devices": []}
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        result["metal"] = True
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run(
                [smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if out.returncode == 0:
                for line in out.stdout.strip().splitlines():
                    name, _, mem = line.partition(",")
                    result["devices"].append(
                        {"kind": "cuda", "name": name.strip(), "vram_mb": _safe_int(mem)}
                    )
                result["count"] = len(result["devices"])
                result["cuda"] = result["count"] > 0
        except (subprocess.SubprocessError, OSError):
            pass
    # Best-effort ROCm detection: rocm-smi shipped with ROCm prints GPU info.
    if not result["devices"] and shutil.which("rocm-smi"):
        try:
            out = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True, text=True, timeout=10,
            )
            if out.returncode == 0 and "GPU" in out.stdout:
                result["rocm"] = True
                # Without a parse-friendly format we just count "GPU[" markers.
                count = out.stdout.count("GPU[")
                result["count"] = count or 1
                result["devices"].append({"kind": "rocm", "name": "rocm-gpu",
                                          "vram_mb": 0})
        except (subprocess.SubprocessError, OSError):
            pass
    return result


def total_vram_gb(gpus: dict) -> float:
    total_mb = sum(int(d.get("vram_mb", 0) or 0) for d in gpus.get("devices", []))
    return round(total_mb / 1024.0, 1)


def wsl2_storage_warnings(storage_root: str) -> list[str]:
    """Flag a storage_root pinned to a Windows drive under WSL2 -- slow I/O."""
    warnings: list[str] = []
    if is_wsl2():
        normalized = os.path.normpath(os.path.abspath(storage_root))
        for drive in ("/mnt/c", "/mnt/d", "/mnt/e", "/mnt/f"):
            if normalized.startswith(drive + os.sep) or normalized == drive:
                warnings.append(
                    f"storage_root is under {drive} on WSL2; move it under the "
                    "Linux filesystem (~/) for usable disk performance."
                )
    return warnings


def _safe_int(text: str) -> int:
    try:
        return int(float(text.strip()))
    except (ValueError, AttributeError):
        return 0


def deployment_mode(gpu_count: int) -> str:
    """Legacy display label for a GPU count (Dev / Single Device / ...).

    Delegates to capability_matrix.recommend_profile so the count thresholds
    live in exactly one place and the two classification systems can't drift.
    Imported lazily because capability_matrix imports this module at load.
    """
    from heimdal.hardware.capability_matrix import (
        DEPLOYMENT_LABELS,
        recommend_profile,
    )

    profile = recommend_profile(
        {"gpu": {"count": gpu_count, "metal": False},
         "os": {"flavour": "linux_native"}}
    )
    return DEPLOYMENT_LABELS[profile]


def detect_ollama(config) -> dict:
    backend = OllamaBackend.from_config(config)
    reachable = backend.is_available()
    return {
        "base_url": backend.base_url,
        "reachable": reachable,
        "models": backend.list_models() if reachable else [],
    }


def quick_profile(config) -> dict:
    """Lightweight profile embedded in Repro Packs."""
    gpus = detect_gpus()
    os_info = detect_os()
    return {
        "os": os_info["flavour"],
        "cpu_cores": detect_cpu()["logical_cores"],
        "ram_gb": detect_ram_gb(),
        "gpu_count": gpus["count"],
        "deployment_mode": deployment_mode(gpus["count"]),
    }


def capability_tests(config, ollama: dict, model_override: str | None = None) -> list[dict]:
    """Light model capability smoke tests; skipped when Ollama is unavailable.

    Tests a *generative* model: a manifest worker/verifier candidate when one
    is installed, otherwise the first installed non-embedding model. Embedding
    models (e.g. nomic-embed-text) would falsely fail a generation test.
    """
    tests: list[dict] = []
    if not ollama.get("reachable") or not ollama.get("models"):
        return tests

    model = model_override or select_generative_model(config, ollama["models"])
    if model is None:
        return [
            {
                "name": "model_selection",
                "model": None,
                "passed": False,
                "error": "No text-generation model installed; only embedding models found.",
            }
        ]

    backend = OllamaBackend.from_config(config)
    try:
        gen = backend.generate("Reply with the single word: OK", model=model, max_tokens=8)
        tests.append(
            {"name": "basic_generation", "model": model, "passed": bool(gen.text.strip())}
        )
    except (RuntimeError, OSError) as exc:
        tests.append(
            {"name": "basic_generation", "model": model, "passed": False, "error": str(exc)}
        )
    try:
        gen = backend.generate(
            'Return JSON: {"ok": true}', model=model, json_mode=True, max_tokens=32
        )
        json.loads(gen.text)
        tests.append({"name": "json_output", "model": model, "passed": True})
    except (RuntimeError, OSError, ValueError) as exc:
        tests.append(
            {"name": "json_output", "model": model, "passed": False, "error": str(exc)}
        )
    return tests


def full_profile(
    config, run_capability_tests: bool = True, capability_model: str | None = None
) -> dict:
    """Full hardware + model profile written by ``heimdal doctor``."""
    os_info = detect_os()
    gpus = detect_gpus()
    ollama = detect_ollama(config)
    profile = {
        "timestamp": now_iso(),
        "os": os_info,
        "cpu": detect_cpu(),
        "ram_gb": detect_ram_gb(),
        "disk_class": detect_disk_class(config.storage_root),
        "gpu": gpus,
        "ollama": ollama,
        "deployment_mode": deployment_mode(gpus["count"]),
        "python": platform.python_version(),
        "warnings": [],
        "capability_tests": [],
    }
    if not ollama["reachable"]:
        profile["warnings"].append(
            "Ollama is not reachable; Heimdal will use the offline backend."
        )
    elif not ollama["models"]:
        profile["warnings"].append(
            "Ollama is reachable but has no installed models."
        )
    if gpus["count"] == 0:
        profile["warnings"].append("No GPU detected; running in CPU/Dev mode.")
    if run_capability_tests:
        profile["capability_tests"] = capability_tests(config, ollama, capability_model)
    return profile
