"""Capability Matrix builder.

Heimdal's adaptive runtime starts with a machine snapshot: what platform,
how much RAM/VRAM, which Ollama models are installed, and which of those
models can actually do the jobs we'd assign them to. The matrix is the
durable artifact every later subsystem (role assigner, runtime profile,
runtime decisions) reads from.

Capability tests stay cheap and ignore embedding models: a generation
test on ``nomic-embed-text`` would falsely fail. ``build_matrix`` always
succeeds; per-model test failures are recorded as ``"fail"`` rather than
raised so ``heimdal doctor`` keeps running.
"""

from __future__ import annotations

import json
import platform

from heimdal import __version__
from heimdal.hardware import profiler
from heimdal.ids import now_iso
from heimdal.models.base import is_embedding_model
from heimdal.models.ollama import OllamaBackend

# The runtime-profile names we recommend. Names are intentionally simple;
# v0.6.2 layers per-profile budgets on top of them.
PROFILES = ("cpu_only", "dev", "single_gpu", "pipeline", "factory")

# Canonical map from runtime-profile name to the legacy deployment-mode
# display label (profiler.deployment_mode). Keeping this here -- next to
# recommend_profile, the single source of truth for the hardware->tier
# decision -- is what stops the two classification systems from drifting.
DEPLOYMENT_LABELS = {
    "cpu_only": "Dev",
    "dev": "Dev",
    "single_gpu": "Single Device",
    "pipeline": "Pipeline",
    "factory": "Factory",
}


def recommend_profile(hardware: dict) -> str:
    """Pick a runtime profile from the hardware snapshot.

    Rules follow the v0.6.2 spec:
        - no GPU + WSL2 -> dev
        - no GPU + linux/macos -> cpu_only
        - 1 GPU or Apple Silicon -> single_gpu
        - 2-3 GPUs -> pipeline
        - 4+ GPUs -> factory
    """
    gpu = hardware.get("gpu", {}) or {}
    count = int(gpu.get("count", 0) or 0)
    metal = bool(gpu.get("metal"))
    flavour = (hardware.get("os") or {}).get("flavour", "")
    if count == 0 and not metal:
        return "dev" if flavour == "wsl2" else "cpu_only"
    if metal or count == 1:
        return "single_gpu"
    if 2 <= count <= 3:
        return "pipeline"
    return "factory"


def safe_context_tokens(vram_mb: int, has_metal: bool) -> int:
    """Conservative ctx budget per device class. Heuristic, not a benchmark."""
    if has_metal:
        return 8192
    if vram_mb >= 40_000:
        return 16384
    if vram_mb >= 20_000:
        return 12000
    if vram_mb >= 10_000:
        return 8192
    if vram_mb >= 6_000:
        return 4096
    return 2048


def _smoke_basic(backend: OllamaBackend, model: str) -> dict:
    try:
        gen = backend.generate(
            "Reply with the single word: OK", model=model, max_tokens=8
        )
        return {"status": "pass" if gen.text.strip() else "fail",
                "sample": gen.text.strip()[:80]}
    except (RuntimeError, OSError) as exc:
        return {"status": "fail", "error": str(exc)[:200]}


def _smoke_json(backend: OllamaBackend, model: str) -> dict:
    try:
        gen = backend.generate(
            'Return JSON: {"ok": true}', model=model, json_mode=True, max_tokens=32,
        )
        json.loads(gen.text)
        return {"status": "pass"}
    except (RuntimeError, OSError, ValueError) as exc:
        return {"status": "fail", "error": str(exc)[:200]}


def _smoke_semantic(backend: OllamaBackend, model: str) -> dict:
    """Mini semantic-verifier prompt -- enough to tell us the model can
    follow a strict-JSON verdict request."""
    prompt = (
        "TASK:\nExplain what a queue is.\n\n"
        "RESPONSE:\nA queue is a first-in, first-out data structure.\n"
    )
    system = (
        "You are a strict semantic verifier. Return ONLY JSON of the form "
        '{"status": "pass"|"fail", "score": 0.0-1.0, '
        '"rationale_short": "..."}.'
    )
    try:
        gen = backend.generate(
            prompt, model=model, system=system, json_mode=True, max_tokens=120,
            temperature=0.0,
        )
        raw = json.loads(gen.text)
        ok = isinstance(raw, dict) and raw.get("status") in ("pass", "fail")
        return {"status": "pass" if ok else "fail"}
    except (RuntimeError, OSError, ValueError):
        return {"status": "fail"}


def test_model(backend: OllamaBackend, model: str) -> dict:
    """Run the per-model capability suite. Embedding models are skipped."""
    if is_embedding_model(model):
        return {
            "skipped": True,
            "reason": "embedding model -- generation/json tests do not apply",
        }
    basic = _smoke_basic(backend, model)
    json_test = _smoke_json(backend, model)
    semantic = _smoke_semantic(backend, model)
    worker_candidate = (
        basic.get("status") == "pass" and json_test.get("status") == "pass"
    )
    verifier_candidate = (
        json_test.get("status") == "pass" and semantic.get("status") == "pass"
    )
    return {
        "basic_generation": basic.get("status", "fail"),
        "json_output": json_test.get("status", "fail"),
        "semantic_judgment": semantic.get("status", "fail"),
        "worker_candidate": worker_candidate,
        "semantic_verifier_candidate": verifier_candidate,
        "safe_context_tokens": None,
        "notes": [],
    }


def build_matrix(
    config,
    *,
    run_capability_tests: bool = True,
    target_model: str | None = None,
    all_models: bool = False,
) -> dict:
    """Compose the v0.6.0 capability matrix from a single inspection pass.

    ``target_model`` tests only one model; ``all_models`` tests every
    installed Ollama generative model. Default tests the first generative
    model (matches existing doctor behavior).
    """
    os_info = profiler.detect_os()
    gpus = profiler.detect_gpus()
    cpu = profiler.detect_cpu()
    ram_gb = profiler.detect_ram_gb()
    disk_class = profiler.detect_disk_class(config.storage_root)
    ollama = profiler.detect_ollama(config)
    storage_warnings = profiler.wsl2_storage_warnings(config.storage_root)
    warnings = list(storage_warnings)

    # Accelerator block: union of CUDA + Metal + ROCm devices observed.
    accelerators: list[dict] = []
    if gpus.get("cuda"):
        accelerators.append({
            "kind": "cuda",
            "count": gpus["count"],
            "devices": [d for d in gpus["devices"] if d.get("kind") == "cuda"],
            "total_vram_gb": profiler.total_vram_gb(gpus),
        })
    if gpus.get("metal"):
        accelerators.append({
            "kind": "metal",
            "count": 1,
            "unified_memory_gb": ram_gb,
        })
    if gpus.get("rocm"):
        accelerators.append({
            "kind": "rocm",
            "count": gpus["count"],
            "devices": [d for d in gpus["devices"] if d.get("kind") == "rocm"],
        })

    hardware = {
        "cpu": cpu,
        "ram": {"total_gb": ram_gb},
        "disk": {"class": disk_class, "root": config.storage_root},
        "gpu": gpus,
        "accelerators": accelerators,
    }

    model_capabilities: dict[str, dict] = {}
    if run_capability_tests and ollama["reachable"] and ollama["models"]:
        backend = OllamaBackend.from_config(config)
        targets: list[str]
        if target_model:
            targets = [target_model] if target_model in ollama["models"] else []
            if not targets:
                warnings.append(
                    f"requested --model {target_model!r} is not installed in Ollama."
                )
        elif all_models:
            targets = list(ollama["models"])
        else:
            from heimdal.models.base import select_generative_model
            chosen = select_generative_model(config, ollama["models"])
            targets = [chosen] if chosen else []
        for model in targets:
            try:
                result = test_model(backend, model)
            except Exception as exc:  # noqa: BLE001 -- doctor must keep going
                result = {"basic_generation": "fail", "error": str(exc)[:200]}
            # safe_context_tokens is computed from accelerator memory.
            total_vram_mb = sum(
                int(d.get("vram_mb", 0) or 0) for d in gpus.get("devices", [])
            )
            ctx = safe_context_tokens(total_vram_mb, has_metal=gpus.get("metal", False))
            if isinstance(result, dict) and "safe_context_tokens" in result:
                result["safe_context_tokens"] = ctx
            model_capabilities[model] = result

    if not ollama["reachable"]:
        warnings.append(
            "Ollama is not reachable; the matrix only records hardware. Heimdal "
            "will fall back to the offline backend until Ollama is started."
        )

    profile = recommend_profile(hardware)

    return {
        "created_at": now_iso(),
        "heimdal_version": __version__,
        "platform": {
            "system": os_info["system"],
            "flavour": os_info["flavour"],
            "machine": os_info["machine"],
            "release": os_info["release"],
            "python": platform.python_version(),
        },
        "hardware": hardware,
        "ollama": ollama,
        "model_capabilities": model_capabilities,
        "recommended_runtime_profile": profile,
        # Role assignments are produced separately by `heimdal models assign`
        # (v0.6.1) and stored at storage/runtime/role_assignments.json. The
        # matrix records the empty stub for backward compatibility with the
        # v0.6.0 schema; the canonical source for "who is the worker model?"
        # is the role_assignments artifact, not this field.
        "recommended_role_assignments": {},
        "warnings": warnings,
    }


def write_matrix(storage, matrix: dict) -> str:
    """Persist a capability matrix under logs/capability_matrix/ AND
    storage/runtime/capability_matrix.json (the canonical latest copy)."""
    from heimdal.ids import now_compact
    storage.write_json(f"logs/capability_matrix/{now_compact()}.json", matrix)
    return storage.write_json("runtime/capability_matrix.json", matrix)
