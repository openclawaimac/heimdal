"""Local file bridge.

Allows external local agents to invoke Heimdal by writing JSON job files into
an inbox directory; Heimdal picks each up, dispatches to the requested
adapter, and writes a result JSON file to an outbox.

The bridge is a *transport layer only* -- it never re-implements Quality
Factory logic. It marshals files to/from the existing Hermes / OpenClaw /
generic handlers and emits machine-readable bridge-level failure codes when
the transport itself can't deliver a job (bad JSON, unknown adapter, Ollama
unreachable, ...).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import time

from heimdal import jsonschema_min
from heimdal.adapters.hermes_host import handle as run_hermes
from heimdal.adapters.openclaw_host import handle as run_openclaw
from heimdal.config import Config
from heimdal.core import status_codes
from heimdal.core.runtime import Runtime
from heimdal.hardware.profiler import detect_ollama
from heimdal.ids import now_iso
from heimdal.storage import Storage

DIRS = ("inbox", "processing", "outbox", "failed", "archive")
DEFAULT_POLL_INTERVAL = 2.0
DEFAULT_MAX_JOBS_PER_CYCLE = 16
MIN_FILE_AGE_SECONDS = 1.0
READY_SUFFIX = ".ready.json"
SUPPORTED_ADAPTERS = ("hermes", "openclaw", "generic")


class BridgeError(Exception):
    """A transport-layer failure with a machine-readable code."""

    def __init__(self, code: str, message: str, suggested_fix: str = ""):
        super().__init__(message)
        self.code = code
        self.message = message
        self.suggested_fix = suggested_fix


# -- paths -----------------------------------------------------------------
def _bridge_root(config: Config) -> str:
    return os.path.join(config.storage_root, "bridge")


def ensure_dirs(config: Config) -> dict:
    """Create storage/bridge/{inbox,...} and return paths by name."""
    root = _bridge_root(config)
    paths: dict[str, str] = {"root": root}
    for sub in DIRS:
        path = os.path.join(root, sub)
        os.makedirs(path, exist_ok=True)
        paths[sub] = path
    return paths


def resolve_paths(config: Config, *, inbox: str | None = None,
                  outbox: str | None = None) -> dict:
    """Resolve the 5 bridge directories, honouring optional --inbox/--outbox.

    The processing/failed/archive directories always sit alongside inbox in
    the same parent (so an explicit ``--inbox /tmp/foo/inbox`` still routes
    failures into ``/tmp/foo/failed`` rather than a stale default).
    """
    if not inbox and not outbox:
        return ensure_dirs(config)
    inbox = inbox or os.path.join(_bridge_root(config), "inbox")
    outbox = outbox or os.path.join(_bridge_root(config), "outbox")
    root = os.path.dirname(os.path.abspath(inbox))
    paths = {
        "root": root,
        "inbox": inbox,
        "outbox": outbox,
        "processing": os.path.join(root, "processing"),
        "failed": os.path.join(root, "failed"),
        "archive": os.path.join(root, "archive"),
    }
    for key in ("inbox", "outbox", "processing", "failed", "archive"):
        os.makedirs(paths[key], exist_ok=True)
    return paths


def status_counts(config: Config) -> dict:
    paths = ensure_dirs(config)
    counts = {}
    for sub in DIRS:
        path = paths[sub]
        counts[sub] = sum(
            1 for name in os.listdir(path)
            if os.path.isfile(os.path.join(path, name))
        )
    return counts


# -- inbox readiness -------------------------------------------------------
def _is_ready(path: str, now: float) -> bool:
    """Avoid reading partially-written files.

    A file is considered ready when its name ends in ``.ready.json`` (the
    explicit external signal) or when it is a plain ``.json`` that has been
    untouched for at least MIN_FILE_AGE_SECONDS.
    """
    name = os.path.basename(path)
    if name.endswith(READY_SUFFIX):
        return True
    if not name.endswith(".json"):
        return False
    try:
        return (now - os.path.getmtime(path)) >= MIN_FILE_AGE_SECONDS
    except OSError:
        return False


def _list_ready_jobs(inbox: str, max_jobs: int) -> list[str]:
    if not os.path.isdir(inbox):
        return []
    now = time.time()
    out: list[str] = []
    for name in sorted(os.listdir(inbox)):
        path = os.path.join(inbox, name)
        if not os.path.isfile(path):
            continue
        if _is_ready(path, now):
            out.append(path)
        if len(out) >= max_jobs:
            break
    return out


# -- filename safety -------------------------------------------------------
def _safe_basename(job_id: str, fallback: str) -> str:
    """A filesystem-safe leaf name derived from ``job_id``; no path traversal.

    ``os.path.basename`` first strips any directory components an attacker
    might smuggle in, then a character whitelist removes everything that
    isn't alphanumeric, ``-``, or ``_``.
    """
    raw = os.path.basename(str(job_id or ""))
    safe = "".join(c for c in raw if c.isalnum() or c in ("-", "_"))
    return safe or fallback


def _unique_path(directory: str, name: str) -> str:
    """Return ``<directory>/<name>``; if it exists, append a UTC timestamp."""
    path = os.path.join(directory, name)
    if not os.path.exists(path):
        return path
    stem, _, ext = name.rpartition(".")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return os.path.join(directory, f"{stem}-{stamp}.{ext}" if stem else f"{name}.{stamp}")


# -- dispatch --------------------------------------------------------------
def _resolved_runtime_args(job: dict, defaults: dict) -> dict:
    """Per-job ``runtime`` block overrides CLI defaults."""
    block = (job.get("runtime") or {}) if isinstance(job, dict) else {}
    return {
        "backend": block.get("backend") or defaults.get("backend"),
        "model": block.get("model") or defaults.get("model"),
        "verifier": block.get("verifier") or defaults.get("verifier"),
    }


def _check_backend(config: Config, args: dict) -> None:
    """Pre-flight backend reachability + model availability."""
    if args.get("backend") != "ollama":
        return
    ollama = detect_ollama(config)
    if not ollama["reachable"]:
        raise BridgeError(
            status_codes.OLLAMA_UNREACHABLE,
            f"Ollama is not reachable at {ollama['base_url']}.",
            "Start Ollama or rerun with backend=offline.",
        )
    model = args.get("model")
    if model and model not in ollama["models"]:
        raise BridgeError(
            status_codes.OLLAMA_MODEL_MISSING,
            f"Model {model!r} is not installed in Ollama.",
            f"Run: ollama pull {model}",
        )


def _run_adapter(job: dict, runtime: Runtime) -> tuple[str, dict, str, str]:
    """Dispatch payload to the named adapter; return (status, result, repro, trace)."""
    adapter = job.get("adapter")
    payload = job.get("payload") or {}
    if adapter == "hermes":
        result = run_hermes(payload, runtime)
        return (
            result["status"],
            result,
            result.get("repro_pack_ref", ""),
            result.get("trace_pack_ref", ""),
        )
    if adapter == "openclaw":
        result = run_openclaw(payload, runtime)
        return (
            result["outcome"],
            result,
            result.get("repro_pack_ref", ""),
            result.get("trace_pack_ref", ""),
        )
    if adapter == "generic":
        result = runtime.run_envelope(payload)
        root = runtime.storage.root
        repro_path = (result.get("repro_pack") or {}).get("path", "")
        trace_path = (result.get("trace_pack") or {}).get("path", "")
        repro_ref = os.path.relpath(repro_path, root) if repro_path else ""
        trace_ref = os.path.relpath(trace_path, root) if trace_path else ""
        return result["status"], result, repro_ref, trace_ref
    raise BridgeError(
        status_codes.ADAPTER_UNSUPPORTED,
        f"Unsupported adapter: {adapter!r}",
        f"Use one of: {', '.join(SUPPORTED_ADAPTERS)}.",
    )


# -- processing one job ----------------------------------------------------
def process_job(job_path: str, config: Config, paths: dict, defaults: dict) -> dict:
    """Process a single inbox file; always return a structured report.

    On success the job is moved to archive/ and a ``<safe_id>.result.json``
    appears in outbox/. On failure the job is moved to failed/ along with a
    machine-readable ``<safe_id>.error.json``. The bridge loop logs the
    returned report; the on-disk files are the durable record.
    """
    started = time.time()
    base = os.path.basename(job_path)
    processing_path = os.path.join(paths["processing"], base)
    os.replace(job_path, processing_path)

    # 1. Parse + schema-validate the job.
    try:
        job = Storage.read_json(processing_path)
        jsonschema_min.validate_or_raise(
            job, config.schema_path("bridge_job.schema.json"), "Bridge Job"
        )
    except (OSError, ValueError) as exc:
        return _emit_failure(
            processing_path, paths, base,
            job_id="", adapter=None,
            code=status_codes.JOB_SCHEMA_INVALID,
            error=str(exc),
            suggested_fix="Fix the job JSON or the bridge job schema.",
            started=started,
        )

    job_id = str(job.get("job_id", ""))
    adapter = job.get("adapter")
    runtime_args = _resolved_runtime_args(job, defaults)

    # 2. Pre-flight + dispatch.
    try:
        _check_backend(config, runtime_args)
        runtime = Runtime(
            config,
            prefer_backend=runtime_args.get("backend"),
            model_override=runtime_args.get("model"),
            verifier_override=runtime_args.get("verifier"),
        )
        status, adapter_result, repro_ref, trace_ref = _run_adapter(job, runtime)
    except BridgeError as exc:
        return _emit_failure(
            processing_path, paths, base,
            job_id=job_id, adapter=adapter,
            code=exc.code, error=exc.message,
            suggested_fix=exc.suggested_fix, started=started,
        )
    except Exception as exc:  # noqa: BLE001 - bridge must never crash the loop
        return _emit_failure(
            processing_path, paths, base,
            job_id=job_id, adapter=adapter,
            code=status_codes.INTERNAL_ERROR, error=str(exc),
            started=started,
        )

    # 3. Persist result; move job to archive.
    duration_ms = round((time.time() - started) * 1000, 2)
    safe_id = _safe_basename(job_id, fallback=base.rsplit(".", 1)[0])
    outbox_path = _unique_path(paths["outbox"], f"{safe_id}.result.json")
    input_ref = os.path.relpath(processing_path, config.storage_root)
    output_ref = os.path.relpath(outbox_path, config.storage_root)
    result = {
        "job_id": job_id,
        "status": status,
        "adapter": adapter,
        "result": adapter_result,
        "trace_pack_ref": trace_ref,
        "repro_pack_ref": repro_ref,
        "bridge": {
            "processed_at": now_iso(),
            "duration_ms": duration_ms,
            "input_ref": input_ref,
            "output_ref": output_ref,
        },
    }
    with open(outbox_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True, default=str)
    archive_path = _unique_path(paths["archive"], base)
    shutil.move(processing_path, archive_path)

    return {
        "job_id": job_id,
        "status": status,
        "adapter": adapter,
        "outbox": output_ref,
        "duration_ms": duration_ms,
    }


def _emit_failure(processing_path, paths, base, *, job_id, adapter, code,
                  error, started, suggested_fix=""):
    """Move a job to failed/ and write a machine-readable error report."""
    duration_ms = round((time.time() - started) * 1000, 2)
    safe_id = _safe_basename(job_id, fallback=base.rsplit(".", 1)[0] or "job")
    failed_job_path = _unique_path(paths["failed"], base)
    try:
        shutil.move(processing_path, failed_job_path)
    except OSError:
        # Source already gone (rare race); ignore and emit the error report.
        pass
    error_path = _unique_path(paths["failed"], f"{safe_id}.error.json")
    report = {
        "job_id": job_id,
        "adapter": adapter,
        "status": "fail",
        "code": code,
        "error": error,
        "suggested_fix": suggested_fix,
        "bridge": {
            "processed_at": now_iso(),
            "duration_ms": duration_ms,
            "input_ref": os.path.basename(failed_job_path),
        },
    }
    with open(error_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, default=str)
    return {
        "job_id": job_id,
        "status": "fail",
        "adapter": adapter,
        "code": code,
        "error": error,
        "duration_ms": duration_ms,
    }


# -- cycle / loop ----------------------------------------------------------
def process_cycle(config: Config, paths: dict, defaults: dict, max_jobs: int) -> list[dict]:
    """Process at most ``max_jobs`` ready files from the inbox."""
    return [
        process_job(path, config, paths, defaults)
        for path in _list_ready_jobs(paths["inbox"], max_jobs)
    ]


def run_loop(config, paths, defaults, *, poll_interval, max_jobs,
             max_cycles=None, on_report=None) -> int:
    """Poll the inbox until SIGINT (or ``max_cycles`` cycles, for tests).

    Returns the number of cycles run. Graceful Ctrl+C: a SIGINT handler sets
    a stop flag; the active cycle finishes, the loop exits cleanly, and the
    previous SIGINT handler is restored.
    """
    stopped = {"flag": False}

    def _stop(*_):
        stopped["flag"] = True

    prev_sigint = signal.signal(signal.SIGINT, _stop)
    cycles = 0
    try:
        while not stopped["flag"]:
            reports = process_cycle(config, paths, defaults, max_jobs)
            if on_report:
                for report in reports:
                    on_report(report)
            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                break
            if stopped["flag"]:
                break
            time.sleep(poll_interval)
    finally:
        signal.signal(signal.SIGINT, prev_sigint)
    return cycles
