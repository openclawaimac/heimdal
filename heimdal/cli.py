"""Heimdal command-line interface.

Commands (docs/builder_pack/04_runtime/CORE_RUNTIME_REQUIREMENTS.md):

    heimdal doctor [--json]
    heimdal run demo
    heimdal run --input <task.json>
    heimdal run --instruction "..."
    heimdal eval run
    heimdal verify --task <task.json> --answer <answer.json>
    heimdal openclaw run --input <openclaw_payload.json>
    heimdal openclaw capabilities [--json]
    heimdal openclaw doctor --input <openclaw_payload.json>
    heimdal hermes run --input <hermes_payload.json>
    heimdal hermes capabilities [--json]
    heimdal hermes doctor --input <hermes_payload.json>
    heimdal bridge init | submit --input <job.json> | once | watch | status
    heimdal patch validate <patch_file>
    heimdal truth list | add <file> | search "<query>"
    heimdal logs latest
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

from heimdal import __version__, bridge, jsonschema_min
from heimdal.adapters.cli_adapter import CLIAdapter
from heimdal.adapters.hermes_adapter import HermesAdapter
from heimdal.adapters.hermes_host import handle as run_hermes
from heimdal.adapters.openclaw_adapter import OpenClawAdapter
from heimdal.adapters.openclaw_host import handle as run_openclaw
from heimdal.config import load_config
from heimdal.core import eval_runner, intake, patch_manager
from heimdal.core.runtime import Runtime
from heimdal.hardware.profiler import detect_ollama, full_profile
from heimdal.ids import now_compact
from heimdal.retrieval.truth_store import TruthStore
from heimdal.storage import Storage


def _prefer_backend(args) -> str | None:
    """Resolve the backend preference from --backend / --offline flags."""
    if getattr(args, "backend", None):
        return args.backend
    if getattr(args, "offline", False):
        return "offline"
    return None


# -- doctor ----------------------------------------------------------------
def cmd_doctor(args) -> int:
    config = load_config(args.manifest)
    storage = Storage(config.storage_root).ensure()
    profile = full_profile(
        config,
        run_capability_tests=not args.no_capability_tests,
        capability_model=args.model,
    )
    profile_path = storage.write_json(
        f"logs/hardware_profiles/{now_compact()}.json", profile
    )
    profile["profile_path"] = profile_path

    if args.json:
        print(json.dumps(profile, indent=2))
        return 0

    os_info = profile["os"]
    gpu = profile["gpu"]
    ollama = profile["ollama"]
    print(f"Heimdal doctor (v{__version__})")
    print(f"  os            : {os_info['system']} / {os_info['flavour']} ({os_info['machine']})")
    print(f"  cpu           : {profile['cpu']['logical_cores']} cores - {profile['cpu']['model']}")
    print(f"  ram           : {profile['ram_gb']} GB")
    print(f"  disk          : {profile['disk_class']}")
    print(f"  gpu           : {gpu['count']} (cuda={gpu['cuda']}, metal={gpu['metal']})")
    print(f"  deployment    : {profile['deployment_mode']}")
    print(f"  ollama        : {ollama['base_url']} reachable={ollama['reachable']}")
    if ollama["models"]:
        print(f"  ollama models : {', '.join(ollama['models'])}")
    for test in profile["capability_tests"]:
        status = "ok" if test["passed"] else "FAIL"
        print(f"  capability    : {test['name']} on {test['model']} [{status}]")
    for warning in profile["warnings"]:
        print(f"  warning       : {warning}")
    print(f"  profile       : {profile_path}")
    return 0  # doctor always exits cleanly, even without Ollama


# -- run -------------------------------------------------------------------
def cmd_run(args) -> int:
    config = load_config(args.manifest)
    runtime = Runtime(
        config,
        prefer_backend=_prefer_backend(args),
        model_override=args.model,
        verifier_override=args.verifier,
    )
    adapter = CLIAdapter()

    if args.input:
        envelope = Storage.read_json(args.input)
        result = runtime.run_envelope(envelope)
    elif args.instruction:
        envelope = adapter.to_host_task_envelope(args.instruction)
        result = runtime.run_envelope(envelope)
    else:  # 'demo' or no target
        result = runtime.run_demo()

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(adapter.from_heimdal_result(result))
    return 0 if result["status"] in ("pass", "need_input") else 1


# -- eval ------------------------------------------------------------------
def cmd_eval(args) -> int:
    config = load_config(args.manifest)
    runtime = Runtime(
        config,
        prefer_backend=_prefer_backend(args),
        model_override=args.model,
        verifier_override=args.verifier,
    )
    summary = eval_runner.run_evals(runtime)
    if args.json:
        print(json.dumps(summary, indent=2, default=str))
        return 0
    print(f"Eval run {summary['eval_run_id']}")
    print(f"  total      : {summary['total']}")
    print(f"  passed     : {summary['passed']}")
    print(f"  pass_rate  : {summary['pass_rate']}")
    for category, stats in summary["categories"].items():
        flag = "ok" if stats["meets_minimum"] else "below-minimum"
        print(
            f"  {category:<10}: {stats['passed']}/{stats['total']} "
            f"(min {stats['minimum']}, {flag})"
        )
    print(f"  must_pass_all_passed: {summary['must_pass_all_passed']}")
    print(f"  regressed  : {summary['regressed']}")
    print(f"  summary    : {summary['summary_path']}")
    return 0 if summary["must_pass_all_passed"] else 1


# -- verify ----------------------------------------------------------------
def _answer_text(answer_doc) -> str:
    """Extract the candidate answer text from a loaded answer file."""
    if isinstance(answer_doc, dict) and "answer" in answer_doc:
        return str(answer_doc["answer"])
    if isinstance(answer_doc, str):
        return answer_doc
    return json.dumps(answer_doc)


def cmd_verify(args) -> int:
    config = load_config(args.manifest)
    runtime = Runtime(
        config,
        prefer_backend=_prefer_backend(args),
        model_override=args.model,
        verifier_override=args.verifier,
    )
    envelope = Storage.read_json(args.task)
    result = runtime.verify_envelope(envelope, _answer_text(Storage.read_json(args.answer)))

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"status  : {result['status']}")
        print(f"code    : {result['code']}")
        print(f"score   : {result['score']}")
        print(f"verifier: {result['verifier']['backend']}")
        for defect in result["defects"]:
            print(f"  defect: ({defect['severity']}) {defect['message']}")
        print(f"repro   : {result['repro_pack_ref']}")
        print(f"trace   : {result['trace_pack_ref']}")
    return 0 if result["status"] == "pass" else 1


# -- host doctor (shared by hermes + openclaw) -----------------------------
def _emit_doctor(status, checks, warnings, suggested, args):
    report = {
        "status": status,
        "checks": checks,
        "warnings": warnings,
        "suggested_fixes": suggested,
    }
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"status: {status}")
        for check in checks:
            mark = "ok  " if check["passed"] else "FAIL"
            extras = {
                k: v for k, v in check.items() if k not in ("name", "passed")
            }
            tail = f" {extras}" if extras else ""
            print(f"  [{mark}] {check['name']}{tail}")
        for warning in warnings:
            print(f"  warning: {warning}")
        for fix in suggested:
            print(f"  fix    : {fix}")
    return 1 if status == "fail" else 0


def _host_doctor(args, *, host_type: str) -> int:
    """Run Hermes/OpenClaw integration diagnostics for an input payload."""
    config = load_config(args.manifest)
    storage = Storage(config.storage_root).ensure()
    checks: list[dict] = []
    warnings: list[str] = []
    suggested: list[str] = []

    def add(name, passed, **data):
        checks.append({"name": name, "passed": passed, **data})

    if not args.input:
        return _emit_doctor(
            "fail",
            [{"name": "input_provided", "passed": False}],
            warnings,
            [f"'{host_type} doctor' requires --input <payload.json>"],
            args,
        )

    try:
        payload = Storage.read_json(args.input)
        add("payload_loaded", True)
    except (OSError, ValueError) as exc:
        add("payload_loaded", False, error=str(exc))
        suggested.append("Provide a valid JSON payload via --input.")
        return _emit_doctor("fail", checks, warnings, suggested, args)

    adapter = HermesAdapter() if host_type == "hermes" else OpenClawAdapter()
    try:
        envelope = adapter.to_host_task_envelope(payload)
        intake.intake(envelope, config)
        add("payload_valid", True)
    except (ValueError, KeyError, TypeError) as exc:
        add("payload_valid", False, error=str(exc))
        suggested.append(
            f"Fix the {host_type} payload to match the documented shape."
        )

    for sub in ("workspace", "logs/trace_packs", "logs/repro_packs"):
        path = storage.path(sub)
        writable = os.path.isdir(path) and os.access(path, os.W_OK)
        add(f"{sub.replace('/', '_')}_writable", writable, path=sub)
        if not writable:
            suggested.append(f"Ensure storage_root/{sub} exists and is writable.")

    if isinstance(payload, dict):
        callback = (payload.get("callback") or {}).get("file")
        if callback:
            add("callback_safe", True,
                target_ref=f"workspace/{os.path.basename(str(callback))}")

    verifier_mode = args.verifier
    if verifier_mode in (None, "rule_based", "hybrid"):
        add("verifier_mode_valid", True, mode=verifier_mode or "default")
    else:
        add("verifier_mode_valid", False, mode=verifier_mode)
        suggested.append("Use --verifier rule_based or hybrid.")

    if args.backend == "ollama":
        ollama = detect_ollama(config)
        if ollama["reachable"]:
            add("ollama_reachable", True, base_url=ollama["base_url"])
            if args.model:
                if args.model in ollama["models"]:
                    add("model_installed", True, model=args.model)
                else:
                    add("model_installed", False, model=args.model,
                        code="OLLAMA_MODEL_MISSING")
                    suggested.append(f"Run: ollama pull {args.model}")
            else:
                warnings.append(
                    "No --model specified; Heimdal will auto-select a candidate."
                )
        else:
            add("ollama_reachable", False, code="OLLAMA_UNREACHABLE",
                base_url=ollama["base_url"])
            suggested.append("Start Ollama or rerun with --backend offline.")

    if host_type == "hermes":
        try:
            jsonschema_min.load_schema(
                config.schema_path("hermes_result.schema.json")
            )
            add("hermes_schema_loadable", True)
        except (OSError, ValueError) as exc:
            add("hermes_schema_loadable", False, error=str(exc))

    payload_ok = any(c["name"] == "payload_valid" and c["passed"] for c in checks)
    if payload_ok:
        try:
            runtime = Runtime(
                config,
                prefer_backend=_prefer_backend(args),
                model_override=args.model,
                verifier_override=args.verifier,
            )
            host_fn = run_hermes if host_type == "hermes" else run_openclaw
            result = host_fn(payload, runtime)
            outcome = result.get("status") or result.get("outcome")
            add("end_to_end_run", True, outcome=outcome)

            blob = json.dumps(result, default=str)
            if storage.root in blob:
                add("no_absolute_paths", False)
                suggested.append("Host result exposed absolute paths.")
            else:
                add("no_absolute_paths", True)

            internal_types = {"context_packet", "task_contract"}
            leaked = [
                a.get("type") for a in result.get("artifacts", [])
                if a.get("type") in internal_types
            ]
            add("no_internal_artifacts", not leaked, leaked=leaked)
            if leaked:
                suggested.append(f"Internal artifacts exposed: {leaked}.")

            internal_fields = ("prompt", "system", "routing", "packet",
                               "context_packet", "models_used")
            leaked_fields = [k for k in internal_fields if k in result]
            add("no_internal_fields", not leaked_fields, leaked=leaked_fields)
            if leaked_fields:
                suggested.append(
                    f"Host result exposed internal fields: {leaked_fields}."
                )
        except (RuntimeError, OSError, ValueError) as exc:
            add("end_to_end_run", False, error=str(exc))
            suggested.append(
                "End-to-end run failed; adjust payload/backend and retry."
            )

    failed = [c for c in checks if not c["passed"]]
    if failed:
        status = "fail"
    elif warnings:
        status = "warning"
    else:
        status = "pass"
    return _emit_doctor(status, checks, warnings, suggested, args)


def _host_capabilities(*, host_type: str, args) -> int:
    config = load_config(args.manifest)
    ollama = detect_ollama(config)
    capabilities = {
        "heimdal_version": __version__,
        "supported_backends": ["offline", "ollama"],
        "supported_verifiers": ["rule_based", "hybrid"],
        "supports_need_input": True,
        "supports_needed_inputs": True,
        "supports_callback": True,
        "supports_verify_only": True,
        f"supports_{host_type}_adapter": True,
        "models": ollama["models"],
    }
    if args.json:
        print(json.dumps(capabilities, indent=2))
    else:
        for key, value in capabilities.items():
            print(f"  {key:<22}: {value}")
    return 0


# -- openclaw --------------------------------------------------------------
def cmd_openclaw(args) -> int:
    if args.openclaw_command == "capabilities":
        return _host_capabilities(host_type="openclaw", args=args)
    if args.openclaw_command == "doctor":
        return _host_doctor(args, host_type="openclaw")

    config = load_config(args.manifest)
    if not args.input:
        print("error: 'openclaw run' requires --input", file=sys.stderr)
        return 2
    payload = Storage.read_json(args.input)
    runtime = Runtime(
        config,
        prefer_backend=_prefer_backend(args),
        model_override=args.model,
        verifier_override=args.verifier,
    )
    result = run_openclaw(payload, runtime)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"outcome : {result['outcome']}")
        print(
            f"task    : {result['heimdal_task_id']} "
            f"(openclaw {result['openclaw_task_id']})"
        )
        print(f"summary : {result['summary']}")
        if result.get("code"):
            print(f"code    : {result['code']}")
        for question in result.get("questions", []):
            print(f"  question: {question}")
        _print_callback(result.get("callback_delivery"))
        if result.get("repro_pack_ref"):
            print(f"repro   : {result['repro_pack_ref']}")
        if result.get("trace_pack_ref"):
            print(f"trace   : {result['trace_pack_ref']}")
    return 0 if result["outcome"] in ("pass", "need_input") else 1


# -- hermes ----------------------------------------------------------------
def _print_callback(delivery) -> None:
    if delivery:
        print(f"callback: {delivery['status']} -> {delivery['target_ref']}")


def cmd_hermes(args) -> int:
    if args.hermes_command == "capabilities":
        return _host_capabilities(host_type="hermes", args=args)
    if args.hermes_command == "doctor":
        return _host_doctor(args, host_type="hermes")

    config = load_config(args.manifest)
    if not args.input:
        print("error: 'hermes run' requires --input", file=sys.stderr)
        return 2
    payload = Storage.read_json(args.input)
    runtime = Runtime(
        config,
        prefer_backend=_prefer_backend(args),
        model_override=args.model,
        verifier_override=args.verifier,
    )
    result = run_hermes(payload, runtime)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"status  : {result['status']}")
        print(f"code    : {result['code']}")
        print(f"session : {result['hermes_session_id']}")
        print(
            f"task    : {result['heimdal_task_id']} "
            f"(invocation {result['invocation_id']})"
        )
        print(f"message : {result['message']}")
        for question in result.get("questions", []):
            print(f"  question: {question}")
        print(f"verifier: {result['verifier']['backend']}")
        _print_callback(result.get("callback_delivery"))
        if result.get("repro_pack_ref"):
            print(f"repro   : {result['repro_pack_ref']}")
        if result.get("trace_pack_ref"):
            print(f"trace   : {result['trace_pack_ref']}")
    return 0 if result["status"] in ("pass", "need_input") else 1


# -- bridge ----------------------------------------------------------------
def _bridge_defaults(args) -> dict:
    return {
        "backend": _prefer_backend(args),
        "model": args.model,
        "verifier": args.verifier,
    }


def cmd_bridge(args) -> int:
    config = load_config(args.manifest)
    command = args.bridge_command

    if command == "init":
        paths = bridge.ensure_dirs(config)
        print(f"Initialised bridge at {paths['root']}")
        for sub in bridge.DIRS:
            print(f"  {sub:<11}: {paths[sub]}")
        print(
            "Examples: examples/bridge/*.example.json (copy one into "
            f"{paths['inbox']} with a .ready.json suffix to run)."
        )
        return 0

    if command == "status":
        counts = bridge.status_counts(config)
        for sub in bridge.DIRS:
            print(f"  {sub:<11}: {counts[sub]}")
        return 0

    if command == "submit":
        paths = bridge.resolve_paths(config, inbox=args.inbox, outbox=args.outbox)
        if not args.input:
            print("error: 'bridge submit' requires --input", file=sys.stderr)
            return 2
        target = bridge.submit_job(args.input, paths, config)
        print(f"submitted: {os.path.relpath(target, config.storage_root)}")
        return 0

    if command in ("once", "run", "watch"):
        paths = bridge.resolve_paths(config, inbox=args.inbox, outbox=args.outbox)
        defaults = _bridge_defaults(args)
        max_jobs = args.max_jobs or bridge.DEFAULT_MAX_JOBS_PER_CYCLE

        if command == "once":
            reports = bridge.process_cycle(config, paths, defaults, max_jobs)
            for report in reports:
                print(json.dumps(report, default=str))
            return 0

        # 'watch' is the canonical name; 'run' is kept for v0.2.8 compatibility.
        cycles = bridge.run_loop(
            config, paths, defaults,
            poll_interval=args.poll_interval,
            max_jobs=max_jobs,
            on_report=lambda r: print(json.dumps(r, default=str), flush=True),
        )
        print(f"bridge stopped after {cycles} cycles", file=sys.stderr)
        return 0

    return 2


# -- patch -----------------------------------------------------------------
def cmd_patch(args) -> int:
    config = load_config(args.manifest)
    ok, errors = patch_manager.validate_patch_file(args.patch_file, config)
    if ok:
        print(f"PASS: patch is valid - {args.patch_file}")
        return 0
    print(f"REJECTED: patch is invalid - {args.patch_file}")
    for error in errors:
        print(f"  - {error}")
    return 1


# -- truth -----------------------------------------------------------------
def cmd_truth(args) -> int:
    config = load_config(args.manifest)
    truth_dir = Storage(config.storage_root).ensure().path("truth")
    store = TruthStore(truth_dir)

    if args.truth_command == "list":
        sources = store.list_sources()
        if not sources:
            print(f"Truth Vault is empty ({truth_dir}).")
            return 0
        for source in sources:
            print(f"  {source['ref']}  ({source['size_bytes']} bytes)")
        print(f"{len(sources)} source(s) in {truth_dir}")
        return 0

    if args.truth_command == "add":
        if not args.argument:
            print("error: 'truth add' requires a file path", file=sys.stderr)
            return 2
        if not os.path.isfile(args.argument):
            print(f"error: file not found - {args.argument}", file=sys.stderr)
            return 2
        if not args.argument.lower().endswith((".md", ".txt")):
            print(
                "error: only .md and .txt files can be added to the Truth Vault",
                file=sys.stderr,
            )
            return 2
        dest = os.path.join(truth_dir, os.path.basename(args.argument))
        shutil.copy2(args.argument, dest)
        print(f"added: {os.path.basename(args.argument)} -> {dest}")
        return 0

    if args.truth_command == "search":
        if not args.argument:
            print("error: 'truth search' requires a query", file=sys.stderr)
            return 2
        hits = store.retrieve(args.argument)
        if not hits:
            print("No matching truth sources.")
            return 0
        for hit in hits:
            print(f"  [score {hit.score}] {hit.ref}")
        return 0

    return 2


# -- logs ------------------------------------------------------------------
def cmd_logs(args) -> int:
    config = load_config(args.manifest)
    storage = Storage(config.storage_root)
    trace_path = storage.latest("logs/trace_packs")
    repro_path = storage.latest("logs/repro_packs")
    if not trace_path:
        print("No runs logged yet.")
        return 0
    trace = Storage.read_json(trace_path)
    print(f"Latest run: trace {trace['id']} (task {trace['task_id']})")
    print(f"  status : {trace.get('status')}")
    print(f"  metrics: {json.dumps(trace.get('metrics', {}))}")
    print(f"  events : {len(trace.get('events', []))}")
    for event in trace.get("events", []):
        print(f"    - {event['ts']} {event['name']}")
    print(f"  trace_pack: {trace_path}")
    if repro_path:
        print(f"  repro_pack: {repro_path}")
    return 0


# -- parser ----------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="heimdal", description="Heimdal Engine CLI")
    parser.add_argument("--version", action="version", version=f"heimdal {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_doctor = sub.add_parser("doctor", help="profile hardware and model backend")
    p_doctor.add_argument("--json", action="store_true", help="emit JSON")
    capability = p_doctor.add_mutually_exclusive_group()
    capability.add_argument(
        "--capability-test",
        action="store_true",
        help="explicitly run model capability tests (the default)",
    )
    capability.add_argument(
        "--no-capability-tests", action="store_true", help="skip model capability tests"
    )
    p_doctor.add_argument("--model", help="model to use for capability tests")
    p_doctor.add_argument("--manifest", help="path to the Heimdal manifest")
    p_doctor.set_defaults(func=cmd_doctor)

    p_run = sub.add_parser("run", help="run a task through the Quality Factory")
    p_run.add_argument("target", nargs="?", help="'demo' (default when no input given)")
    p_run.add_argument("--input", help="path to a Host Task Envelope JSON file")
    p_run.add_argument("--instruction", help="run a plain instruction string")
    p_run.add_argument("--offline", action="store_true", help="force the offline backend")
    p_run.add_argument("--backend", choices=["ollama", "offline"], help="force a backend")
    p_run.add_argument("--model", help="override the worker model")
    p_run.add_argument(
        "--verifier", choices=["rule_based", "hybrid"], help="override the verifier mode"
    )
    p_run.add_argument("--json", action="store_true", help="emit the Result Envelope as JSON")
    p_run.add_argument("--manifest", help="path to the Heimdal manifest")
    p_run.set_defaults(func=cmd_run)

    p_eval = sub.add_parser("eval", help="run the eval suite")
    p_eval.add_argument("eval_command", choices=["run"])
    p_eval.add_argument("--offline", action="store_true", help="force the offline backend")
    p_eval.add_argument("--backend", choices=["ollama", "offline"], help="force a backend")
    p_eval.add_argument("--model", help="override the worker model")
    p_eval.add_argument(
        "--verifier", choices=["rule_based", "hybrid"], help="override the verifier mode"
    )
    p_eval.add_argument("--json", action="store_true", help="emit JSON")
    p_eval.add_argument("--manifest", help="path to the Heimdal manifest")
    p_eval.set_defaults(func=cmd_eval)

    p_verify = sub.add_parser(
        "verify", help="verify a host-supplied candidate answer against a task"
    )
    p_verify.add_argument(
        "--task", required=True, help="path to a Host Task Envelope JSON file"
    )
    p_verify.add_argument(
        "--answer", required=True, help="path to a candidate answer JSON file"
    )
    p_verify.add_argument("--offline", action="store_true", help="force the offline backend")
    p_verify.add_argument("--backend", choices=["ollama", "offline"], help="force a backend")
    p_verify.add_argument("--model", help="override the worker model")
    p_verify.add_argument(
        "--verifier", choices=["rule_based", "hybrid"], help="override the verifier mode"
    )
    p_verify.add_argument(
        "--json", action="store_true", help="emit the verification result as JSON"
    )
    p_verify.add_argument("--manifest", help="path to the Heimdal manifest")
    p_verify.set_defaults(func=cmd_verify)

    p_oc = sub.add_parser("openclaw", help="run a task from an OpenClaw payload")
    p_oc.add_argument(
        "openclaw_command", choices=["run", "capabilities", "doctor"]
    )
    p_oc.add_argument(
        "--input",
        help="path to an OpenClaw payload JSON file (required for 'run' / 'doctor')",
    )
    p_oc.add_argument("--offline", action="store_true", help="force the offline backend")
    p_oc.add_argument("--backend", choices=["ollama", "offline"], help="force a backend")
    p_oc.add_argument("--model", help="override the worker model")
    p_oc.add_argument(
        "--verifier", choices=["rule_based", "hybrid"], help="override the verifier mode"
    )
    p_oc.add_argument("--json", action="store_true", help="emit the OpenClaw result as JSON")
    p_oc.add_argument("--manifest", help="path to the Heimdal manifest")
    p_oc.set_defaults(func=cmd_openclaw)

    p_hermes = sub.add_parser("hermes", help="run a task from a Hermes payload")
    p_hermes.add_argument(
        "hermes_command", choices=["run", "capabilities", "doctor"]
    )
    p_hermes.add_argument(
        "--input",
        help="path to a Hermes payload JSON file (required for 'run' / 'doctor')",
    )
    p_hermes.add_argument("--offline", action="store_true", help="force the offline backend")
    p_hermes.add_argument("--backend", choices=["ollama", "offline"], help="force a backend")
    p_hermes.add_argument("--model", help="override the worker model")
    p_hermes.add_argument(
        "--verifier", choices=["rule_based", "hybrid"], help="override the verifier mode"
    )
    p_hermes.add_argument("--json", action="store_true", help="emit the Hermes result as JSON")
    p_hermes.add_argument("--manifest", help="path to the Heimdal manifest")
    p_hermes.set_defaults(func=cmd_hermes)

    p_bridge = sub.add_parser(
        "bridge", help="local file bridge for external local agents"
    )
    p_bridge.add_argument(
        "bridge_command",
        choices=["init", "submit", "once", "watch", "run", "status"],
    )
    p_bridge.add_argument(
        "--input", help="path to a bridge job JSON file (required for 'submit')"
    )
    p_bridge.add_argument(
        "--inbox", help="inbox directory (default: <storage>/bridge/inbox)"
    )
    p_bridge.add_argument(
        "--outbox", help="outbox directory (default: <storage>/bridge/outbox)"
    )
    p_bridge.add_argument(
        "--offline", action="store_true", help="force the offline backend"
    )
    p_bridge.add_argument(
        "--backend", choices=["ollama", "offline"], help="force a backend"
    )
    p_bridge.add_argument("--model", help="default worker model for jobs")
    p_bridge.add_argument(
        "--verifier", choices=["rule_based", "hybrid"], help="default verifier mode"
    )
    p_bridge.add_argument(
        "--poll-interval",
        type=float,
        default=bridge.DEFAULT_POLL_INTERVAL,
        help="seconds between polling cycles for 'run'",
    )
    p_bridge.add_argument(
        "--max-jobs", type=int, help="max jobs per cycle (default 16)"
    )
    p_bridge.add_argument("--manifest", help="path to the Heimdal manifest")
    p_bridge.set_defaults(func=cmd_bridge)

    p_patch = sub.add_parser("patch", help="patch tools")
    p_patch.add_argument("patch_command", choices=["validate"])
    p_patch.add_argument("patch_file", help="path to a patch JSON file")
    p_patch.add_argument("--manifest", help="path to the Heimdal manifest")
    p_patch.set_defaults(func=cmd_patch)

    p_truth = sub.add_parser("truth", help="manage the local Truth Vault")
    p_truth.add_argument("truth_command", choices=["list", "add", "search"])
    p_truth.add_argument(
        "argument", nargs="?", help="file path (add) or query string (search)"
    )
    p_truth.add_argument("--manifest", help="path to the Heimdal manifest")
    p_truth.set_defaults(func=cmd_truth)

    p_logs = sub.add_parser("logs", help="inspect run logs")
    p_logs.add_argument("logs_command", choices=["latest"])
    p_logs.add_argument("--manifest", help="path to the Heimdal manifest")
    p_logs.set_defaults(func=cmd_logs)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(f"error: file not found - {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
