"""Heimdal command-line interface.

Commands (docs/builder_pack/04_runtime/CORE_RUNTIME_REQUIREMENTS.md):

    heimdal doctor [--json]
    heimdal run demo
    heimdal run --input <task.json>
    heimdal run --instruction "..."
    heimdal eval run
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

from heimdal import __version__
from heimdal.adapters.cli_adapter import CLIAdapter
from heimdal.config import load_config
from heimdal.core import eval_runner, patch_manager
from heimdal.core.runtime import Runtime
from heimdal.hardware.profiler import full_profile
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
