"""Heimdal Core Runtime.

End-to-end Work Mode execution: validate the Host Task Envelope, resolve the
role, build the Task Contract, run the Quality Factory, and emit a Heimdal
Result Envelope plus Repro/Trace packs.
"""

from __future__ import annotations

import os
import shutil
import time

from heimdal.config import Config, load_config
from heimdal.core import (
    context_os,
    intake,
    model_router,
    quality_factory,
    repro_trace,
    status_codes,
    verifier,
)
from heimdal.core.constants import FAIL, NEED_INPUT, PASS
from heimdal.core.role_binding import resolve_role
from heimdal.core.scheduler import WORK, Scheduler
from heimdal.core.task_contract import build_contract
from heimdal.hardware.profiler import quick_profile
from heimdal.ids import new_id, repo_root, sha256_obj
from heimdal.hardware import runtime_profile
from heimdal.hardware.role_assigner import assigned_worker_model
from heimdal.models.base import select_backend
from heimdal.skills.registry import SkillRegistry
from heimdal.storage import Storage

DEMO_TASK = "examples/tasks/simple_task.json"


def _seed_storage(storage: Storage) -> None:
    """Populate truth/skills from bundled examples on first run.

    Truth files copy flat (one .md/.txt per file). Skills can be organized
    under role subdirectories (Skill Library 2.0), so the skills copy walks
    recursively and preserves layout. Existing non-empty destinations are
    skipped so a user's edits are never overwritten.
    """
    src_truth = os.path.join(repo_root(), "examples/truth")
    dst_truth = storage.path("truth")
    if os.path.isdir(src_truth) and not (os.path.isdir(dst_truth) and os.listdir(dst_truth)):
        for name in os.listdir(src_truth):
            shutil.copy2(os.path.join(src_truth, name), os.path.join(dst_truth, name))

    src_skills = os.path.join(repo_root(), "examples/skills")
    dst_skills = storage.path("skills")
    if not os.path.isdir(src_skills):
        return
    # "Empty" for skill seeding ignores the role subdirectories the layout
    # creates up front -- otherwise seed never fires after Storage.ensure().
    existing = [
        name for name in os.listdir(dst_skills)
        if os.path.isfile(os.path.join(dst_skills, name))
        or (
            os.path.isdir(os.path.join(dst_skills, name))
            and os.listdir(os.path.join(dst_skills, name))
        )
    ]
    if existing:
        return
    for root, _dirs, files in os.walk(src_skills):
        rel = os.path.relpath(root, src_skills)
        dest_root = dst_skills if rel == "." else os.path.join(dst_skills, rel)
        os.makedirs(dest_root, exist_ok=True)
        for name in files:
            shutil.copy2(os.path.join(root, name), os.path.join(dest_root, name))


class Runtime:
    """The Heimdal Core Runtime."""

    def __init__(
        self,
        config: Config | None = None,
        prefer_backend: str | None = None,
        model_override: str | None = None,
        verifier_override: str | None = None,
    ):
        self.config = config or load_config()
        self.storage = Storage(self.config.storage_root).ensure()
        _seed_storage(self.storage)
        self.backend = select_backend(self.config, prefer=prefer_backend)
        # v0.6.1: when no explicit --model is given, fall back to whatever
        # `heimdal models assign --write` recorded as the worker role.
        # Explicit CLI / programmatic overrides always win.
        self.assignment_source = "explicit_cli" if model_override else None
        if not model_override and self.backend.name == "ollama":
            assigned_model, source = assigned_worker_model(self.storage)
            if assigned_model:
                model_override = assigned_model
                self.assignment_source = source or "auto_tuner"
        self.model_override = model_override
        self.verifier_override = verifier_override
        self.scheduler = Scheduler(self.config)
        # Hardware does not change during a session; profile once and reuse.
        self.hardware_profile = quick_profile(self.config)
        # v0.6.2: active runtime profile (auto / manual / manifest). Limits
        # are layered onto task contracts as profile_overrides so a fresh
        # machine downsizes budgets without an explicit --quality flag.
        self.runtime_profile = runtime_profile.active(
            self.storage, self.config,
            manifest_profiles=self.config.runtime_profiles,
        )

    # -- public API --------------------------------------------------------
    def run_envelope(self, envelope: dict, mode: str = WORK) -> dict:
        """Run one Host Task Envelope and return a Heimdal Result Envelope."""
        started = time.time()
        validated = intake.intake(envelope, self.config)
        role = resolve_role(validated.get("role_binding", {}))
        contract = build_contract(validated, role, self.config, profile_overrides=runtime_profile.task_contract_overrides(self.runtime_profile["limits"]))

        self.scheduler.submit(mode, contract["task_id"])
        allowed, reason = self.scheduler.can_run(mode)
        if not allowed:
            return self._envelope(
                validated=validated,
                contract=contract,
                status=FAIL,
                code=None,
                needed_inputs=[],
                message=reason,
                artifacts=[],
                questions=[],
                repro={},
                trace={},
                metrics={},
            )

        trace = repro_trace.TraceBuilder(contract["task_id"])
        trace.event("intake_ok", host=validated.get("host", {}).get("type"))
        trace.event("role_resolved", role_id=role["role_id"])

        outcome = quality_factory.run_quality_factory(
            contract,
            role,
            validated,
            self.backend,
            self.storage,
            self.config,
            trace,
            model_override=self.model_override,
            verifier_override=self.verifier_override,
        )

        run_id = new_id("run")
        routing = outcome["routing"]
        artifacts = self._persist_artifacts(run_id, contract, outcome)
        duration_ms = round((time.time() - started) * 1000, 2)
        metrics = {
            "duration_ms": duration_ms,
            "repair_iterations": outcome["repair_iterations"],
            "verification_score": outcome["verification"]["score"],
            "backend": self.backend.name,
            "quality_level": routing["quality_level"],
            "worker_model": routing["worker_model"],
            "verifier_backend": routing["verifier_backend"],
            "semantic_verifier_model": routing["semantic_verifier_model"],
            "assignment_source": self.assignment_source,
            "runtime_profile": self.runtime_profile["name"],
            "profile_source": self.runtime_profile["source"],
            "profile_limits": self.runtime_profile["limits"],
        }

        repro = repro_trace.build_repro_pack(
            models=outcome["models_used"],
            params={
                "quality_level": routing["quality_level"],
                "verifier_strictness": routing["verifier_strictness"],
                "verifier_backend": routing["verifier_backend"],
                "semantic_verifier_model": routing["semantic_verifier_model"],
                "worker_model": routing["worker_model"],
                "samples": routing["samples"],
                "max_repair_iterations": routing["max_repair_iterations"],
            },
            hashes={
                "contract": sha256_obj(contract),
                "context_packet": outcome["packet"]["hashes"]["packet"],
            },
            hardware_profile=self.hardware_profile,
            retrieval_refs=context_os.retrieval_refs(outcome["packet"]),
            selected_skills=[
                {"skill_id": s["skill_id"], "source": s.get("source", "registry")}
                for s in outcome["packet"].get("skills_context", [])
            ],
        )
        trace_pack = trace.build(outcome["status"], metrics)
        pack_paths = repro_trace.write_packs(
            self.storage, self.config, repro, trace_pack
        )

        # v0.4.2: bump per-skill usage stats. Pass = the task itself passed
        # verification; any other status counts as fail for the skill's record.
        selected_skill_ids = [
            s["skill_id"] for s in outcome["packet"].get("skills_context", [])
        ]
        if selected_skill_ids:
            SkillRegistry(self.storage.path("skills")).record_usage(
                selected_skill_ids, passed=outcome["status"] == PASS
            )

        return self._envelope(
            validated=validated,
            contract=contract,
            status=outcome["status"],
            code=outcome.get("code"),
            needed_inputs=outcome.get("needed_inputs", []),
            message=self._message(outcome),
            artifacts=artifacts,
            questions=outcome.get("questions", []),
            repro={"id": repro["id"], "path": pack_paths["repro_pack"]},
            trace={"id": trace_pack["id"], "path": pack_paths["trace_pack"]},
            metrics=metrics,
        )

    def run_demo(self) -> dict:
        """Run the built-in demo task."""
        demo_path = os.path.join(repo_root(), DEMO_TASK)
        envelope = Storage.read_json(demo_path)
        return self.run_envelope(envelope)

    def verify_envelope(self, envelope: dict, answer_text: str) -> dict:
        """Verify a host-supplied candidate answer against a task.

        Heimdal does not draft the answer here: a host (e.g. Hermes) hands in
        its own candidate and asks Heimdal's verifier to judge it. Returns the
        Verification Result, a machine-readable code, and host-safe Repro /
        Trace pack refs for the verification run.
        """
        started = time.time()
        validated = intake.intake(envelope, self.config)
        role = resolve_role(validated.get("role_binding", {}))
        contract = build_contract(validated, role, self.config, profile_overrides=runtime_profile.task_contract_overrides(self.runtime_profile["limits"]))

        trace = repro_trace.TraceBuilder(contract["task_id"])
        trace.event("verify_intake_ok", host=validated.get("host", {}).get("type"))
        trace.event("role_resolved", role_id=role["role_id"])
        trace.event("contract_ready", contract_id=contract["contract_id"])

        packet = context_os.build_packet(
            contract, role, validated, self.storage, self.config
        )
        trace.event("context_packet_ready", packet_id=packet["packet_id"])

        routing = model_router.route(
            contract, role, self.backend, self.config,
            self.model_override, self.verifier_override,
        )
        trace.event("routing", **routing)

        self.backend.event_sink = trace.event
        try:
            verification = verifier.verify(
                answer_text, contract, packet, routing, self.config, self.backend
            )
        finally:
            self.backend.event_sink = None

        semantic = verification.get("semantic")
        models: list[dict] = []
        if semantic is not None:
            trace.event(
                "semantic_verify",
                semantic_verifier_model=semantic["model"],
                semantic_verifier_status=semantic["status"],
                semantic_verifier_score=semantic["score"],
                semantic_verifier_confidence=semantic["confidence"],
            )
            models.append(
                {
                    "role": "semantic_verifier",
                    "model": semantic["model"],
                    "backend": self.backend.name,
                }
            )
        trace.event("verify", status=verification["status"], score=verification["score"])

        status = verification["status"]
        code = status_codes.OK if status == PASS else status_codes.fail_code(verification)
        metrics = {
            "duration_ms": round((time.time() - started) * 1000, 2),
            "verification_score": verification["score"],
            "backend": self.backend.name,
            "quality_level": routing["quality_level"],
            "verifier_backend": routing["verifier_backend"],
            "semantic_verifier_model": routing["semantic_verifier_model"],
        }
        repro = repro_trace.build_repro_pack(
            models=models,
            params={
                "quality_level": routing["quality_level"],
                "verifier_strictness": routing["verifier_strictness"],
                "verifier_backend": routing["verifier_backend"],
                "semantic_verifier_model": routing["semantic_verifier_model"],
                "worker_model": routing["worker_model"],
                "samples": routing["samples"],
                "max_repair_iterations": routing["max_repair_iterations"],
            },
            hashes={
                "contract": sha256_obj(contract),
                "context_packet": packet["hashes"]["packet"],
            },
            hardware_profile=self.hardware_profile,
            retrieval_refs=context_os.retrieval_refs(packet),
            selected_skills=[
                {"skill_id": s["skill_id"], "source": s.get("source", "registry")}
                for s in packet.get("skills_context", [])
            ],
        )
        trace_pack = trace.build(status, metrics)
        pack_paths = repro_trace.write_packs(
            self.storage, self.config, repro, trace_pack
        )

        return {
            "task_id": contract["task_id"],
            "status": status,
            "code": code,
            "score": verification["score"],
            "defects": verification["defects"],
            "schema_errors": verification.get("schema_errors", []),
            "verifier": {
                "backend": routing["verifier_backend"],
                "semantic_model": routing["semantic_verifier_model"],
            },
            "repro_pack_ref": os.path.relpath(pack_paths["repro_pack"], self.storage.root),
            "trace_pack_ref": os.path.relpath(pack_paths["trace_pack"], self.storage.root),
            "metrics": metrics,
        }

    # -- helpers -----------------------------------------------------------
    def _persist_artifacts(self, run_id: str, contract: dict, outcome: dict) -> list[dict]:
        base = f"artifacts/{run_id}"
        artifacts = [
            {
                "type": "task_contract",
                "path": self.storage.write_json(f"{base}/task_contract.json", contract),
            },
            {
                "type": "context_packet",
                "path": self.storage.write_json(
                    f"{base}/context_packet.json", outcome["packet"]
                ),
            },
            {
                "type": "verification_result",
                "path": self.storage.write_json(
                    f"{base}/verification_result.json", outcome["verification"]
                ),
            },
        ]
        if outcome.get("output_text"):
            response_path = self.storage.path(f"{base}/response.md")
            with open(response_path, "w", encoding="utf-8") as fh:
                fh.write(outcome["output_text"])
            artifacts.append({"type": "response", "path": response_path})
        return artifacts

    @staticmethod
    def _message(outcome: dict) -> str:
        if outcome["status"] == NEED_INPUT:
            return outcome["questions"][0] if outcome["questions"] else "Input required."
        if outcome["status"] == PASS:
            return "Task completed and verified PASS."
        defects = outcome["verification"].get("defects", [])
        top = defects[0]["message"] if defects else "Verification failed."
        return f"Task did not pass verification: {top}"

    @staticmethod
    def _envelope(
        *, validated, contract, status, code, needed_inputs, message, artifacts,
        questions, repro, trace, metrics
    ) -> dict:
        return {
            "task_id": contract["task_id"],
            "host_task_id": validated.get("host", {}).get("host_task_id", ""),
            "status": status,
            "code": code,
            "message": message,
            "artifacts": artifacts,
            "questions": questions,
            "needed_inputs": needed_inputs,
            "repro_pack": repro,
            "trace_pack": trace,
            "metrics": metrics,
        }
