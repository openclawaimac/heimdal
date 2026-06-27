"""v0.5.0: Mirror Mode -- optional cloud-teacher comparison."""

import contextlib
import io
import json
import os
import tempfile
import unittest

from tests.helpers import temp_config, write_temp_manifest

from heimdal.cli import main
from heimdal.core.runtime import Runtime
from heimdal.mirror import redaction, runner as mirror_runner
from heimdal.mirror.provider import TeacherInput
from heimdal.mirror.stub_teacher import HallucinatingStub, StubTeacher
from heimdal.storage import Storage


def _seed_local_run(storage: Storage, *, case_id: str = "case-1",
                    response: str = "A queue is a FIFO data structure.",
                    objective: str = "Explain what a queue is.",
                    status: str = "pass",
                    role: str = "general") -> None:
    """Write the minimum artifact triplet Mirror's selector needs."""
    run_dir = storage.path(f"artifacts/run_{case_id}")
    os.makedirs(run_dir, exist_ok=True)
    storage.write_json(
        f"artifacts/run_{case_id}/task_contract.json",
        {
            "task_id": case_id, "role_id": role, "objective": objective,
            "constraints": {}, "expected_outputs": ["markdown_response"],
            "definition_of_done": ["Address the objective"],
        },
    )
    storage.write_json(
        f"artifacts/run_{case_id}/verification_result.json",
        {"status": status, "score": 0.9, "defects": []},
    )
    with open(os.path.join(run_dir, "response.md"), "w", encoding="utf-8") as fh:
        fh.write(response)


class TeacherStubTests(unittest.TestCase):
    def test_stub_is_deterministic(self):
        teacher = StubTeacher()
        input_ = TeacherInput(
            case_id="c1", task={"title": "T", "objective": "obj"},
            local_output="One. Two. Three.",
        )
        a = teacher.generate(input_)
        b = teacher.generate(input_)
        self.assertEqual(a.output, b.output)
        self.assertEqual(a.provider, "stub")

    def test_hallucinator_injects_specific_claim(self):
        teacher = HallucinatingStub()
        result = teacher.generate(TeacherInput(
            case_id="c1", task={"title": "T", "objective": "obj"},
            local_output="A short answer.",
        ))
        self.assertIn("$249.99", result.output)


class RedactionTests(unittest.TestCase):
    def test_strips_openai_key(self):
        text = "key=sk-proj-abcdefghijklmnopqrstuvwxyz0123"
        result = redaction.redact(text)
        self.assertNotIn("sk-proj-abcdefghijklmnopqrstuvwxyz", result.text)
        self.assertTrue(any(r["kind"] == "openai_api_key" for r in result.redactions))

    def test_strips_env_secret_assignment(self):
        result = redaction.redact("PASSWORD=hunter2supersecret")
        self.assertIn("[REDACTED:env_secret_assignment]", result.text)

    def test_strips_private_key_block(self):
        text = (
            "-----BEGIN RSA PRIVATE KEY-----\nMIIBOgIBAAJBA\n-----END RSA PRIVATE KEY-----"
        )
        result = redaction.redact(text)
        self.assertIn("[REDACTED:private_key_block]", result.text)

    def test_walks_dicts_recursively(self):
        cleaned, redactions = redaction.redact_payload({
            "task": {"objective": "ok"},
            "local_output": "TOKEN=ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        })
        self.assertNotIn("ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", cleaned["local_output"])
        self.assertTrue(redactions)


class MirrorRunnerTests(unittest.TestCase):
    def setUp(self):
        self.config = temp_config(tempfile.mkdtemp())
        self.storage = Storage(self.config.storage_root).ensure()

    def test_disabled_mirror_blocks_real_cloud_calls(self):
        # Default manifest has enabled=false; openai provider must be blocked.
        run = mirror_runner.run_mirror(
            self.config, source="mixed", teacher="openai",
        )
        self.assertTrue(run["blocked_reason"])
        self.assertEqual(run["usage"]["calls"], 0)

    def test_local_only_blocks_real_cloud_but_allows_stub(self):
        # Simulate enabled=true with local_only.
        self.config.manifest["mirror"] = dict(
            self.config.manifest.get("mirror", {}),
            enabled=True, privacy_mode="local_only",
        )
        blocked = mirror_runner.run_mirror(
            self.config, teacher="openai",
        )
        self.assertTrue(blocked["blocked_reason"])
        _seed_local_run(self.storage)
        ok = mirror_runner.run_mirror(self.config, teacher="stub")
        self.assertIsNone(ok["blocked_reason"])

    def test_dry_run_makes_zero_teacher_calls(self):
        _seed_local_run(self.storage)
        run = mirror_runner.run_mirror(
            self.config, dry_run=True, teacher="openai",
        )
        # Dry run is allowed even with cloud provider name + disabled mirror.
        self.assertIsNone(run["blocked_reason"])
        for call in run["teacher_calls"]:
            self.assertEqual(call["status"], "dry_run")
        self.assertEqual(run["usage"]["calls"], 0)

    def test_stub_run_writes_report_and_run_artifact(self):
        _seed_local_run(self.storage)
        run = mirror_runner.run_mirror(self.config, teacher="stub")
        self.assertTrue(run["teacher_calls"])
        report_path = os.path.join(
            self.storage.path("mirror/reports"),
            f"{run['mirror_run_id']}_report.json",
        )
        self.assertTrue(os.path.exists(report_path))
        run_path = os.path.join(
            self.storage.path("mirror/runs"),
            f"{run['mirror_run_id']}.json",
        )
        self.assertTrue(os.path.exists(run_path))

    def test_max_teacher_calls_caps_real_calls(self):
        for index in range(5):
            _seed_local_run(self.storage, case_id=f"c{index}")
        run = mirror_runner.run_mirror(
            self.config, teacher="stub", max_teacher_calls=2, limit=5,
        )
        passing = [c for c in run["teacher_calls"] if c.get("status") == "pass"]
        self.assertLessEqual(len(passing), 2)

    def test_redaction_runs_before_provider_call(self):
        _seed_local_run(
            self.storage,
            response="Secret key: sk-proj-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
        run = mirror_runner.run_mirror(self.config, teacher="stub")
        call = run["teacher_calls"][0]
        # Redactions are reported alongside the call.
        self.assertTrue(call.get("redactions"))
        self.assertTrue(any(
            r["kind"] == "openai_api_key" for r in call["redactions"]
        ))

    def test_store_teacher_outputs_false_inlines_no_external_file(self):
        _seed_local_run(self.storage)
        run = mirror_runner.run_mirror(self.config, teacher="stub")
        call = run["teacher_calls"][0]
        # store_teacher_outputs defaults to false -> inline the output and
        # don't leave a separate file behind.
        self.assertIn("teacher_output_inline", call)
        self.assertNotIn("teacher_output_ref", call)
        teacher_dir = self.storage.path("mirror/teacher_outputs")
        if os.path.isdir(teacher_dir):
            self.assertEqual(os.listdir(teacher_dir), [])

    def test_reports_never_contain_provider_api_keys(self):
        # Even if the env had a key, mirror reports must not include it.
        os.environ.pop("OPENAI_API_KEY", None)
        _seed_local_run(self.storage)
        run = mirror_runner.run_mirror(self.config, teacher="stub")
        blob = json.dumps(run)
        self.assertNotIn("sk-proj-", blob)
        self.assertNotIn("ANTHROPIC_API_KEY", blob)


class MirrorCLITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.manifest = write_temp_manifest(self.tmp, self.tmp)

    def test_mirror_run_dry_run_offline(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(
                ["mirror", "run", "--dry-run", "--json", "--manifest", self.manifest]
            )
        self.assertEqual(code, 0)
        run = json.loads(buf.getvalue())
        self.assertTrue(run["dry_run"])
        self.assertEqual(run["usage"]["calls"], 0)

    def test_mirror_list_after_run(self):
        # Seed a run.
        _ = mirror_runner.run_mirror(temp_config(self.tmp), teacher="stub")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(
                main(["mirror", "list", "--json", "--manifest", self.manifest]), 0
            )
        runs = json.loads(buf.getvalue())
        self.assertTrue(runs)

    def test_mirror_report_with_no_runs_errors_cleanly(self):
        self.assertEqual(
            main(["mirror", "report", "--manifest", self.manifest]), 2
        )


class DiffEngineTests(unittest.TestCase):
    """v0.5.1: scoring + diff comparison."""

    def test_score_output_returns_all_documented_dimensions(self):
        from heimdal.mirror import scoring
        scores = scoring.score_output(
            "A queue is a FIFO data structure.",
            {"objective": "Explain a queue.", "constraints": {}},
        )
        for dim in scoring.DIMENSIONS:
            self.assertIn(dim, scores)
            self.assertGreaterEqual(scores[dim], 0.0)
            self.assertLessEqual(scores[dim], 1.0)

    def test_teacher_with_better_structure_beats_local(self):
        from heimdal.mirror import diff_engine
        local = "A queue is a thing where stuff gets added and removed."
        teacher = (
            "# Queue\n\n## Key idea\n- FIFO: first-in, first-out.\n"
            "## Use cases\n- Job scheduling.\n- Buffering.\n"
        )
        diff = diff_engine.compare(
            case_id="c1", local_output=local, teacher_output=teacher,
            task={"objective": "Explain a queue.", "constraints": {}},
        )
        self.assertTrue(diff["teacher_better"])
        self.assertFalse(diff["teacher_hallucinated"])
        dims = [f["dimension"] for f in diff["findings"]
                if "teacher=" in f["finding"]]
        self.assertIn("structure_format", dims)

    def test_teacher_hallucination_disqualifies_teacher(self):
        from heimdal.mirror import diff_engine
        local = "I cannot find the documented price for Product Y in the Truth Vault."
        teacher = "The price is $249.99 effective 2024-03-15 per official guidance."
        diff = diff_engine.compare(
            case_id="c2", local_output=local, teacher_output=teacher,
            task={"objective": "State the price of Product Y.",
                  "constraints": {"requires_sources": True}},
        )
        # Teacher fabricated specifics with no sources -- diff must NOT
        # mark teacher better even if its other dimensions look fine.
        self.assertTrue(diff["teacher_hallucinated"])
        self.assertFalse(diff["teacher_better"])
        self.assertTrue(diff["local_better"])

    def test_local_no_guess_beats_teacher_guess_on_source_required_task(self):
        from heimdal.mirror import diff_engine
        local = (
            "I cannot find the refund policy for Product Y in the Truth Vault. "
            "Please provide the source."
        )
        teacher = "Refunds for Product Y are available within 30 days under policy 4.2."
        diff = diff_engine.compare(
            case_id="c3", local_output=local, teacher_output=teacher,
            task={"objective": "State the refund policy for Product Y.",
                  "constraints": {"requires_sources": True}},
        )
        # Teacher invented a policy reference without source -> hallucinated.
        # Local correctly hedged -> wins no_guess_behavior + factuality.
        self.assertTrue(diff["local_better"])


class ProposalBuilderTests(unittest.TestCase):
    def test_findings_carry_explicit_winner(self):
        from heimdal.mirror import diff_engine
        diff = diff_engine.compare(
            case_id="w", local_output="Short.",
            teacher_output="# T\n\n## Points\n- a\n- b\n## Caveat\nverify.",
            task={"objective": "Explain a queue.", "constraints": {}},
        )
        for f in diff["findings"]:
            self.assertIn(f["winner"], ("teacher_better", "local_better"))

    def test_local_win_dimension_does_not_spawn_proposal(self):
        # Regression: a teacher_better diff can still contain dimensions where
        # LOCAL won. Those must never produce "adopt the teacher" proposals.
        # (The old substring check on "teacher=" matched "vs teacher=" in
        # local-win findings and mis-fired.)
        from heimdal.mirror import proposal_builder
        diff = {
            "diff_id": "d", "case_id": "c", "teacher_better": True,
            "local_better": False, "mixed": False, "teacher_hallucinated": False,
            "findings": [
                {"dimension": "conciseness", "severity": "high",
                 "winner": "local_better",
                 "finding": "local=0.9 vs teacher=0.2",
                 "recommendation": "local was tighter"},
            ],
        }
        proposals = proposal_builder.build_proposals(
            diff, case={"case_id": "c", "task": {"role_id": "general"}},
        )
        # conciseness was a LOCAL win -> no proposal should come from it.
        self.assertEqual(proposals, [])

    def test_no_proposals_when_teacher_hallucinated(self):
        from heimdal.mirror import diff_engine, proposal_builder
        diff = diff_engine.compare(
            case_id="c-h", local_output="Local hedged answer.",
            teacher_output="Price is $249.99 on 2024-03-15.",
            task={"objective": "State the price.",
                  "constraints": {"requires_sources": True}},
        )
        self.assertEqual(
            proposal_builder.build_proposals(diff, case={"case_id": "c-h", "task": diff}),
            [],
        )

    def test_teacher_better_structure_produces_skill_proposal(self):
        from heimdal.mirror import diff_engine, proposal_builder
        diff = diff_engine.compare(
            case_id="c-s", local_output="Short answer.",
            teacher_output=(
                "# Queue\n\n## Key idea\n- FIFO.\n- Used for scheduling.\n"
                "## Caveat\nverify before use."
            ),
            task={"objective": "Explain a queue.", "constraints": {}},
        )
        proposals = proposal_builder.build_proposals(
            diff, case={"case_id": "c-s", "task": {"role_id": "general"}},
        )
        kinds = [p["kind"] for p in proposals]
        self.assertIn("skill_proposal", kinds)

    def test_diff_generated_patch_validates_against_patch_schema(self):
        from heimdal import jsonschema_min
        from heimdal.config import load_config
        from heimdal.mirror import diff_engine, proposal_builder
        # A case that triggers a patch_proposal (rubric on source_grounding).
        diff = diff_engine.compare(
            case_id="c-g",
            local_output="The refund policy is generous.",
            teacher_output=(
                "Refunds covered per (refund_policy.md). May vary -- verify "
                "before quoting. (refund_policy.md)"
            ),
            task={"objective": "Summarize the refund policy.",
                  "role_id": "research",
                  "constraints": {"requires_sources": True}},
        )
        proposals = proposal_builder.build_proposals(
            diff, case={"case_id": "c-g",
                        "task": {"role_id": "research",
                                 "objective": "Summarize the refund policy.",
                                 "constraints": {"requires_sources": True}}},
        )
        patch_proposals = [p for p in proposals if p["kind"] == "patch_proposal"]
        self.assertTrue(patch_proposals)
        config = load_config()
        schema = jsonschema_min.load_schema(config.schema_path("patch.schema.json"))
        for proposal in patch_proposals:
            errors = jsonschema_min.validate(proposal["patch"], schema)
            self.assertEqual(errors, [], f"{proposal['id']}: {errors}")


class MirrorRunWritesDiffsAndProposalsTests(unittest.TestCase):
    def test_mirror_run_emits_diffs_and_proposals_when_teacher_succeeds(self):
        config = temp_config(tempfile.mkdtemp())
        storage = Storage(config.storage_root).ensure()
        _seed_local_run(
            storage,
            response="Short, plain answer.",
            objective="Explain queues and stacks in depth.",
        )
        run = mirror_runner.run_mirror(config, teacher="stub")
        # The stub adds structure -> teacher_better on structure_format ->
        # at least one skill_proposal in the report + on disk.
        self.assertTrue(run["diffs"])
        kinds = {p["kind"] for p in run["proposals"]}
        self.assertIn("skill_proposal", kinds)

    def test_mirror_promote_proposal_installs_into_patches_experimental(self):
        config = temp_config(tempfile.mkdtemp())
        storage = Storage(config.storage_root).ensure()
        # Seed a case that drives a patch_proposal (source_grounding gap).
        _seed_local_run(
            storage,
            objective="Summarize the refund policy for Product Y.",
            response="Refunds are generous.",
            role="research",
        )
        # Stitch a requires_sources constraint into the contract so the
        # diff treats this as source-required.
        contract_path = os.path.join(
            storage.path("artifacts"), "run_case-1", "task_contract.json",
        )
        contract = Storage.read_json(contract_path)
        contract["constraints"] = {"requires_sources": True}
        contract["verification"] = {"requires_sources": True, "requires_citations": True}
        storage.write_json("artifacts/run_case-1/task_contract.json", contract)
        # Force the teacher to add inline source refs so the patch builder
        # sees teacher_better on source_grounding.
        from heimdal.mirror.provider import TeacherInput, TeacherResult
        class _GroundedTeacher:
            name = "stub"
            model = "test"
            def generate(self, input_: TeacherInput) -> TeacherResult:
                return TeacherResult(
                    provider="stub", model="test", status="pass",
                    output=(
                        "Per (refund_policy.md), refunds are available within "
                        "30 days; verify before quoting. (refund_policy.md)"
                    ),
                    usage={"input_tokens": 0, "output_tokens": 0,
                           "estimated_cost": 0.0},
                )

        # Monkey-patch the provider resolver for this run.
        import heimdal.mirror.runner as runner_mod
        original = runner_mod._resolve_provider
        runner_mod._resolve_provider = lambda *a, **kw: _GroundedTeacher()
        try:
            run = mirror_runner.run_mirror(config, teacher="stub")
        finally:
            runner_mod._resolve_provider = original
        patch_proposals = [p for p in run["proposals"]
                           if p["kind"] == "patch_proposal"]
        self.assertTrue(patch_proposals, f"expected a patch_proposal in {run['proposals']}")
        # Promoting via the CLI lands the inner patch in patches/experimental/.
        proposal = patch_proposals[0]
        from heimdal.core import patch_manager
        patch_manager.install_patch(config, proposal["patch"])
        found = patch_manager.find_patch(config, proposal["patch"]["id"])
        self.assertIsNotNone(found)
        self.assertEqual(found[1], "experimental")


if __name__ == "__main__":
    unittest.main()
