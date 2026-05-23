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


if __name__ == "__main__":
    unittest.main()
