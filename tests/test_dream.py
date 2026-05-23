"""v0.4.0: Dream Mode -- offline failure mining + proposal generation.

Dream Mode reads existing Trace Packs, Repro Packs, eval summaries, and
bridge failure reports, then writes structured improvement proposals. It is
never allowed to mutate stable state.
"""

import contextlib
import io
import json
import os
import tempfile
import unittest

from tests.helpers import temp_config, write_temp_manifest

from heimdal import jsonschema_min
from heimdal.cli import main
from heimdal.dream import analysis, proposer, runner as dream_runner
from heimdal.ids import new_id
from heimdal.storage import Storage


def _write_trace(storage: Storage, *, task_id: str, events: list[dict],
                 status: str = "fail") -> str:
    trace = {
        "id": new_id("trace"),
        "task_id": task_id,
        "events": events,
        "status": status,
        "metrics": {},
    }
    return storage.write_json(f"logs/trace_packs/{trace['id']}.json", trace)


class DreamAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.config = temp_config(tempfile.mkdtemp())
        self.storage = Storage(self.config.storage_root).ensure()

    def test_no_inputs_yields_empty_patterns(self):
        mining = analysis.gather_inputs(self.storage, source="mixed", limit=10)
        self.assertEqual(analysis.detect_patterns(mining), [])

    def test_detects_missing_source_from_trace_no_guess_gate(self):
        _write_trace(
            self.storage,
            task_id="t1",
            events=[
                {"ts": "x", "name": "no_guess_gate",
                 "data": {"outcome": "need_input", "code": "SOURCE_MISSING",
                          "reason": "no sources retrieved"}}
            ],
            status="need_input",
        )
        mining = analysis.gather_inputs(self.storage, source="mixed", limit=10)
        patterns = analysis.detect_patterns(mining)
        categories = {p["category"] for p in patterns}
        self.assertIn("missing_source", categories)

    def test_detects_weak_retrieval_from_insufficient_coverage_code(self):
        _write_trace(
            self.storage,
            task_id="t2",
            events=[
                {"ts": "x", "name": "no_guess_gate",
                 "data": {"outcome": "need_input",
                          "code": "SOURCE_SUPPORT_INSUFFICIENT",
                          "reason": "coverage 33% < 50%"}}
            ],
            status="need_input",
        )
        mining = analysis.gather_inputs(self.storage, source="mixed", limit=10)
        categories = {p["category"] for p in analysis.detect_patterns(mining)}
        self.assertIn("weak_retrieval", categories)

    def test_detects_semantic_miss_from_trace_semantic_verify_event(self):
        _write_trace(
            self.storage,
            task_id="t3",
            events=[
                {"ts": "x", "name": "semantic_verify",
                 "data": {"semantic_verifier_status": "fail",
                          "semantic_verifier_score": 0.2}}
            ],
            status="fail",
        )
        mining = analysis.gather_inputs(self.storage, source="mixed", limit=10)
        categories = {p["category"] for p in analysis.detect_patterns(mining)}
        self.assertIn("semantic_miss", categories)


class DreamProposerTests(unittest.TestCase):
    def test_proposals_for_missing_source_include_eval_case_proposal(self):
        patterns = [{
            "category": "missing_source", "count": 2,
            "description": "x",
            "examples": [{"task_id": "t1", "detail": "no sources", "source_ref": "logs/x"}],
        }]
        proposals = proposer.generate_proposals(patterns)
        self.assertTrue(any(p["kind"] == "eval_case_proposal" for p in proposals))

    def test_synthetic_proposal_is_low_risk_skill(self):
        synthetic = proposer.synthetic_proposal()
        self.assertEqual(synthetic["kind"], "skill_proposal")
        self.assertEqual(synthetic["risk_level"], "low")


class DreamRunnerTests(unittest.TestCase):
    def setUp(self):
        self.config = temp_config(tempfile.mkdtemp())
        self.storage = Storage(self.config.storage_root).ensure()

    def test_run_dream_with_no_inputs_still_writes_synthetic_proposal(self):
        # Spec: "at least one synthetic proposal can be generated if no failures exist".
        report = dream_runner.run_dream(self.config, source="mixed")
        self.assertEqual(report["failure_patterns"], [])
        all_proposals = (
            report["patch_proposals"] + report["skill_proposals"]
            + report["eval_case_proposals"]
        )
        self.assertEqual(len(all_proposals), 1)
        # Report file is written and schema-valid.
        report_path = os.path.join(
            self.storage.path("dream/reports"),
            f"{report['dream_run_id']}_report.json",
        )
        self.assertTrue(os.path.exists(report_path))
        # The dream-run handle is also written.
        run_path = os.path.join(
            self.storage.path("dream/runs"),
            f"{report['dream_run_id']}.json",
        )
        self.assertTrue(os.path.exists(run_path))

    def test_run_dream_emits_proposals_when_failures_are_mined(self):
        _write_trace(
            self.storage, task_id="t-missing",
            events=[{"ts": "x", "name": "no_guess_gate",
                     "data": {"outcome": "need_input", "code": "SOURCE_MISSING"}}],
            status="need_input",
        )
        report = dream_runner.run_dream(
            self.config, source="mixed", count=5,
        )
        self.assertTrue(report["failure_patterns"])
        proposals = (
            report["patch_proposals"] + report["skill_proposals"]
            + report["eval_case_proposals"]
        )
        self.assertTrue(proposals)
        # Each emitted proposal lives in storage/dream/proposals/.
        proposals_dir = self.storage.path("dream/proposals")
        files = os.listdir(proposals_dir)
        self.assertEqual(len(files), len(proposals))
        # The report itself validates against the dream report schema.
        errors = jsonschema_min.validate(
            report,
            jsonschema_min.load_schema(
                self.config.schema_path("dream_report.schema.json")
            ),
        )
        self.assertEqual(errors, [])

    def test_list_dream_runs_returns_runs_newest_first(self):
        first = dream_runner.run_dream(self.config, source="mixed")
        second = dream_runner.run_dream(self.config, source="mixed")
        runs = dream_runner.list_dream_runs(self.config)
        self.assertEqual(len(runs), 2)
        self.assertEqual(runs[0]["dream_run_id"], second["dream_run_id"])
        self.assertEqual(runs[1]["dream_run_id"], first["dream_run_id"])

    def test_patch_proposals_are_installed_into_experimental_channel(self):
        # v0.4.1 fix: Dream-generated patch_proposal must register the inner
        # patch into storage/patches/experimental/ so the patch lifecycle
        # CLI can see it without manual surgery.
        _write_trace(
            self.storage, task_id="t-weak",
            events=[{"ts": "x", "name": "no_guess_gate",
                     "data": {"outcome": "need_input",
                              "code": "SOURCE_SUPPORT_INSUFFICIENT",
                              "reason": "weak coverage"}}],
            status="need_input",
        )
        report = dream_runner.run_dream(self.config, source="mixed", count=5)
        patch_proposals = report["patch_proposals"]
        self.assertTrue(patch_proposals, "expected weak_retrieval to yield a patch_proposal")
        from heimdal.core import patch_manager
        for proposal in patch_proposals:
            patch_id = proposal["patch"]["id"]
            found = patch_manager.find_patch(self.config, patch_id)
            self.assertIsNotNone(found, f"Dream patch {patch_id} missing from lifecycle")
            patch, channel, _ = found
            self.assertEqual(channel, "experimental")
            # The inner patch must carry the v0.4.1-required metadata so
            # beta promotion doesn't bounce on missing intent / rollback.
            self.assertTrue(patch["intent"], "Dream patch missing 'intent'")
            self.assertIn("rollback", patch)
            self.assertEqual(patch["created_by"], "dream_mode")

    def test_dream_mode_never_modifies_stable_storage(self):
        # No writes outside storage/dream/ for a Dream run.
        sentinel = self.storage.write_json("patches/stable/sentinel.json", {"id": "s"})
        baseline = os.path.getmtime(sentinel)
        dream_runner.run_dream(self.config, source="mixed")
        self.assertEqual(os.path.getmtime(sentinel), baseline)


class DreamCLITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.manifest = write_temp_manifest(self.tmp, self.tmp)

    def test_dream_run_then_list_then_report(self):
        run_buf = io.StringIO()
        with contextlib.redirect_stdout(run_buf):
            code = main(["dream", "run", "--json", "--manifest", self.manifest])
        self.assertEqual(code, 0)
        report = json.loads(run_buf.getvalue())
        self.assertIn("dream_run_id", report)

        list_buf = io.StringIO()
        with contextlib.redirect_stdout(list_buf):
            self.assertEqual(
                main(["dream", "list", "--json", "--manifest", self.manifest]), 0
            )
        runs = json.loads(list_buf.getvalue())
        self.assertTrue(any(r["dream_run_id"] == report["dream_run_id"] for r in runs))

        report_buf = io.StringIO()
        with contextlib.redirect_stdout(report_buf):
            self.assertEqual(
                main(
                    [
                        "dream", "report", "--id", report["dream_run_id"],
                        "--json", "--manifest", self.manifest,
                    ]
                ), 0
            )
        loaded = json.loads(report_buf.getvalue())
        self.assertEqual(loaded["dream_run_id"], report["dream_run_id"])

    def test_dream_report_with_no_runs_errors_cleanly(self):
        self.assertEqual(
            main(["dream", "report", "--manifest", self.manifest]), 2
        )


if __name__ == "__main__":
    unittest.main()
