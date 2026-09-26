"""An eval run that hits a dead backend measured availability, not quality.

Scoring an outage as a quality failure is worse than useless: the run
reports pass_rate 0.0, flags a regression that never happened, and -- if
adopted as the baseline -- sets a bar so low that a later real regression
clears it. These tests pin the separation.
"""

import json
import os
import tempfile
import unittest
from unittest import mock

from tests.helpers import temp_config

from heimdal.core import eval_runner, model_router, patch_manager, status_codes
from heimdal.core.runtime import Runtime
from heimdal.models.ollama import OllamaError
from heimdal.storage import Storage

_EVAL_DIR = os.path.join(tempfile.mkdtemp(), "eval")


def _tiny_suite(quality_level: str = "B1") -> str:
    """A two-case suite, so these tests do not run the full 40."""
    directory = os.path.join(_EVAL_DIR, quality_level)
    os.makedirs(directory, exist_ok=True)
    cases = [
        {"id": "e1", "quality_level": quality_level,
         "instruction": "Explain what a queue is and how it behaves."},
        {"id": "e2", "quality_level": quality_level,
         "instruction": "Explain what a stack is and how it behaves."},
    ]
    for category, filename in eval_runner.CATEGORY_FILES.items():
        payload = cases if category in ("smoke", "must_pass") else []
        with open(os.path.join(directory, filename), "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
    return directory


def _envelope(task_id: str, quality_level: str = "B1") -> dict:
    return {
        "host": {"type": "cli", "host_task_id": task_id,
                 "source_agent": None, "callback": {}},
        "role_binding": {"role_id": "general", "risk_mode": "balanced",
                         "privacy_mode": "local_only",
                         "output_profiles": ["markdown"]},
        "task_request": {"task_id": task_id, "title": "Outage demo",
                         "instruction": "Explain what a queue is and how it behaves.",
                         "inputs": {}, "constraints": {}, "priority": "P2",
                         "budget": {"quality_level": quality_level},
                         "expected_outputs": ["markdown_response"]},
        "runtime_hints": {},
    }


def _runtime() -> Runtime:
    return Runtime(temp_config(tempfile.mkdtemp()), prefer_backend="offline")


def _outage(*_args, **_kwargs):
    raise OllamaError("Ollama went away.", code=status_codes.OLLAMA_UNREACHABLE)


def _write_summary(runtime: Runtime, name: str, **fields) -> None:
    directory = runtime.storage.path(f"eval_runs/{name}")
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(fields, fh)


class DegradedRunTests(unittest.TestCase):
    def _degraded_run(self, runtime=None) -> dict:
        runtime = runtime or _runtime()
        with mock.patch("heimdal.models.offline.OfflineBackend.generate", _outage):
            return eval_runner.run_evals(runtime, eval_dir=_tiny_suite())

    def test_outage_cases_are_errors_not_quality_failures(self):
        summary = self._degraded_run()
        actuals = {r["actual"] for r in summary["results"]}
        self.assertEqual(actuals, {"error"})
        self.assertNotIn("fail", actuals)
        for row in summary["results"]:
            self.assertEqual(row["code"], status_codes.OLLAMA_UNREACHABLE)

    def test_the_run_is_flagged_and_counted(self):
        summary = self._degraded_run()
        self.assertTrue(summary["backend_degraded"])
        self.assertEqual(summary["backend_failures"], summary["total"])
        self.assertEqual(summary["backend_codes"],
                         [status_codes.OLLAMA_UNREACHABLE])

    def test_an_outage_is_never_reported_as_a_regression(self):
        runtime = _runtime()
        _write_summary(runtime, "healthy", pass_rate=1.0, targeted=False,
                       backend_degraded=False)
        summary = self._degraded_run(runtime)
        self.assertEqual(summary["pass_rate"], 0.0)
        # Without the guard this would be a textbook regression.
        self.assertFalse(summary["regressed"])

    def test_the_summary_is_still_written(self):
        summary = self._degraded_run()
        self.assertTrue(os.path.exists(summary["summary_path"]))

    def test_a_healthy_run_is_unaffected(self):
        summary = eval_runner.run_evals(_runtime(), eval_dir=_tiny_suite())
        self.assertFalse(summary["backend_degraded"])
        self.assertEqual(summary["backend_failures"], 0)
        self.assertEqual(summary["backend_codes"], [])
        self.assertEqual(summary["pass_rate"], 1.0)


class BaselineSelectionTests(unittest.TestCase):
    def test_a_degraded_run_is_not_adopted_as_the_baseline(self):
        runtime = _runtime()
        _write_summary(runtime, "degraded", pass_rate=0.0, targeted=False,
                       backend_degraded=True)
        self.assertIsNone(eval_runner._previous_pass_rate(runtime))

    def test_the_last_healthy_run_is_used_instead(self):
        runtime = _runtime()
        _write_summary(runtime, "healthy", pass_rate=0.95, targeted=False,
                       backend_degraded=False)
        _write_summary(runtime, "degraded", pass_rate=0.0, targeted=False,
                       backend_degraded=True)
        self.assertEqual(eval_runner._previous_pass_rate(runtime), 0.95)


class MetadataResilienceTests(unittest.TestCase):
    """Resolving run metadata asks the backend what it has installed."""

    def test_a_dead_backend_does_not_discard_the_whole_suite(self):
        runtime = _runtime()

        def boom(*_args, **_kwargs):
            raise model_router.ModelUnavailableError(
                "Ollama is not reachable at http://127.0.0.1:9.",
                code=status_codes.OLLAMA_UNREACHABLE,
            )

        with mock.patch.object(model_router, "resolve_run_verifier", boom):
            summary = eval_runner.run_evals(runtime, eval_dir=_tiny_suite())

        # The cases themselves ran fine; only the metadata step failed.
        self.assertEqual(summary["total"], 4)
        self.assertTrue(os.path.exists(summary["summary_path"]))
        error = summary["metadata"]["metadata_error"]
        self.assertEqual(error["code"], status_codes.OLLAMA_UNREACHABLE)
        # Metadata we could not resolve must not masquerade as a real value.
        self.assertEqual(summary["metadata"]["verifier_backend"], "unknown")
        # And the run is still marked unreliable.
        self.assertTrue(summary["backend_degraded"])


class PromotionGateTests(unittest.TestCase):
    """A degraded eval run must not move a patch through the lifecycle."""

    def test_stable_promotion_is_refused_on_a_degraded_run(self):
        ok, reason = patch_manager.can_promote_to_stable(
            {"type": "prompt_patch"},
            {"must_pass_all_passed": True, "regressed": False,
             "backend_degraded": True},
        )
        self.assertFalse(ok)
        self.assertIn("outage", reason)

    def test_a_healthy_run_still_passes_the_gate(self):
        ok, _ = patch_manager.can_promote_to_stable(
            {"type": "prompt_patch"},
            {"must_pass_all_passed": True, "regressed": False,
             "backend_degraded": False},
        )
        self.assertTrue(ok)

    def test_a_summary_predating_the_flag_is_not_treated_as_degraded(self):
        ok, _ = patch_manager.can_promote_to_stable(
            {"type": "prompt_patch"},
            {"must_pass_all_passed": True, "regressed": False},
        )
        self.assertTrue(ok)


class PatchEvalBaselineTests(unittest.TestCase):
    """The comparison a patch is judged by must not rest on an outage."""

    def _patch(self) -> dict:
        return {
            "id": "patch_demo", "type": "prompt_patch", "channel": "experimental",
            "target": "role_pack:general:system_context",
            "change": {"append": "Lead with the direct answer."},
            "rationale": "demo", "created_at": "2026-01-01T00:00:00Z",
            "intent": "Make answers lead with the answer.",
            "rollback": "Drop the appended line.",
        }

    _HEALTHY_CANDIDATE = {
        "eval_run_id": "evalrun_candidate",
        "pass_rate": 1.0,
        "must_pass_all_passed": True,
        "backend_degraded": False,
        "categories_run": ["must_pass", "smoke"],
    }

    def _eval_against(self, runtime, candidate=None):
        """Run eval_patch's comparison logic without running the suite.

        `targeted=False` is essential: a targeted run discards the baseline
        unconditionally, so it cannot exercise baseline selection at all.
        """
        with mock.patch.object(patch_manager.eval_runner, "run_evals",
                               return_value=candidate or self._HEALTHY_CANDIDATE):
            return patch_manager.eval_patch(
                runtime.config, self._patch(), runtime, targeted=False,
            )

    def test_a_degraded_baseline_is_not_counted_as_an_improvement(self):
        runtime = _runtime()
        # The only prior run was an outage: pass_rate 0.0. A healthy
        # candidate would clear that bar on availability alone.
        _write_summary(runtime, "degraded", pass_rate=0.0, targeted=False,
                       backend_degraded=True, must_pass_all_passed=False)

        report = self._eval_against(runtime)
        self.assertIsNone(report["baseline_eval"])
        self.assertEqual(report["improvements"], [])

    def test_a_healthy_baseline_is_still_compared_against(self):
        # Guards the test above from passing for the wrong reason: the same
        # call path does find and use a baseline when one is trustworthy.
        runtime = _runtime()
        _write_summary(runtime, "healthy", pass_rate=0.5, targeted=False,
                       backend_degraded=False, must_pass_all_passed=True,
                       eval_run_id="evalrun_baseline")

        report = self._eval_against(runtime)
        self.assertIsNotNone(report["baseline_eval"])
        self.assertEqual(report["baseline_eval"]["pass_rate"], 0.5)
        self.assertTrue(report["improvements"])

    def test_a_degraded_candidate_is_rejected_outright(self):
        runtime = _runtime()
        report = self._eval_against(runtime, candidate={
            "eval_run_id": "evalrun_degraded",
            "pass_rate": 0.0,
            "must_pass_all_passed": False,
            "backend_degraded": True,
            "backend_codes": [status_codes.OLLAMA_UNREACHABLE],
            "categories_run": ["must_pass"],
        })
        self.assertEqual(report["eval_recommendation"], "reject")
        self.assertEqual(report["recommendation"], "reject")
        self.assertIn("outage", report["reason"])
        self.assertTrue(report["candidate_eval"]["backend_degraded"])


class EvalCLIExitCodeTests(unittest.TestCase):
    """An outage and a failed answer must stay distinguishable to a shell."""

    def setUp(self):
        from tests.helpers import write_temp_manifest
        self.manifest = write_temp_manifest(tempfile.mkdtemp(), tempfile.mkdtemp())

    def _eval(self, extra=()) -> tuple[int, str]:
        import contextlib
        import io
        from heimdal.cli import main
        buf = io.StringIO()
        argv = ["eval", "run", "--backend", "offline", "--manifest", self.manifest]
        with contextlib.redirect_stdout(buf):
            code = main(argv + list(extra))
        return code, buf.getvalue()

    def test_a_degraded_run_exits_2(self):
        with mock.patch("heimdal.models.offline.OfflineBackend.generate", _outage):
            code, out = self._eval()
        self.assertEqual(code, 2)
        self.assertIn("DEGRADED", out)
        self.assertIn(status_codes.OLLAMA_UNREACHABLE, out)

    def test_a_degraded_run_exits_2_in_json_mode_too(self):
        with mock.patch("heimdal.models.offline.OfflineBackend.generate", _outage):
            code, out = self._eval(["--json"])
        self.assertEqual(code, 2)
        self.assertTrue(json.loads(out)["backend_degraded"])

    def test_a_healthy_run_exits_0(self):
        code, out = self._eval()
        self.assertEqual(code, 0)
        self.assertNotIn("DEGRADED", out)

    def test_a_genuine_must_pass_failure_exits_1(self):
        # Not an outage: the suite ran and the answers were wrong.
        real = eval_runner.run_evals

        def failing(*args, **kwargs):
            summary = real(*args, **kwargs)
            summary["must_pass_all_passed"] = False
            summary["backend_degraded"] = False
            return summary

        with mock.patch.object(eval_runner, "run_evals", failing):
            code, out = self._eval()
        self.assertEqual(code, 1)
        self.assertNotIn("DEGRADED", out)


class DreamMustNotMineAnOutageTests(unittest.TestCase):
    """Self-improvement must not learn from a run that measured nothing."""

    def _degraded_eval(self):
        runtime = _runtime()
        with mock.patch("heimdal.models.offline.OfflineBackend.generate", _outage):
            summary = eval_runner.run_evals(runtime, eval_dir=_tiny_suite())
        self.assertTrue(summary["backend_degraded"])
        return runtime

    def test_a_degraded_eval_run_yields_no_failure_patterns(self):
        # Otherwise one outage becomes a pile of "quality failures" and Dream
        # proposes patches for defects that were never observed.
        from heimdal.dream import analysis
        runtime = self._degraded_eval()
        mining = analysis.gather_inputs(runtime.storage, source="eval", limit=50)
        patterns = analysis.detect_patterns(mining)
        self.assertEqual(patterns, [])

    def test_an_outage_trace_is_mined_as_infrastructure_not_quality(self):
        from heimdal.dream import analysis
        runtime = self._degraded_eval()
        mining = analysis.gather_inputs(runtime.storage, source="recent", limit=50)
        by_category = {p["category"]: p["count"]
                       for p in analysis.detect_patterns(mining)}
        self.assertIn("timeout_or_backend_error", by_category)
        self.assertNotIn("format_failure", by_category)
        self.assertNotIn("schema_failure", by_category)

    def test_every_backend_code_maps_to_the_infrastructure_pattern(self):
        # OLLAMA_TIMEOUT and OLLAMA_REQUEST_FAILED were newer than the map and
        # fell through to "unknown", which has no proposal builder.
        from heimdal.dream import analysis
        for code in sorted(status_codes.BACKEND_CODES):
            with self.subTest(code=code):
                mining = analysis.MiningInput(
                    bridge_failures=[{"job_id": "j", "code": code, "error": "x",
                                      "bridge": {"input_ref": "r"}}],
                )
                categories = {p["category"]
                              for p in analysis.detect_patterns(mining)}
                self.assertEqual(categories, {"timeout_or_backend_error"})

    def test_a_partially_degraded_case_is_classified_by_its_code(self):
        # Belt and braces for summaries written before backend_degraded, and
        # for a run where only some cases hit the backend.
        from heimdal.dream import analysis
        mining = analysis.MiningInput(eval_summaries=[{
            "eval_run_id": "evalrun_old",
            "results": [
                {"id": "c1", "category": "smoke", "passed": False,
                 "expected": "pass", "actual": "error",
                 "code": status_codes.OLLAMA_TIMEOUT},
                {"id": "c2", "category": "schema", "passed": False,
                 "expected": "pass", "actual": "fail", "code": None},
            ],
        }])
        by_category = {p["category"]: p["count"]
                       for p in analysis.detect_patterns(mining)}
        self.assertEqual(by_category.get("timeout_or_backend_error"), 1)
        self.assertEqual(by_category.get("schema_failure"), 1)


class SemanticVerifierOutageTests(unittest.TestCase):
    """Passing without being checked is not the same as passing."""

    @staticmethod
    def _semantic_only_outage():
        from heimdal.models.offline import OfflineBackend
        real = OfflineBackend.generate

        def patched(self, prompt, **kwargs):
            if (kwargs.get("structured") or {}).get("verify_task") == "semantic":
                raise OllamaError("verifier GPU gone",
                                  code=status_codes.OLLAMA_UNREACHABLE)
            return real(self, prompt, **kwargs)

        return mock.patch.object(OfflineBackend, "generate", patched)

    def _hybrid_runtime(self) -> Runtime:
        return Runtime(temp_config(tempfile.mkdtemp()), prefer_backend="offline",
                       verifier_override="hybrid")

    def test_the_verdict_is_unchanged_but_the_gap_is_reported(self):
        # The deterministic gate stays decisive, so the answer is not declared
        # wrong. What must not happen is the run looking fully verified.
        runtime = self._hybrid_runtime()
        with self._semantic_only_outage():
            result = runtime.run_envelope(_envelope("sv-1", "B2"))
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["metrics"]["semantic_verifier_unavailable"],
                         status_codes.OLLAMA_UNREACHABLE)

    def test_the_trace_records_that_nothing_was_semantically_verified(self):
        runtime = self._hybrid_runtime()
        with self._semantic_only_outage():
            result = runtime.run_envelope(_envelope("sv-2", "B2"))
        trace = Storage.read_json(result["trace_pack"]["path"])
        event = next(e for e in trace["events"] if e["name"] == "semantic_verify")
        self.assertEqual(event["data"]["semantic_verifier_unavailable"],
                         status_codes.OLLAMA_UNREACHABLE)

    def test_an_unverified_eval_run_cannot_become_evidence(self):
        runtime = self._hybrid_runtime()
        with self._semantic_only_outage():
            summary = eval_runner.run_evals(runtime, eval_dir=_tiny_suite("B2"))
        # Every case passed its deterministic gate...
        self.assertEqual(summary["pass_rate"], 1.0)
        # ...but nothing was semantically checked, so the run is not a baseline
        # and cannot promote a patch.
        self.assertTrue(summary["backend_degraded"])
        self.assertIn(status_codes.OLLAMA_UNREACHABLE, summary["backend_codes"])
        ok, reason = patch_manager.can_promote_to_stable(
            {"type": "prompt_patch"}, summary,
        )
        self.assertFalse(ok)
        self.assertIn("outage", reason)

    def test_a_healthy_hybrid_run_is_not_marked_degraded(self):
        runtime = self._hybrid_runtime()
        summary = eval_runner.run_evals(runtime, eval_dir=_tiny_suite("B2"))
        self.assertFalse(summary["backend_degraded"])
        self.assertEqual(summary["backend_codes"], [])

    def test_a_garbled_reply_is_still_treated_as_non_fatal_and_not_an_outage(self):
        # A malformed answer is a different thing from a dead backend; it must
        # not start marking runs degraded.
        from heimdal.core import verifier
        result = verifier._normalize_semantic("not a dict", "m")
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["confidence"], 0.0)
        self.assertNotIn("unavailable_code", result)
