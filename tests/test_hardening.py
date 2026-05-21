"""Post-validation hardening: model selection, router fallback, Ollama errors,
CLI overrides, and the optional semantic verifier."""

import io
import json
import tempfile
import unittest
import urllib.error

from tests.helpers import temp_config

from heimdal.cli import build_parser
from heimdal.core import model_router, verifier
from heimdal.core.runtime import Runtime
from heimdal.models.base import GenerationResult, ModelBackend, select_generative_model
from heimdal.models.offline import OfflineBackend
from heimdal.models.ollama import OllamaError, _describe_error


class _StubBackend(ModelBackend):
    """A backend that returns a fixed generation, used to drive the verifier."""

    name = "ollama"

    def __init__(self, text: str, models: list[str] | None = None):
        self._text = text
        self._models = models or ["qwen2.5:7b"]

    def is_available(self) -> bool:
        return True

    def list_models(self) -> list[str]:
        return self._models

    def generate(self, prompt, *, model, system="", json_mode=False,
                 max_tokens=512, temperature=0.2, structured=None):
        return GenerationResult(text=self._text, model=model, backend=self.name)


class ModelSelectionTests(unittest.TestCase):
    def setUp(self):
        self.config = temp_config(tempfile.mkdtemp())

    def test_skips_embedding_models(self):
        installed = ["nomic-embed-text:latest", "qwen2.5:7b"]
        self.assertEqual(select_generative_model(self.config, installed), "qwen2.5:7b")

    def test_returns_none_when_only_embedding_models(self):
        self.assertIsNone(
            select_generative_model(self.config, ["nomic-embed-text:latest"])
        )

    def test_prefers_manifest_candidate(self):
        installed = ["llama3.2:3b", "qwen2.5:7b"]  # qwen2.5:7b is a worker candidate
        self.assertEqual(select_generative_model(self.config, installed), "qwen2.5:7b")


class RouterFallbackTests(unittest.TestCase):
    def setUp(self):
        self.config = temp_config(tempfile.mkdtemp())

    def test_fallback_to_installed_generative_model(self):
        # No manifest candidate installed, but a usable generative model is.
        model = model_router._resolve_model("worker", {"llama3.2:3b"}, self.config)
        self.assertEqual(model, "llama3.2:3b")

    def test_raises_actionable_error_when_no_generative_model(self):
        with self.assertRaises(model_router.ModelUnavailableError) as ctx:
            model_router._resolve_model("worker", {"nomic-embed-text:latest"}, self.config)
        self.assertIn("ollama pull", str(ctx.exception))


class OllamaErrorTests(unittest.TestCase):
    def test_missing_model_is_actionable(self):
        exc = urllib.error.HTTPError(
            "http://x/api/generate", 404, "Not Found", {}, io.BytesIO(b"not found")
        )
        message = _describe_error(exc, "http://x", "qwen2.5:7b", 120)
        self.assertIn("ollama pull qwen2.5:7b", message)

    def test_unreachable_is_described(self):
        exc = urllib.error.URLError("Connection refused")
        message = _describe_error(exc, "http://x", "m", 120)
        self.assertIn("not reachable", message)

    def test_timeout_is_described(self):
        message = _describe_error(TimeoutError(), "http://x", "m", 30)
        self.assertIn("timed out", message)

    def test_ollama_error_is_runtime_error(self):
        self.assertTrue(issubclass(OllamaError, RuntimeError))


class OfflineWordLimitTests(unittest.TestCase):
    """Offline responses must respect constraints.max_words, even small limits."""

    def test_small_word_limits_are_enforced(self):
        backend = OfflineBackend()
        long_truth = "This is a long grounding sentence. " * 30
        for limit in (1, 3, 5, 8, 12, 40, 120):
            result = backend.generate(
                "",
                structured={
                    "title": "Demo task",
                    "instruction": "Explain the system.",
                    "truth": [long_truth],
                    "max_words": limit,
                },
            )
            self.assertLessEqual(
                len(result.text.split()), limit, f"exceeded max_words={limit}"
            )

    def test_no_limit_leaves_text_untrimmed(self):
        backend = OfflineBackend()
        result = backend.generate(
            "", structured={"title": "T", "instruction": "Explain.", "truth": []}
        )
        self.assertTrue(result.text.strip())


class CLIOverrideTests(unittest.TestCase):
    def test_run_accepts_model_and_backend(self):
        args = build_parser().parse_args(
            ["run", "demo", "--model", "qwen2.5:7b", "--backend", "offline"]
        )
        self.assertEqual(args.model, "qwen2.5:7b")
        self.assertEqual(args.backend, "offline")

    def test_doctor_accepts_model(self):
        args = build_parser().parse_args(["doctor", "--model", "qwen2.5:7b"])
        self.assertEqual(args.model, "qwen2.5:7b")

    def test_model_override_flows_into_routing(self):
        config = temp_config(tempfile.mkdtemp())
        runtime = Runtime(config, prefer_backend="offline", model_override="custom-model:1b")
        result = runtime.run_demo()
        self.assertEqual(result["metrics"]["worker_model"], "custom-model:1b")


class SemanticVerifierTests(unittest.TestCase):
    """Item 9/10: schema-valid JSON that semantically fails the task."""

    def setUp(self):
        self.config = temp_config(tempfile.mkdtemp())
        self.contract = {
            "objective": "Return a JSON object with a short answer field.",
            "constraints": {},
            "verification": {},
        }
        self.packet = {"truth_context": []}
        self.routing = {
            "verifier_strictness": "standard",
            "verifier_backend": "hybrid",
            "verifier_model": "qwen2.5:7b",
        }

    def test_semantic_failure_produces_defect(self):
        # Valid JSON, but the "answer" dodges the task by asking a question back.
        bad = json.dumps({"answer": "What question would you like me to answer?"})
        judge = _StubBackend(json.dumps({"satisfies": False, "reason": "asks a question back"}))
        result = verifier.verify(bad, self.contract, self.packet, self.routing, self.config, judge)
        self.assertEqual(result["status"], "fail")
        self.assertTrue(any("Semantic verifier" in d["message"] for d in result["defects"]))

    def test_semantic_pass_leaves_no_semantic_defect(self):
        good = json.dumps({"answer": "A queue is a first-in first-out collection."})
        judge = _StubBackend(json.dumps({"satisfies": True, "reason": "ok"}))
        result = verifier.verify(good, self.contract, self.packet, self.routing, self.config, judge)
        self.assertFalse(any("Semantic verifier" in d["message"] for d in result["defects"]))

    def test_router_marks_hybrid_only_when_enabled_and_budget_qualifies(self):
        self.config.manifest["verifier"]["semantic_enabled"] = True
        backend = _StubBackend("{}")
        b2 = {"budget": {"quality_level": "B2", "max_iterations": 3}, "verification": {}}
        b1 = {"budget": {"quality_level": "B1", "max_iterations": 3}, "verification": {}}
        role = {"role_id": "general"}
        self.assertEqual(
            model_router.route(b2, role, backend, self.config)["verifier_backend"], "hybrid"
        )
        self.assertEqual(
            model_router.route(b1, role, backend, self.config)["verifier_backend"], "rule_based"
        )


if __name__ == "__main__":
    unittest.main()
