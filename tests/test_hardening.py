"""Post-validation hardening: model selection, router fallback, Ollama errors,
and CLI overrides."""

import io
import tempfile
import unittest
import urllib.error

from tests.helpers import temp_config

from heimdal.cli import build_parser
from heimdal.core import model_router
from heimdal.core.runtime import Runtime
from heimdal.models.base import select_generative_model
from heimdal.models.offline import OfflineBackend
from heimdal.models.ollama import OllamaError, _describe_error


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


if __name__ == "__main__":
    unittest.main()
