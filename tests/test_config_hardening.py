"""Hardening: load_config and load_schema fail with clear messages."""

import os
import tempfile
import unittest

from tests.helpers import temp_config

from heimdal import jsonschema_min
from heimdal.config import ConfigError, load_config


class LoadConfigHardeningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_missing_manifest_raises_config_error(self):
        missing = os.path.join(self.tmp, "nope.yml")
        with self.assertRaises(ConfigError) as ctx:
            load_config(missing)
        self.assertIn("not found", str(ctx.exception))

    def test_malformed_yaml_raises_config_error(self):
        bad = os.path.join(self.tmp, "bad.yml")
        with open(bad, "w", encoding="utf-8") as fh:
            fh.write("runtime: [unclosed\n  : : :")
        with self.assertRaises(ConfigError) as ctx:
            load_config(bad)
        self.assertIn("not valid YAML", str(ctx.exception))

    def test_non_mapping_manifest_raises_config_error(self):
        listy = os.path.join(self.tmp, "list.yml")
        with open(listy, "w", encoding="utf-8") as fh:
            fh.write("- a\n- b\n")
        with self.assertRaises(ConfigError) as ctx:
            load_config(listy)
        self.assertIn("mapping", str(ctx.exception))


class LoadSchemaHardeningTests(unittest.TestCase):
    def test_missing_schema_raises_clear_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            jsonschema_min.load_schema("/no/such/schema.json")
        self.assertIn("Schema file not found", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()


class EmptyManifestBlockTests(unittest.TestCase):
    """A section written with no body parses to None, not {}.

    Commenting out everything under `ollama:` is an ordinary edit; it used to
    surface as an AttributeError deep inside whichever consumer touched the
    section first (profiler, endpoint pool, router).
    """

    _BLOCKS = ("runtime", "ollama", "model_profiles", "model_roles",
               "runtime_profiles", "scheduler", "budgets", "verifier",
               "retrieval", "mirror")

    def _config_with_empty(self, block):
        config = temp_config(tempfile.mkdtemp())
        config.manifest[block] = None
        return config

    def test_every_block_accessor_returns_a_mapping(self):
        for block in self._BLOCKS:
            with self.subTest(block=block):
                config = self._config_with_empty(block)
                self.assertEqual(getattr(config, block), {})

    def test_a_non_mapping_block_is_also_tolerated(self):
        config = temp_config(tempfile.mkdtemp())
        config.manifest["ollama"] = "http://localhost:11434"  # scalar by mistake
        self.assertEqual(config.ollama, {})

    def test_the_runtime_starts_with_an_empty_ollama_block(self):
        from heimdal.core.runtime import Runtime
        runtime = Runtime(self._config_with_empty("ollama"), prefer_backend="offline")
        self.assertEqual(runtime.endpoint_pool.failover_mode, "auto")

    def test_an_empty_sandbox_block_still_resolves_a_policy(self):
        config = self._config_with_empty("runtime")
        config.manifest["sandbox"] = None
        self.assertIsInstance(config.sandbox_policy(), dict)
