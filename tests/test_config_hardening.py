"""Hardening: load_config and load_schema fail with clear messages."""

import os
import tempfile
import unittest

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
