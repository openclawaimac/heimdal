"""v0.5.0 hardening: redaction edge cases, manual + cloud teacher providers."""

import os
import tempfile
import unittest

from heimdal.mirror import redaction
from heimdal.mirror.cloud_teacher import (
    AnthropicTeacher,
    CloudProviderUnavailable,
    OpenAITeacher,
)
from heimdal.mirror.manual_teacher import ManualTeacher
from heimdal.mirror.provider import TeacherInput


def _input(case_id: str = "case-1", local: str = "local answer") -> TeacherInput:
    return TeacherInput(case_id=case_id, task={"objective": "obj"}, local_output=local)


class RedactionEdgeCaseTests(unittest.TestCase):
    def test_multiple_secrets_in_one_string(self):
        text = (
            "OPENAI_API_KEY=sk-proj-abcdefghijklmnopqrstuvwx0123 and "
            "PASSWORD=hunter2supersecret in one line"
        )
        result = redaction.redact(text)
        self.assertNotIn("sk-proj-abcdefghijklmnopqrstuvwx", result.text)
        self.assertNotIn("hunter2supersecret", result.text)
        kinds = {r["kind"] for r in result.redactions}
        # Both the key and the password assignment are caught.
        self.assertTrue(
            {"openai_api_key", "env_secret_assignment"} & kinds,
            f"expected secret kinds, got {kinds}",
        )
        self.assertGreaterEqual(len(result.redactions), 2)

    def test_aws_access_key(self):
        result = redaction.redact("aws key AKIAIOSFODNN7EXAMPLE here")
        self.assertNotIn("AKIAIOSFODNN7EXAMPLE", result.text)
        self.assertTrue(any(r["kind"] == "aws_access_key" for r in result.redactions))

    def test_bearer_header(self):
        result = redaction.redact("Authorization: Bearer abcDEF1234567890tokenvalue")
        self.assertIn("[REDACTED:bearer_header]", result.text)

    def test_clean_text_is_unchanged(self):
        clean = "A queue is a first-in, first-out data structure."
        result = redaction.redact(clean)
        self.assertEqual(result.text, clean)
        self.assertEqual(result.redactions, [])


class ManualTeacherTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def test_returns_pass_when_md_file_present(self):
        with open(os.path.join(self.dir, "case-1.md"), "w", encoding="utf-8") as fh:
            fh.write("# Teacher answer\nGrounded and structured.")
        result = ManualTeacher(self.dir).generate(_input("case-1"))
        self.assertEqual(result.status, "pass")
        self.assertIn("Teacher answer", result.output)

    def test_returns_pass_when_txt_file_present(self):
        with open(os.path.join(self.dir, "case-2.txt"), "w", encoding="utf-8") as fh:
            fh.write("plain text teacher answer")
        result = ManualTeacher(self.dir).generate(_input("case-2"))
        self.assertEqual(result.status, "pass")

    def test_returns_skipped_when_no_file(self):
        result = ManualTeacher(self.dir).generate(_input("missing-case"))
        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.output, "")


class CloudTeacherUnavailableTests(unittest.TestCase):
    """Cloud providers must raise the typed CloudProviderUnavailable (never a
    bare ImportError/KeyError) when the SDK or API key is missing. No real
    SDK is imported here -- the missing key check fires first."""

    def setUp(self):
        # Ensure no credential is present so the env check trips.
        self._saved = {}
        for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            self._saved[var] = os.environ.pop(var, None)

    def tearDown(self):
        for var, value in self._saved.items():
            if value is not None:
                os.environ[var] = value

    def test_openai_without_key_raises_typed_error(self):
        with self.assertRaises(CloudProviderUnavailable):
            OpenAITeacher(model="gpt-4o-mini").generate(_input())

    def test_anthropic_without_key_raises_typed_error(self):
        with self.assertRaises(CloudProviderUnavailable):
            AnthropicTeacher(model="claude-3-5-sonnet-latest").generate(_input())


if __name__ == "__main__":
    unittest.main()
