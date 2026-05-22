"""v0.2.6: heimdal verify -- the verifier-only path.

A host (e.g. Hermes) drafts its own candidate answer and asks Heimdal's
verifier to judge it, without Heimdal drafting the answer itself.
"""

import contextlib
import io
import json
import os
import tempfile
import unittest

from tests.helpers import repo_path, temp_config, write_temp_manifest

from heimdal.cli import main
from heimdal.core.runtime import Runtime
from heimdal.storage import Storage

_QUEUE_TASK = "examples/tasks/queue_task.json"

_GOOD_ANSWER = (
    "A queue is a first-in, first-out (FIFO) data structure. Items are added "
    "at the back and removed from the front, so the earliest item added is "
    "served next. Queues are used for scheduling and buffering work."
)
# Structurally fine, but it dodges the task instead of answering it.
_DODGE_ANSWER = "What kind of queue do you mean?"


def _runtime(verifier=None) -> Runtime:
    return Runtime(
        temp_config(tempfile.mkdtemp()),
        prefer_backend="offline",
        verifier_override=verifier,
    )


class VerifyEnvelopeTests(unittest.TestCase):
    def _envelope(self) -> dict:
        return Storage.read_json(repo_path(_QUEUE_TASK))

    def test_good_answer_passes(self):
        result = _runtime("hybrid").verify_envelope(self._envelope(), _GOOD_ANSWER)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["code"], "OK")
        # Verification run refs are host-safe relative refs, not absolute paths.
        self.assertFalse(os.path.isabs(result["repro_pack_ref"]))
        self.assertFalse(os.path.isabs(result["trace_pack_ref"]))

    def test_semantically_bad_answer_fails_in_hybrid_mode(self):
        # The dodge answer is structurally valid but does not fulfil the task;
        # only the hybrid semantic verifier catches it.
        result = _runtime("hybrid").verify_envelope(self._envelope(), _DODGE_ANSWER)
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["code"], "VERIFIER_SEMANTIC_FAIL")

    def test_rule_based_mode_runs_without_semantic_verifier(self):
        result = _runtime("rule_based").verify_envelope(self._envelope(), _GOOD_ANSWER)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["verifier"]["backend"], "rule_based")
        self.assertIsNone(result["verifier"]["semantic_model"])

    def test_empty_answer_fails_rule_based(self):
        result = _runtime("rule_based").verify_envelope(self._envelope(), "")
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["code"], "VERIFIER_RULE_FAIL")


class VerifyCLITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.manifest = write_temp_manifest(self.tmp, self.tmp)

    def _answer_file(self, text: str) -> str:
        path = os.path.join(self.tmp, "answer.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"answer": text}, fh)
        return path

    def _run(self, answer_text: str):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(
                [
                    "verify",
                    "--task", repo_path(_QUEUE_TASK),
                    "--answer", self._answer_file(answer_text),
                    "--offline", "--verifier", "hybrid", "--json",
                    "--manifest", self.manifest,
                ]
            )
        return code, json.loads(buf.getvalue())

    def test_verify_passes_good_answer(self):
        code, result = self._run(_GOOD_ANSWER)
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "pass")

    def test_verify_fails_semantically_bad_answer(self):
        code, result = self._run(_DODGE_ANSWER)
        self.assertEqual(code, 1)
        self.assertEqual(result["code"], "VERIFIER_SEMANTIC_FAIL")

    def test_verify_requires_task_and_answer(self):
        with self.assertRaises(SystemExit):
            main(["verify", "--manifest", self.manifest])


if __name__ == "__main__":
    unittest.main()
