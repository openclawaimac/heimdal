"""v0.2.2: Truth Vault grounded retrieval — BM25 retrieval, the No-Guess Gate,
and the `heimdal truth` CLI."""

import os
import tempfile
import unittest

from tests.helpers import temp_config, write_temp_manifest

from heimdal.cli import main
from heimdal.core.runtime import Runtime
from heimdal.retrieval.truth_store import TruthStore
from heimdal.storage import Storage


def _write(directory: str, name: str, content: str) -> None:
    with open(os.path.join(directory, name), "w", encoding="utf-8") as fh:
        fh.write(content)


def _source_task(instruction: str) -> dict:
    """A source-required Host Task Envelope (research role, requires_sources)."""
    return {
        "host": {"type": "cli", "host_task_id": "truth-test", "source_agent": None, "callback": {}},
        "role_binding": {
            "role_id": "research",
            "risk_mode": "conservative",
            "privacy_mode": "local_only",
            "output_profiles": ["markdown"],
        },
        "task_request": {
            "task_id": "truth-test",
            "title": "Sourced task",
            "instruction": instruction,
            "inputs": {},
            "constraints": {"requires_sources": True},
            "priority": "P1",
            "budget": {"quality_level": "B2"},
            "expected_outputs": ["markdown_response_with_sources"],
        },
        "runtime_hints": {},
    }


class TruthRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.truth_dir = tempfile.mkdtemp()

    def test_empty_vault_returns_nothing(self):
        self.assertEqual(TruthStore(self.truth_dir).retrieve("anything at all"), [])

    def test_relevant_document_is_retrieved(self):
        _write(
            self.truth_dir,
            "refund_policy.md",
            "Product Zeta refund policy. Customers may return Product Zeta "
            "within thirty days for a full refund of the purchase price.",
        )
        hits = TruthStore(self.truth_dir).retrieve("State the refund policy for Product Zeta.")
        self.assertEqual([h.ref for h in hits], ["refund_policy.md"])
        self.assertGreater(hits[0].score, 0)

    def test_irrelevant_document_is_not_retrieved(self):
        _write(
            self.truth_dir,
            "cooking.md",
            "To cook pasta, bring a large pot of water to a rolling boil, "
            "add salt, then add the pasta and stir occasionally.",
        )
        hits = TruthStore(self.truth_dir).retrieve("State the refund policy for Product Zeta.")
        self.assertEqual(hits, [])

    def test_bm25_ranks_the_stronger_match_first(self):
        _write(
            self.truth_dir,
            "pricing.md",
            "Widget pricing discount. Widget pricing discount. Widget pricing discount.",
        )
        _write(
            self.truth_dir,
            "filler.md",
            "We sell a widget and our pricing includes a discount. "
            + "lorem ipsum dolor sit amet consectetur " * 12,
        )
        hits = TruthStore(self.truth_dir).retrieve("widget pricing discount")
        self.assertEqual(hits[0].ref, "pricing.md")
        self.assertGreater(hits[0].score, hits[1].score)

    def test_list_sources(self):
        _write(self.truth_dir, "a.md", "alpha content here")
        _write(self.truth_dir, "b.txt", "beta content here")
        refs = {s["ref"] for s in TruthStore(self.truth_dir).list_sources()}
        self.assertEqual(refs, {"a.md", "b.txt"})


class TruthGroundedRuntimeTests(unittest.TestCase):
    """No-Guess Gate: missing/irrelevant -> need_input; relevant -> grounded pass."""

    def _runtime(self, truth_files: dict[str, str]) -> Runtime:
        root = tempfile.mkdtemp()
        truth_dir = os.path.join(root, "truth")
        os.makedirs(truth_dir)
        for name, content in truth_files.items():
            _write(truth_dir, name, content)
        return Runtime(temp_config(root), prefer_backend="offline")

    def test_relevant_source_grounds_a_passing_answer(self):
        runtime = self._runtime(
            {
                "refund_policy.md": "Product Zeta refund policy. Customers may "
                "return Product Zeta within thirty days for a full refund."
            }
        )
        result = runtime.run_envelope(
            _source_task("State the refund policy for Product Zeta.")
        )
        self.assertEqual(result["status"], "pass")
        repro = Storage.read_json(result["repro_pack"]["path"])
        self.assertIn("refund_policy.md", repro["retrieval_refs"])

    def test_missing_source_returns_need_input(self):
        # `.keep` makes the vault non-empty so seeding is skipped, but it is not
        # a .md/.txt file, so retrieval sees an effectively empty vault.
        runtime = self._runtime({".keep": ""})
        result = runtime.run_envelope(
            _source_task("State the refund policy for Product Zeta.")
        )
        self.assertEqual(result["status"], "need_input")

    def test_irrelevant_source_still_returns_need_input(self):
        runtime = self._runtime(
            {"cooking.md": "To cook pasta, boil water, add salt and stir."}
        )
        result = runtime.run_envelope(
            _source_task("State the refund policy for Product Zeta.")
        )
        self.assertEqual(result["status"], "need_input")


class TruthCLITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.manifest = write_temp_manifest(self.tmp, self.tmp)

    def test_truth_list_add_search(self):
        # Empty vault lists cleanly.
        self.assertEqual(main(["truth", "list", "--manifest", self.manifest]), 0)

        # Add a markdown source.
        src = os.path.join(self.tmp, "vendor_policy.md")
        _write(
            os.path.dirname(src),
            "vendor_policy.md",
            "Vendor Kappa shipping policy: orders ship within two business days.",
        )
        self.assertEqual(main(["truth", "add", src, "--manifest", self.manifest]), 0)
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "truth", "vendor_policy.md")))

        # Search finds it.
        self.assertEqual(
            main(["truth", "search", "Kappa shipping policy", "--manifest", self.manifest]), 0
        )

    def test_truth_add_rejects_non_text_file(self):
        bad = os.path.join(self.tmp, "data.json")
        _write(self.tmp, "data.json", "{}")
        self.assertEqual(main(["truth", "add", bad, "--manifest", self.manifest]), 2)

    def test_truth_add_missing_file(self):
        self.assertEqual(
            main(["truth", "add", "/no/such/file.md", "--manifest", self.manifest]), 2
        )


if __name__ == "__main__":
    unittest.main()
