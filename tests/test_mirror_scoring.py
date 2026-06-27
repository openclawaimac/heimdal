"""v0.5.1 hardening: direct tests for the per-dimension scoring heuristics.

The diff engine tests exercise scoring indirectly; these pin the individual
dimension behaviors so a heuristic change can't silently drift.
"""

import unittest

from heimdal.mirror import scoring


def _score(text, **task):
    return scoring.score_output(text, task)


class ScoreOutputTests(unittest.TestCase):
    def test_all_dimensions_present_and_in_range(self):
        scores = _score("A queue is FIFO.", objective="Explain a queue.")
        self.assertEqual(set(scores), set(scoring.DIMENSIONS))
        for dim, value in scores.items():
            self.assertGreaterEqual(value, 0.0, dim)
            self.assertLessEqual(value, 1.0, dim)

    def test_empty_answer_scores_zero_semantic_quality(self):
        self.assertEqual(_score("", objective="Explain a queue.")["semantic_quality"], 0.0)

    def test_unsourced_specifics_raise_hallucination_risk(self):
        # hallucination_risk is inverted: lower score = higher risk. A number
        # + date with no source ref must score worse than a clean answer.
        risky = _score("The price is $249.99 effective 2024-03-15.",
                       objective="State the price.")
        clean = _score("The price depends on the documented tier.",
                       objective="State the price.")
        self.assertLess(risky["hallucination_risk"], clean["hallucination_risk"])

    def test_source_ref_offsets_hallucination_risk(self):
        with_ref = _score("Per (pricing.md), the price is $249.99.",
                          objective="State the price.")
        without = _score("The price is $249.99.", objective="State the price.")
        self.assertGreater(with_ref["hallucination_risk"], without["hallucination_risk"])

    def test_no_guess_behavior_rewards_hedge_on_source_required(self):
        hedge = scoring.score_output(
            "I cannot find a source for this; please provide one.",
            {"objective": "State the refund policy.",
             "constraints": {"requires_sources": True}},
        )
        guess = scoring.score_output(
            "The refund policy is 30 days.",
            {"objective": "State the refund policy.",
             "constraints": {"requires_sources": True}},
        )
        self.assertGreater(hedge["no_guess_behavior"], guess["no_guess_behavior"])

    def test_conciseness_penalizes_exceeding_max_words(self):
        long_text = " ".join(["word"] * 200)
        scores = scoring.score_output(
            long_text, {"objective": "Be brief.", "constraints": {"max_words": 20}},
        )
        self.assertLess(scores["conciseness"], 1.0)

    def test_structure_format_rewards_headers_and_bullets(self):
        structured = _score("# Title\n\n## Section\n- point one\n- point two",
                            objective="Explain.")
        flat = _score("just a flat sentence with no structure at all",
                      objective="Explain.")
        self.assertGreater(structured["structure_format"], flat["structure_format"])

    def test_deterministic(self):
        a = _score("# Q\n- FIFO\n- LIFO", objective="Compare queue and stack.")
        b = _score("# Q\n- FIFO\n- LIFO", objective="Compare queue and stack.")
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
