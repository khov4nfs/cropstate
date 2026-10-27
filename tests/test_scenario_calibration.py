"""Regression tests for the calibrated image-to-retrieval handoff.

These tests do not require image files, checkpoints, or a knowledge corpus.
"""
import unittest

import numpy as np

from cropstate.confidence import combined_confidence
from scripts.experiments.build_belief_scenarios import prediction_from_logits, softmax_np


class ScenarioBeliefTests(unittest.TestCase):
    def setUp(self):
        self.logits = np.array([0.5, 2.0, 1.0, -1.0, -2.0, -3.0])

    def test_stage_belief_matches_calibrated_distribution(self):
        temperature = 1.7
        belief, stage, confidence = prediction_from_logits(self.logits, temperature)
        expected = softmax_np((self.logits / temperature)[None, :])[0]
        np.testing.assert_allclose(belief, expected)
        self.assertEqual(stage, 1)
        self.assertAlmostEqual(confidence, combined_confidence(expected))

    def test_temperature_changes_belief_without_changing_argmax(self):
        cold, a, _ = prediction_from_logits(self.logits, 0.75)
        warm, b, _ = prediction_from_logits(self.logits, 2.0)
        self.assertEqual(a, b)
        self.assertGreater(cold.max(), warm.max())

    def test_top1_option_is_explicit_ablation(self):
        belief, _, confidence = prediction_from_logits(self.logits, 1.0, "top1")
        self.assertAlmostEqual(confidence, float(belief.max()))

    def test_rejects_invalid_temperature_and_method(self):
        for t in (0.0, -1.0, float("nan")):
            with self.subTest(temperature=t):
                with self.assertRaises(ValueError):
                    prediction_from_logits(self.logits, t)
        with self.assertRaises(ValueError):
            prediction_from_logits(self.logits, 1.0, "unknown")

    def test_rejects_invalid_logits(self):
        with self.assertRaises(ValueError):
            prediction_from_logits(np.array([1.0, 2.0]), 1.0)
        with self.assertRaises(ValueError):
            prediction_from_logits(np.array([0, 1, 2, 3, 4, np.nan]), 1.0)


if __name__ == "__main__":
    unittest.main()
