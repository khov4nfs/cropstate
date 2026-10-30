"""Prevent mixing confidence ranking definitions in RIVF'26 risk--coverage figures."""
import math
import unittest

import numpy as np

from scripts.experiments.export_risk_coverage import (
    exact_aurc,
    exact_curve,
    probabilities,
    selective_point,
)


class RiskCoverageExporterTests(unittest.TestCase):
    def test_exact_full_prefix_curve_no_interpolated_points(self):
        scores = np.array([0.9, 0.7, 0.8])
        correct = np.array([True, False, True])
        curve = exact_curve(scores, correct)
        self.assertEqual([x["retained"] for x in curve], [1, 2, 3])
        self.assertEqual([x["risk"] for x in curve], [0, 0, 1 / 3])
        self.assertEqual([x["coverage"] for x in curve], [1 / 3, 2 / 3, 1.0])
        self.assertAlmostEqual(exact_aurc(curve), 1 / 9)

    def test_threshold_includes_ties_without_synthetic_partial_cases(self):
        scores = np.array([0.9, 0.8, 0.8, 0.1])
        correct = np.array([True, True, False, False])
        pt = selective_point(scores, correct, 0.8)
        self.assertEqual(pt["retained"], 3)
        self.assertAlmostEqual(pt["coverage"], 0.75)
        self.assertAlmostEqual(pt["risk"], 1 / 3)

    def test_temperature_scaling_stable_preserves_top1(self):
        logits = np.array([[900., 1., -500., -800., 3., 0.]])
        probs = probabilities(logits, 0.8746500341029899)
        self.assertTrue(np.isfinite(probs).all())
        self.assertEqual(probs.argmax(axis=1).tolist(), [0])
        self.assertAlmostEqual(probs.sum(), 1.0)

    def test_invalid_input_is_rejected(self):
        with self.assertRaises(ValueError):
            probabilities(np.zeros((2, 6)), 0)
        with self.assertRaises(ValueError):
            exact_curve(np.array([0.5, 0.8]), np.array([True]))
        with self.assertRaises(ValueError):
            exact_curve(np.array([math.nan]), np.array([True]))


if __name__ == "__main__":
    unittest.main()
