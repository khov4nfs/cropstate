"""Synthetic scenarios test only provenance checks, NEVER reported benchmark metrics."""
import csv
import tempfile
import unittest
from pathlib import Path

from scripts.reproduce_retrieval_benchmark import assert_scenario_provenance


class ReproductionPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.file = Path(self.temp.name) / "synthetic_scenarios.csv"

    def tearDown(self):
        self.temp.cleanup()

    def make_rows(self, *, legacy=False):
        fields = [
            "scenario_id", "stage_belief", "confidence", "ground_truth_stage",
            "predicted_stage", "relevant_chunk_ids", "incompatible_chunk_ids",
            "confidence_method", "calibration_temperature",
        ]
        rows = []
        for i in range(199):
            for topic in (["water", "nutrient", "pest"] if i < 22 else ["water", "nutrient"]):
                rows.append({
                    "scenario_id": f"image{i}__{topic}",
                    "stage_belief": "[0.1, 0.7, 0.1, 0.05, 0.04, 0.01]",
                    "confidence": "0.72",
                    "ground_truth_stage": "tillering",
                    "predicted_stage": "tillering",
                    "relevant_chunk_ids": "chunk1",
                    "incompatible_chunk_ids": "chunk2",
                    "confidence_method": "top1" if legacy else "combined",
                    "calibration_temperature": "" if legacy else "0.875",
                })
        with self.file.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerows(rows)
        return rows

    def test_exact_420_scenarios_from_199_source_images(self):
        self.make_rows()
        checked = assert_scenario_provenance(self.file, 420, 199)
        self.assertEqual(checked, {"scenario_count": 420, "contributing_images": 199})

    def test_legacy_probs_require_explicit_ablation(self):
        self.make_rows(legacy=True)
        with self.assertRaisesRegex(ValueError, "calibrated belief"):
            assert_scenario_provenance(self.file, 420, 199)
        self.assertEqual(assert_scenario_provenance(
            self.file, 420, 199, allow_legacy=True
        )["scenario_count"], 420)

    def test_missing_scenarios_are_not_validated(self):
        self.make_rows()
        with self.assertRaisesRegex(ValueError, "Expected 421"):
            assert_scenario_provenance(self.file, 421, 199)

    def test_repeated_scenario_ids_are_rejected(self):
        rows = self.make_rows()
        rows[-1]["scenario_id"] = rows[0]["scenario_id"]
        with self.file.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            assert_scenario_provenance(self.file, 420, 199)


if __name__ == "__main__":
    unittest.main()
