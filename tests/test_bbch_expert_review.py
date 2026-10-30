"""Unit tests use synthetic rows; they are NOT expert annotation results."""
import csv
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_bbch_expert_review import prepare_review, select_one_per_parent
from scripts.score_bbch_expert_review import (
    annotation_stage, cohens_kappa, score_review, stage_for_bbch, read_unique_csv,
)


class BBCHPacketTests(unittest.TestCase):
    def setUp(self):
        self.rows = []
        for i, stage in enumerate((
            "establishment", "tillering", "stem_booting", "reproductive",
            "grain_filling", "ripening",
        )):
            for j in range(4):
                for patch in range(2):
                    self.rows.append({
                        "image_path": f"scene_{i}_{j}_patch{patch}.jpg",
                        "macro_stage": stage,
                        "parent_image_id": f"scene_{i}_{j}",
                        "review_flag": str(j % 2),
                        "split": "test",
                    })

    def test_stratified_unique_parents_and_seed(self):
        first = select_one_per_parent(self.rows, per_stage=3, seed=42)
        second = select_one_per_parent(self.rows, per_stage=3, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 18)
        self.assertEqual(len({r["parent_image_id"] for r in first}), 18)
        for stage in ("establishment", "tillering", "stem_booting",
                      "reproductive", "grain_filling", "ripening"):
            self.assertEqual(sum(r["macro_stage"] == stage for r in first), 3)

    def test_reviewer_sheet_blinded_from_weak_label(self):
        with tempfile.TemporaryDirectory() as work:
            out = Path(work)
            report = prepare_review(self.rows, out, per_stage=2, seed=42)
            self.assertEqual(report["total"], 12)
            admin = read_unique_csv(out / "admin_manifest.csv")
            review1 = read_unique_csv(out / "reviewer_1.csv")
            self.assertEqual(set(admin), set(review1))
            self.assertEqual({v["reviewer_stage"] for v in review1.values()}, {""})
            self.assertFalse({"weak_stage", "review_flag", "source_image_path"} &
                             set(next(iter(review1.values()))))
            self.assertEqual({v["image_filename"] for v in review1.values()},
                             {v["image_filename"] for v in admin.values()})

    def test_missing_assets_fail_closed(self):
        with tempfile.TemporaryDirectory() as work:
            with self.assertRaises(FileNotFoundError):
                prepare_review(self.rows, Path(work) / "out", per_stage=1,
                               data_root=Path(work) / "missing")


class BBCHScoringTests(unittest.TestCase):
    def test_bbch_macro_stage_boundaries(self):
        for code, expected in (
            ("00", "establishment"), ("19", "establishment"),
            ("20", "tillering"), ("29", "tillering"),
            ("30", "stem_booting"), ("49", "stem_booting"),
            ("50", "reproductive"), ("69", "reproductive"),
            ("70", "grain_filling"), ("79", "grain_filling"),
            ("80", "ripening"), ("89", "ripening"),
        ):
            self.assertEqual(stage_for_bbch(code), expected)
        with self.assertRaises(ValueError):
            stage_for_bbch("BBCH 92")

    def test_conflicting_bbch_annotation_rejected(self):
        with self.assertRaises(ValueError):
            annotation_stage({
                "reviewer_stage": "tillering",
                "reviewer_bbch": "70",
                "uncertain": "",
            })

    def test_unfilled_review_creates_no_expert_consensus(self):
        admin = {"id": {"source_image_path": "x.jpg", "parent_image_id": "parent",
                        "weak_stage": "tillering", "image_filename": "BBCH00001.jpg"}}
        blank = {"id": {"reviewer_stage": "", "reviewer_bbch": "", "uncertain": ""}}
        stats, confirmed, adjudicate = score_review(admin, blank, blank)
        self.assertEqual(stats["expert_consensus_count"], 0)
        self.assertEqual(stats["paired_completed_count"], 0)
        self.assertIsNone(stats["weak_label_agreement_on_consensus_subset"])
        self.assertEqual(len(adjudicate), 1)
        self.assertFalse(confirmed)

    def test_independent_consensus_and_disagreement(self):
        admin = {
            "a": {"source_image_path": "a.jpg", "parent_image_id": "a", "weak_stage": "tillering", "image_filename": "a.jpg"},
            "b": {"source_image_path": "b.jpg", "parent_image_id": "b", "weak_stage": "ripening", "image_filename": "b.jpg"},
        }
        r1 = {"a": {"reviewer_stage": "tillering"}, "b": {"reviewer_stage": "ripening"}}
        r2 = {"a": {"reviewer_bbch": "24"}, "b": {"reviewer_bbch": "77"}}
        stats, consensus, adjudicate = score_review(admin, r1, r2)
        self.assertEqual(stats["paired_completed_count"], 2)
        self.assertEqual(stats["expert_consensus_count"], 1)
        self.assertEqual(len(adjudicate), 1)
        self.assertEqual(consensus[0]["review_provenance"],
                         "two_independent_reviewers_agree")

    def test_kappa_uses_only_paired_labels(self):
        self.assertIsNone(cohens_kappa([], []))
        self.assertAlmostEqual(cohens_kappa(["tillering", "ripening"],
                                            ["tillering", "ripening"]), 1.0)


if __name__ == "__main__":
    unittest.main()
