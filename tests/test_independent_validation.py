"""Regression tests for validation-only selection and independent expert judgments."""
from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.experiments.run_retrieval_ablations import (
    choose_abstention_threshold, choose_fixed_beta, validate_partitions,
    variant_ranking,
)
from scripts.prepare_retrieval_expert_review import (
    REVIEW_COLS, context_hash, grade_case, judgment, read_csv, score, select_cases, write_csv,
)
from cropstate.knowledge import KnowledgeChunk


class AblationIntegrityTests(unittest.TestCase):
    def test_no_image_overlap_and_calibrated_only(self):
        def row(image):
            return {"scenario_id": image + "__pest", "calibration_applied": "true",
                    "confidence_method": "combined", "raw_stage_belief": "[1,0,0,0,0,0]"}
        validate_partitions([row("a")], [row("b")])
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_partitions([row("a")], [row("a")])
        with self.assertRaisesRegex(ValueError, "calibrated"):
            validate_partitions([row("a")], [{**row("b"), "confidence_method": "top1"}])

    def test_select_beta_from_validation_only(self):
        metrics = {0.05: {"ndcg_at_k": 0.8, "sirr_at_k": 0.3},
                   0.20: {"ndcg_at_k": 0.7, "sirr_at_k": 0.1}}
        self.assertEqual(choose_fixed_beta(metrics), 0.20)
        self.assertEqual(choose_fixed_beta({0.1: {"ndcg_at_k": 0.7, "sirr_at_k": 0.1},
                                             0.3: {"ndcg_at_k": 0.7, "sirr_at_k": 0.1}}), 0.1)

    def test_validation_threshold_is_deterministic(self):
        logits = np.array([[9., 0, 0, 0, 0, 0],
                           [3., 1, 0, 0, 0, 0],
                           [1., 1, 1, 1, 1, 1]], dtype=float)
        a = choose_abstention_threshold(logits, 0.9)
        b = choose_abstention_threshold(logits.copy(), 0.9)
        self.assertEqual(a, b)
        self.assertGreaterEqual(a, 0.0)

    def test_ablation_full_vs_top1_stage_distribution(self):
        def chunk(cid, v):
            return KnowledgeChunk(chunk_id=cid, text="Test text", topic="water_management",
                                  stage_compatibility=v, authority_score=0.5)
        chunks = {"a": chunk("a", (1, 0, 0, 0, 0, 0)),
                  "b": chunk("b", (0, 1, 0, 0, 0, 0))}
        s = {"stage_belief": "[0.51,0.49,0,0,0,0]", "raw_stage_belief": "[0.51,0.49,0,0,0,0]",
             "predicted_stage": "establishment"}
        self.assertEqual(set(variant_ranking({"a": 0.5, "b": 0.5}, chunks, s,
                                              "adaptive_top1_belief")), {"a", "b"})


class BlindedRetrievalTests(unittest.TestCase):
    def test_blind_selection_one_case_per_image(self):
        cases = [
            {"scenario_id": "img1__a", "ground_truth_stage": "establishment"},
            {"scenario_id": "img1__b", "ground_truth_stage": "establishment"},
            {"scenario_id": "img2__a", "ground_truth_stage": "ripening"},
        ]
        selected = select_cases(cases, 2, 42)
        self.assertEqual(len(selected), 2)
        self.assertEqual(len({r["scenario_id"].rsplit("__", 1)[0] for r in selected}), 2)

    def test_blank_and_partial_are_not_expert_labels(self):
        blank = {"item_id": "x", "uncertain": "", "relevance_grade": "",
                 "stage_inappropriate": "", "image_evidence_sufficient": ""}
        self.assertIsNone(judgment(blank))
        with self.assertRaisesRegex(ValueError, "partial"):
            judgment({**blank, "relevance_grade": "2"})
        self.assertIsNone(judgment({**blank, "uncertain": "yes"}))

    def test_scores_are_from_expert_grades_not_stage_metadata(self):
        graded = {"a": (2, False, True), "b": (0, True, True)}
        result = grade_case(["a", "b"], graded)
        self.assertAlmostEqual(result["precision_at_5"], 0.2)
        self.assertAlmostEqual(result["stage_inappropriate_at_5"], 0.2)

    def test_no_fabricated_results_from_blank_packet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = {"case_id": "CASE0001", "ranking_json": json.dumps({"A": ["chunkA"]})}
            item = {"item_id": "CASE0001D001", "case_id": "CASE0001", "chunk_id": "chunkA"}
            write_csv(root / "admin_cases.csv", [case], list(case))
            row = {key: "" for key in REVIEW_COLS}
            row.update(item_id=item["item_id"], case_id=item["case_id"])
            item["reviewer_context_sha256"] = context_hash(row)
            write_csv(root / "admin_items.csv", [item], list(item))
            for num in (1, 2):
                write_csv(root / f"reviewer_{num}.csv", [row], REVIEW_COLS)
            args = argparse.Namespace(admin_dir=root, reviewer_1=root/"reviewer_1.csv",
                                      reviewer_2=root/"reviewer_2.csv",
                                      output=root/"report.json")
            result = score(args)
            self.assertEqual(result["fully_consensus_cases"], 0)
            self.assertEqual(result["methods"], {})
            self.assertEqual(result["status"], "insufficient_expert_evidence")


if __name__ == "__main__":
    unittest.main()
