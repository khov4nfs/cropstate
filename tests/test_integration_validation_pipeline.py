"""Offline integration test for the full blinded review prepare -> score lifecycle.

All labels are synthetic TEST FIXTURES only, never real expert annotations.
"""
from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from cropstate.knowledge import KnowledgeChunk
from scripts.prepare_retrieval_expert_review import (
    REVIEW_COLS, prepare, read_csv, score, write_csv,
)


class ReviewPacketIntegrationTest(unittest.TestCase):
    def test_end_to_end_blinding_and_double_consensus(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            data_root = root / "images"
            data_root.mkdir()
            scenarios = root / "scenarios.csv"
            manifest = root / "manifest.csv"
            evaluation = root / "evaluation.json"
            corpus = root / "chunks.jsonl"
            packet = root / "packet"
            image_ids = ("synthetic_img_A", "synthetic_img_B")
            for i, image_id in enumerate(image_ids):
                Image.new("RGB", (24, 24), color=(i*40+10, 80, 80)).save(data_root / (image_id + ".jpg"))
            write_csv(manifest, [
                {"image_id": image_id, "image_path": image_id + ".jpg",
                 "parent_image_id": f"parent{i}"} for i, image_id in enumerate(image_ids)
            ], ["image_id", "image_path", "parent_image_id"])
            scs = [
                {"scenario_id": image_id + "__water_management", "ground_truth_stage": stage,
                 "predicted_stage": stage, "topic": "water_management"}
                for image_id, stage in zip(image_ids, ("establishment", "ripening"))
            ]
            write_csv(scenarios, scs, list(scs[0]))
            methods = ("B0_ungated", "B1_query_expansion", "B2_hard_filter",
                       "B3_fixed_soft", "P_adaptive_soft", "oracle_reference")
            evaluation.write_text(json.dumps({
                "scenario_count": 2,
                "rows": [{"scenario_id": sc["scenario_id"], "method": method,
                          "top_ids": ["A", "B"] if method != "B2_hard_filter" else ["A"]}
                         for sc in scs for method in methods],
            }))
            chunks = [
                KnowledgeChunk(chunk_id="A", text="Rice irrigation evidence passage.",
                               topic="water_management", stage_compatibility=(1,)*6,
                               authority_score=0.5, source_title="IRRI resource"),
                KnowledgeChunk(chunk_id="B", text="Rice cultivation evidence passage.",
                               topic="water_management", stage_compatibility=(0,)*6,
                               authority_score=0.5, source_title="IRRI resource"),
            ]
            with patch("scripts.prepare_retrieval_expert_review.load_knowledge_chunks",
                       return_value=chunks):
                status = prepare(argparse.Namespace(
                    scenarios=scenarios, evaluation=evaluation, corpus=corpus,
                    image_manifest=manifest, data_root=data_root, output_dir=packet,
                    n_cases=2, seed=42,
                ))
            self.assertEqual(status["n_cases"], 2)
            self.assertEqual(status["n_items"], 4)
            blinded = read_csv(packet/"reviewer_1.csv", "item_id")
            for field in ("weak_stage", "predicted_stage", "method", "chunk_id",
                          "stage_compatibility", "ground_truth_stage"):
                self.assertNotIn(field, REVIEW_COLS)
            for row in blinded.values():
                with Image.open(packet/"assets"/row["image_filename"]) as im:
                    self.assertFalse(im.getexif())
            blank = score(argparse.Namespace(
                admin_dir=packet, reviewer_1=packet/"reviewer_1.csv",
                reviewer_2=packet/"reviewer_2.csv", output=packet/"blank.json"))
            self.assertEqual(blank["fully_consensus_cases"], 0)
            # Fill artificial test-only consensus: reviewer files are independently constructed.
            items = read_csv(packet/"admin_items.csv", "item_id")
            for reviewer_num in (1, 2):
                rows = []
                for rid, row in blinded.items():
                    row = dict(row)
                    row["relevance_grade"] = "2" if items[rid]["chunk_id"] == "A" else "0"
                    row["stage_inappropriate"] = "no"
                    row["image_evidence_sufficient"] = "yes"
                    row["uncertain"] = "no"
                    rows.append(row)
                write_csv(packet/f"fixture_reviewer_{reviewer_num}.csv", rows, REVIEW_COLS)
            completed = score(argparse.Namespace(
                admin_dir=packet, reviewer_1=packet/"fixture_reviewer_1.csv",
                reviewer_2=packet/"fixture_reviewer_2.csv", output=packet/"synthetic_only.json"))
            self.assertEqual(completed["fully_consensus_cases"], 2)
            self.assertEqual(completed["cases_with_sufficient_image_evidence"], 2)
            self.assertEqual(completed["methods"]["B0_ungated"]["pooled_ndcg_at_5"], 1.0)


if __name__ == "__main__":
    unittest.main()
