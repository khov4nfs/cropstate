"""Deterministic fixtures for English lexical matching, parent clustering and top-k fill."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cropstate.retrieval import BM25Okapi, build_topic_query, tokenize


class QueryLanguageProtocolTests(unittest.TestCase):
    def test_english_topic_query_has_real_lexical_signal(self):
        texts = [
            "Rice water management irrigation practices for small farms",
            "Rice pests and disease management crop protection",
            "Soil testing before fertilizer applications",
        ]
        model = BM25Okapi([tokenize(t) for t in texts])
        query = build_topic_query("water_management", language="en")
        scores = list(model.get_scores(tokenize(query)))
        self.assertGreater(scores[0], scores[2])
        self.assertIn("rice", tokenize(query))
        self.assertIn("water", tokenize(query))

    def test_legacy_vietnamese_query_remains_explicit(self):
        vi = build_topic_query("water_management", language="vi")
        en = build_topic_query("water_management", language="en")
        self.assertNotEqual(vi, en)
        self.assertIn("water", en)


class ParentGroupInferenceTests(unittest.TestCase):
    def test_shared_parent_crops_form_one_cluster(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            manifest = folder / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, ["image_id", "parent_image_id"])
                writer.writeheader()
                writer.writerows([
                    {"image_id": "crop_a", "parent_image_id": "photo_1"},
                    {"image_id": "crop_b", "parent_image_id": "photo_1"},
                    {"image_id": "crop_c", "parent_image_id": "photo_2"},
                ])
            methods = ("B0_ungated", "B1_query_expansion", "B2_hard_filter",
                       "B3_fixed_soft", "P_adaptive_soft", "oracle_reference")
            rows = []
            for image in ("crop_a", "crop_b", "crop_c"):
                for method in methods:
                    rows.append({
                        "scenario_id": image + "__water_management",
                        "method": method,
                        "ground_truth_stage": "tillering",
                        "predicted_stage": "tillering",
                        "ndcg_at_k": 0.8 if method == "P_adaptive_soft" else 0.5,
                        "recall_at_k": 0.7 if method == "P_adaptive_soft" else 0.4,
                        "precision_at_k": 0.6 if method == "P_adaptive_soft" else 0.3,
                        "sirr_at_k": 0.1 if method == "P_adaptive_soft" else 0.2,
                    })
            incoming = folder / "evaluated.json"
            outgoing = folder / "significance.json"
            incoming.write_text(json.dumps({"rows": rows}), encoding="utf-8")
            script = Path(__file__).resolve().parents[1] / "scripts" / "analyze_retrieval_significance_clustered.py"
            subprocess.run([
                sys.executable, str(script), "--input", str(incoming),
                "--image-manifest", str(manifest), "--output", str(outgoing),
                "--n-bootstrap", "100",
            ], check=True, capture_output=True, text=True)
            result = json.loads(outgoing.read_text())
            self.assertEqual(result["n_images"], 3)
            self.assertEqual(result["n_parent_groups"], 2)
            self.assertIn("parent_image_id", result["clustering"])

    def test_missing_parent_ids_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "manifest.csv").write_text("image_id,parent_image_id\ncrop_a,photo_1\n")
            (p / "input.json").write_text(json.dumps({
                "rows": [{
                    "scenario_id": "unknown__topic",
                    "method": "B0_ungated",
                    "ground_truth_stage": "tillering",
                    "predicted_stage": "tillering",
                }]
            }))
            script = Path(__file__).resolve().parents[1] / "scripts" / "analyze_retrieval_significance_clustered.py"
            r = subprocess.run([sys.executable, str(script),
                                "--input", str(p / "input.json"),
                                "--image-manifest", str(p / "manifest.csv"),
                                "--output", str(p / "out.json"),
                                "--n-bootstrap", "100"],
                               capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("Missing original parent_image_id", r.stderr)


if __name__ == "__main__":
    unittest.main()
