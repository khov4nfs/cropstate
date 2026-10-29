"""Reproduce the 420 image-derived, metadata-defined retrieval scenarios.

FAIL CLOSED: no score, table, or 'successful rerun' is produced if source data
or the expected count is absent. Run with original corpus and logits externally.

Example:
 PYTHONPATH=src python scripts/reproduce_retrieval_benchmark.py \
   --corpus CROPSTATE_KNOWLEDGE_BASE/chunks/rice_knowledge_irri_en_nonrestricted.jsonl \
   --logits-index CROPSTATE_RESULTS/novelty/logits/index.json \
   --calibration CROPSTATE_RESULTS/novelty/calibration.json \
   --output-dir CROPSTATE_RESULTS/retrieval/reverified
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def assert_scenario_provenance(scenarios: Path, expected_scenarios: int,
                               expected_images: int, allow_legacy: bool = False) -> dict[str, int]:
    with scenarios.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "scenario_id", "stage_belief", "confidence", "ground_truth_stage",
        "predicted_stage", "relevant_chunk_ids", "incompatible_chunk_ids",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Scenario file lacks required columns: {sorted(required - set(rows[0]) if rows else required)}")
    if len(rows) != expected_scenarios:
        raise ValueError(
            f"Expected {expected_scenarios} independently identified scenario rows, got {len(rows)}. "
            "Do not relabel a different evaluation set as the reported 420 scenarios."
        )
    # Old scenario CSV files used raw softmax probabilities and top-1
    # confidence. Never silently call them calibrated experimental results.
    if not allow_legacy:
        if any(r.get("confidence_method") != "combined" or
               not r.get("calibration_temperature") or
               r.get("calibration_applied") != "true" for r in rows):
            raise ValueError(
                "Expected scenario rows generated with calibrated belief and "
                "confidence_method=combined. Regenerate from original logits or "
                "explicitly pass --allow-legacy-beliefs for the old-protocol ablation."
            )
    ids = [r["scenario_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate scenario IDs")
    # Scenario creator encodes '{image_id}__{topic}', not a free-form scenario id.
    if any("__" not in sid for sid in ids):
        raise ValueError("Image-derived scenario IDs must include '__{topic}'")
    images = {sid.rsplit("__", 1)[0] for sid in ids}
    if len(images) != expected_images:
        raise ValueError(f"Expected {expected_images} contributing images, got {len(images)}")
    if any(not r["relevant_chunk_ids"].strip() or not r["incompatible_chunk_ids"].strip()
           for r in rows):
        raise ValueError("Empty automated relevance/incompatibility lists")
    return {"scenario_count": len(rows), "contributing_images": len(images)}


def run(command: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT, env=env)


def write_camera_ready_table(evaluation: Path, significance: Path, path: Path) -> None:
    ev = json.loads(evaluation.read_text(encoding="utf-8"))
    sig = json.loads(significance.read_text(encoding="utf-8"))
    summary = ev["summary"]
    methods = [
        ("B0_ungated", "B0: no gating"),
        ("B1_query_expansion", "B1: query expansion"),
        ("B2_hard_filter", "B2: hard filter"),
        ("B3_fixed_soft", "B3: fixed soft"),
        ("P_adaptive_soft", "P: adaptive soft"),
        ("oracle_reference", "Oracle"),
    ]
    lines = [
        "# Recomputed Retrieval Results — NOT Expert-Judged",
        "",
        "These metrics use automatically derived relevance labels based on stage "
        "compatibility metadata; they measure mechanism consistency, not agronomic efficacy.",
        "",
        "| Method | P@5 | R@5 | nDCG@5 | SIRR@5 (lower) |",
        "|---|---:|---:|---:|---:|",
    ]
    keys = ("precision_at_k", "recall_at_k", "ndcg_at_k", "sirr_at_k")
    for name, label in methods:
        values = summary[name]
        lines.append("| " + label + " | " + " | ".join(f"{values[k]:.3f}" for k in keys) + " |")
    lines += ["", "## Clustered paired bootstrap (P minus B1 query expansion)", ""]
    comparisons = sig["paired_bootstrap"]["B1_query_expansion"]
    for key, label in [("ndcg_at_k", "nDCG@5"), ("recall_at_k", "Recall@5"),
                       ("precision_at_k", "Precision@5"), ("sirr_at_k", "SIRR@5")]:
        diff = comparisons[key]
        lo, hi = diff["ci95"]
        lines.append(f"- {label}: Δ={diff['diff']:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}], "
                     f"Holm-adjusted cluster sign-flip p={diff['p_holm']:.4f}.")
    lines += [
        "",
        "A nonsignificant ranking difference is **not proof of equivalence**.",
        "SIRR is a metadata-defined stage incompatibility rate, not comprehensive "
        "recommendation safety.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--scenarios", type=Path, help="Original, real 420-scenario CSV")
    inputs.add_argument("--logits-index", type=Path, help="Index with real saved test logits")
    parser.add_argument("--calibration", type=Path,
                        help="Validation-fitted temperature when generating from logits")
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--embedding-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--embedding-backend", choices=("sentence-transformers", "onnx"),
                        default="sentence-transformers")
    parser.add_argument("--onnx-model-dir", type=Path,
                        help="Locally pinned model snapshot when --embedding-backend=onnx")
    parser.add_argument("--expected-scenarios", type=int, default=420)
    parser.add_argument("--expected-images", type=int, default=199)
    parser.add_argument("--query-language", choices=("en", "vi"), default="en",
                        help="English corpus: en for corrected study; vi only to replay legacy.")
    parser.add_argument("--image-manifest", type=Path,
                        help="Manifest of image_id -> parent_image_id; REQUIRED for scientific inference.")
    parser.add_argument("--fixed-beta", type=float, default=0.20)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--allow-legacy-beliefs", action="store_true",
                        help="Allow historical raw/top1 beliefs; results must be labeled legacy.")
    parser.add_argument("--check-only", action="store_true",
                        help="Validate required inputs and scenarios without importing encoder/model")
    args = parser.parse_args()

    paths = [args.corpus, args.scenarios or args.logits_index]
    if not args.check_only:
        paths.append(args.image_manifest)
    if args.logits_index:
        paths.append(args.calibration)
    missing = [str(p) for p in paths if p is None or not p.is_file()]
    if missing:
        parser.error("Required experiment artifacts are missing: " + ", ".join(missing))
    if args.logits_index:
        logits_index = json.loads(args.logits_index.read_text())
        if args.model not in logits_index:
            parser.error(f"Model {args.model!r} absent from logits index")
        logits_file = Path(logits_index[args.model]["path"])
        if not logits_file.is_absolute():
            # Exported logits indexes commonly store paths relative to the
            # working repository root, not relative to the index file itself.
            from_cwd = (ROOT / logits_file).resolve()
            from_index = (args.logits_index.parent / logits_file).resolve()
            logits_file = from_cwd if from_cwd.is_file() else from_index
        if not logits_file.is_file():
            parser.error(f"Referenced logits archive does not exist: {logits_file}")
        # Existing scenario builder expects paths recorded by index to be usable
        # from the working directory; adjust only the copied index, never source.
        args.output_dir.mkdir(parents=True, exist_ok=True)
        normalized = dict(logits_index)
        normalized[args.model] = {**normalized[args.model], "path": str(logits_file)}
        index = args.output_dir / "normalized_logits_index.json"
        index.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
        scenarios = args.output_dir / "belief_scenarios_calibrated.csv"
        if not args.check_only:
            run([sys.executable, "scripts/experiments/build_belief_scenarios.py",
                 "--logits-index", str(index.resolve()), "--model", args.model,
                 "--corpus", str(args.corpus.resolve()),
                 "--calibration", str(args.calibration.resolve()),
                 "--confidence-method", "combined",
                 "--output", str(scenarios.resolve())])
    else:
        scenarios = args.scenarios
    if scenarios.is_file():
        checked = assert_scenario_provenance(
            scenarios, args.expected_scenarios, args.expected_images,
            allow_legacy=args.allow_legacy_beliefs,
        )
    elif args.check_only and args.logits_index:
        print("CHECK ONLY: logits and corpus exist; scenario counts require generation.")
        return
    else:
        raise FileNotFoundError(f"Scenarios missing: {scenarios}")
    print("Verified scenario structure:", checked)
    if args.check_only:
        print("CHECK ONLY: no benchmark scores were generated.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluation = args.output_dir / "retrieval_evaluation.json"
    significance = args.output_dir / "retrieval_significance.json"
    run([sys.executable, "scripts/evaluate_retrieval.py",
         "--corpus", str(args.corpus.resolve()), "--scenarios", str(scenarios.resolve()),
         "--embedding-model", args.embedding_model,
         "--embedding-backend", args.embedding_backend,
         * (["--onnx-model-dir", str(args.onnx_model_dir.resolve())]
            if args.onnx_model_dir else []),
         "--mode", "research", "--source-weight", "0",
         "--fixed-beta", str(args.fixed_beta), "--query-language", args.query_language,
         "--output", str(evaluation.resolve())])
    run([sys.executable, "scripts/analyze_retrieval_significance_clustered.py",
         "--input", str(evaluation.resolve()),
         "--image-manifest", str(args.image_manifest.resolve()),
         "--output", str(significance.resolve()), "--n-bootstrap", str(args.bootstrap)])
    write_camera_ready_table(evaluation, significance,
                             args.output_dir / "CAMERA_READY_RETRIEVAL_TABLE.md")
    provenance = {
        "corpus_sha256": digest(args.corpus),
        "scenario_sha256": digest(scenarios),
        "evaluation_sha256": digest(evaluation),
        "significance_sha256": digest(significance),
        "embedding_backend_metadata": json.loads(evaluation.read_text())[
            "evaluation_config"]["embedding_backend"],
        "n_bootstrap": args.bootstrap,
        "p_value_test": "paired ORIGINAL-PARENT-group sign-flip with +1 correction",
        "query_language": args.query_language,
        "image_manifest_sha256": digest(args.image_manifest),
        "calibration_sha256": digest(args.calibration) if args.calibration else None,
        "scenario_count": checked["scenario_count"],
        "contributing_images": checked["contributing_images"],
        "generated_with": "scripts/reproduce_retrieval_benchmark.py",
        "embedding_backend": args.embedding_backend,
        "embedding_model": args.embedding_model,
        "allow_legacy_beliefs": args.allow_legacy_beliefs,
        "notes": "Metadata-only mechanism study; independent agronomic validity is an explicit limitation.",
    }
    (args.output_dir / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print("Recomputed benchmark artifacts:", args.output_dir)


if __name__ == "__main__":
    main()
