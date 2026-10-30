"""Leakage-controlled retrieval ablations using validation-only selection.

This is a *metadata-derived diagnostic*, NOT independent agronomic validation.
Candidate retrieval/embeddings are cached per topic; no test metric participates
in selecting the fixed soft weight or the selective prediction threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cropstate.confidence import entropy_concentration, top_two_margin, combined_confidence
from cropstate.knowledge import load_knowledge_chunks
from cropstate.metrics import ndcg_at_k, precision_recall_at_k, sirr_at_k
from cropstate.retrieval import (
    HybridRetriever, build_topic_query, hard_filter, minmax,
    reciprocal_rank_fusion, rerank,
)
from scripts.evaluate_retrieval import (
    normalize_stage, one_hot, parse_belief, parse_ids, parse_relevance, read_scenarios,
)

METRICS = ("precision_at_k", "recall_at_k", "ndcg_at_k", "sirr_at_k")
GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_ids(rows: list[dict]) -> set[str]:
    return {str(r["scenario_id"]).rsplit("__", 1)[0] for r in rows}


def validate_partitions(validation: list[dict], test: list[dict]) -> None:
    if not validation or not test:
        raise ValueError("Both validation and test scenarios are required")
    for part in (validation, test):
        ids = [r["scenario_id"] for r in part]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate scenario IDs within a partition")
        if any(r.get("calibration_applied") != "true" or
               r.get("confidence_method") != "combined" for r in part):
            raise ValueError("Expected temperature-calibrated combined-confidence scenarios")
        if any(not r.get("raw_stage_belief") for r in part):
            raise ValueError("Missing raw belief for calibrated/raw ablation")
    if image_ids(validation) & image_ids(test):
        raise ValueError("Validation/test source image leakage")


def choose_fixed_beta(validation_means: dict[float, dict[str, float]]) -> float:
    """Predeclared pilot utility: equal-weight nDCG@5 minus metadata-SIRR@5."""
    if not validation_means:
        raise ValueError("No validation metrics")
    return min(validation_means, key=lambda b: (
        -(validation_means[b]["ndcg_at_k"] - validation_means[b]["sirr_at_k"]), b
    ))


def make_confidence(belief: np.ndarray, kind: str) -> float:
    if kind == "combined":
        return combined_confidence(belief)
    if kind == "entropy":
        return entropy_concentration(belief)
    if kind == "margin":
        return top_two_margin(belief)
    if kind == "top1":
        return float(np.max(belief))
    raise ValueError(f"Unknown confidence {kind}")


def variant_ranking(
    base_scores: dict[str, float], chunk_map: dict, scenario: dict,
    variant: str, selected_beta: float = 0.20,
) -> list[str]:
    belief = parse_belief(scenario["stage_belief"], scenario["predicted_stage"])
    confidence_kind = {
        "adaptive_combined": "combined",
        "adaptive_entropy": "entropy",
        "adaptive_margin": "margin",
        "adaptive_top1_confidence": "top1",
        "adaptive_top1_belief": "combined",
        "adaptive_raw_belief": "combined",
    }.get(variant, "combined")
    if variant == "adaptive_raw_belief":
        belief = parse_belief(scenario["raw_stage_belief"], scenario["predicted_stage"])
    if variant == "adaptive_top1_belief":
        belief = one_hot(scenario["predicted_stage"])
    conf = make_confidence(belief, confidence_kind)
    if variant == "hard_filter":
        return hard_filter(base_scores, chunk_map, int(np.argmax(belief)))
    beta = selected_beta if variant == "fixed_validation_selected" else (
        0.20 if variant == "fixed_legacy_020" else None
    )
    return rerank(base_scores, chunk_map, belief, conf, source_weight=0.0, fixed_beta=beta)


def measure(ranking: list[str], scenario: dict, k: int = 5) -> dict[str, float]:
    relevance = parse_relevance(scenario)
    relevant = {key for key, grade in relevance.items() if grade > 0}
    incompatible = parse_ids(scenario["incompatible_chunk_ids"])
    precision, recall = precision_recall_at_k(ranking, relevant, k)
    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "ndcg_at_k": ndcg_at_k(ranking, relevance, k),
        "sirr_at_k": sirr_at_k(ranking, incompatible, k),
    }


def average(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        raise ValueError("No scored cases")
    return {metric: float(np.mean([r[metric] for r in metrics])) for metric in METRICS}


def base_candidates(retriever: HybridRetriever, scenarios: list[dict]) -> dict[str, dict[str, float]]:
    cached = {}
    for topic in sorted({str(s["topic"]) for s in scenarios}):
        bm25, dense = retriever.retrieve(build_topic_query(topic, language="en"), depth=50, topic=topic,
                                         include_general_fallback=False)
        cached[topic] = minmax(reciprocal_rank_fusion([bm25, dense]))
    return cached


def score_variant(rows: list[dict], candidates: dict, chunks: dict,
                  variant: str, beta: float = 0.20) -> dict[str, float]:
    return average([
        measure(variant_ranking(candidates[r["topic"]], chunks, r, variant, beta), r)
        for r in rows
    ])


def choose_abstention_threshold(validation_logits: np.ndarray, temperature: float,
                                coverage_target: float = 0.70) -> float:
    """Select threshold by validation coverage ONLY, not by looking at test errors."""
    from scripts.experiments.build_belief_scenarios import softmax_np
    probs = softmax_np(validation_logits.astype(float) / temperature)
    conf = np.array([combined_confidence(row) for row in probs])
    thresholds = [0.0] + sorted(set(float(x) for x in conf))
    return min(thresholds, key=lambda t: (
        abs(float(np.mean(conf >= t)) - coverage_target),
        -float(np.mean(conf >= t)),
        t,
    ))


def selective_evaluation(logits: np.ndarray, labels: np.ndarray,
                         temperature: float, threshold: float) -> dict:
    from scripts.experiments.build_belief_scenarios import softmax_np
    probs = softmax_np(logits.astype(float) / temperature)
    conf = np.array([combined_confidence(row) for row in probs])
    correct = probs.argmax(axis=1) == labels
    def at(t: float) -> dict:
        mask = conf >= t
        return {"threshold": round(t, 6), "coverage": float(mask.mean()),
                "retained": int(mask.sum()),
                "selective_risk": float(1.0 - correct[mask].mean()) if mask.any() else None}
    quantiles = sorted(set(float(x) for x in np.quantile(conf, np.linspace(0, 1, 21))))
    curve = [at(t) for t in quantiles]
    return {"selected": at(threshold), "full_risk": float(1.0 - correct.mean()),
            "risk_coverage_curve": curve, "n": len(labels)}



def paired_cluster_comparisons(test_rows: list[dict], candidates: dict, chunks: dict,
                               selected_beta: float, parent_by_image: dict[str, str],
                               n_resamples: int = 5000, seed: int = 42) -> dict:
    """Paired ORIGINAL-PARENT-group bootstrap CIs and sign-flip p values."""
    from collections import defaultdict
    variants = ("adaptive_combined", "fixed_validation_selected",
                "adaptive_top1_belief", "adaptive_top1_confidence")
    scores = {v: [
        measure(variant_ranking(candidates[r["topic"]], chunks, r, v, selected_beta), r)
        for r in test_rows
    ] for v in variants}
    ids = sorted({parent_by_image[name] for name in image_ids(test_rows)})
    cluster_idx = {name: i for i, name in enumerate(ids)}
    counts = np.zeros(len(ids), dtype=int)
    for row in test_rows:
        counts[cluster_idx[parent_by_image[row["scenario_id"].rsplit("__", 1)[0]]]] += 1
    rng = np.random.default_rng(seed)
    comparisons = {}
    tests = []
    for variant in variants[1:]:
        comparisons[variant] = {}
        for metric in ("ndcg_at_k", "sirr_at_k"):
            sums = np.zeros(len(ids), dtype=float)
            for i, row in enumerate(test_rows):
                name = parent_by_image[row["scenario_id"].rsplit("__", 1)[0]]
                sums[cluster_idx[name]] += (scores[variant][i][metric] -
                                             scores["adaptive_combined"][i][metric])
            observed = float(sums.sum() / counts.sum())
            picks = rng.integers(0, len(ids), size=(n_resamples, len(ids)))
            bs = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
            low, high = np.quantile(bs, [0.025, 0.975])
            signs = rng.choice((-1.0, 1.0), size=(n_resamples, len(ids)))
            null = (signs * sums).sum(axis=1) / counts.sum()
            raw = (1 + int(np.count_nonzero(np.abs(null) >= abs(observed) - 1e-14))) / (n_resamples + 1)
            entry = {"delta_vs_adaptive": observed, "cluster_ci95": [float(low), float(high)],
                     "p_raw_two_sided": raw}
            comparisons[variant][metric] = entry
            tests.append(entry)
    ordered = sorted(tests, key=lambda d: d["p_raw_two_sided"])
    previous = 0.0
    for i, entry in enumerate(ordered):
        previous = max(previous, min(1.0, (len(tests) - i) * entry["p_raw_two_sided"]))
        entry["p_holm"] = previous
    return {"n_parent_groups": len(ids), "n_images": len(image_ids(test_rows)),
            "clustering": "original parent_image_id",
            "n_scenarios": len(test_rows), "n_resamples": n_resamples,
            "seed": seed, "multiple_testing": "Holm across 6 exploratory tests",
            "comparisons": comparisons}

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--validation-scenarios", type=Path, required=True)
    ap.add_argument("--test-scenarios", type=Path, required=True)
    ap.add_argument("--logits", type=Path, required=True)
    ap.add_argument("--image-manifest", type=Path, required=True,
                    help="Validate disjoint original parent_image_id groups, not only image IDs")
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--reference-evaluation", type=Path,
                    help="Optional previously verified controlled-protocol evaluation for concordance assertion")
    ap.add_argument("--embedding-backend", choices=("onnx",), default="onnx")
    ap.add_argument("--onnx-model-dir", type=Path)
    args = ap.parse_args()

    val_rows, test_rows = read_scenarios(args.validation_scenarios), read_scenarios(args.test_scenarios)
    validate_partitions(val_rows, test_rows)
    import csv
    with args.image_manifest.open(encoding="utf-8-sig", newline="") as stream:
        manifest = {r["image_id"]: r["parent_image_id"] for r in csv.DictReader(stream)}
    if not image_ids(val_rows).issubset(manifest) or not image_ids(test_rows).issubset(manifest):
        raise ValueError("Missing original parent identifiers")
    val_parents = {manifest[i] for i in image_ids(val_rows)}
    test_parents = {manifest[i] for i in image_ids(test_rows)}
    if not all(val_parents | test_parents) or val_parents & test_parents:
        raise ValueError("Validation/test parent_image_id leakage")
    logits = np.load(args.logits, allow_pickle=False)
    temp = float(json.loads(args.calibration.read_text())["resnet18"]["temperature"])
    if not np.isfinite(temp) or temp <= 0:
        raise ValueError("Invalid prefit temperature")
    if not image_ids(val_rows).issubset(set(map(str, logits["validation_image_ids"]))) or not image_ids(test_rows).issubset(set(map(str, logits["test_image_ids"]))):
        raise ValueError("Validation/test scenarios do not match their corresponding logits image IDs")
    from cropstate.onnx_encoder import ONNXSentenceEncoder
    encoder = ONNXSentenceEncoder(model_dir=args.onnx_model_dir)
    chunks = load_knowledge_chunks(args.corpus, mode="research")
    chunk_map = {c.chunk_id: c for c in chunks}
    retriever = HybridRetriever(chunks, encoder=encoder)
    val_cache = base_candidates(retriever, val_rows)
    test_cache = base_candidates(retriever, test_rows)

    grid = {b: score_variant(val_rows, val_cache, chunk_map, "fixed_validation_selected", b) for b in GRID}
    selected_beta = choose_fixed_beta(grid)
    variants = ("adaptive_combined", "adaptive_entropy", "adaptive_margin",
                "adaptive_top1_confidence", "adaptive_top1_belief",
                "adaptive_raw_belief", "fixed_validation_selected",
                "fixed_legacy_020", "hard_filter")
    validation = {v: score_variant(val_rows, val_cache, chunk_map, v, selected_beta) for v in variants}
    test = {v: score_variant(test_rows, test_cache, chunk_map, v, selected_beta) for v in variants}
    if args.reference_evaluation:
        baseline = json.loads(args.reference_evaluation.read_text())["summary"]
        for original, variant in (("P_adaptive_soft", "adaptive_combined"),
                                  ("B3_fixed_soft", "fixed_legacy_020"),
                                  ("B2_hard_filter", "hard_filter")):
            for metric in METRICS:
                if abs(baseline[original][metric] - test[variant][metric]) > 1e-9:
                    raise ValueError(f"Reference mismatch for {original} {metric}; don't report ablation")
    paired = paired_cluster_comparisons(test_rows, test_cache, chunk_map, selected_beta, manifest)
    threshold = choose_abstention_threshold(logits["validation_logits"], temp)
    out = {
        "protocol": "English queries on English corpus; parent-group clustered; metadata-derived ablation, not expert relevance",
        "selection": {"partition": "validation_only", "fixed_beta_grid": list(GRID),
                      "fixed_beta_objective": "maximize nDCG@5 minus metadata-defined SIRR@5",
                      "fixed_beta_selected": selected_beta,
                      "validation_grid_metrics": {str(b): grid[b] for b in GRID},
                      "abstention_coverage_target": 0.70,
                      "abstention_threshold_selected_validation": threshold},
        "counts": {"validation_scenarios": len(val_rows), "test_scenarios": len(test_rows),
                   "validation_parent_groups": len(val_parents), "test_parent_groups": len(test_parents),
                   "validation_images": len(image_ids(val_rows)), "test_images": len(image_ids(test_rows))},
        "validation": validation, "test": test,
        "paired_cluster_comparisons": paired,
        "reference_evaluation_sha256": sha256(args.reference_evaluation) if args.reference_evaluation else None,
        "selective_prediction_weak_labels": {
            "validation": selective_evaluation(logits["validation_logits"], logits["validation_labels"], temp, threshold),
            "test": selective_evaluation(logits["test_logits"], logits["test_labels"], temp, threshold),
        },
        "source_checksums": {
            str(p): sha256(p) for p in (args.corpus, args.validation_scenarios,
                                        args.test_scenarios, args.logits, args.calibration, args.image_manifest)
        },
        "embedding_backend": {"model_sha256": encoder.model_sha256,
                              "tokenizer_sha256": encoder.tokenizer_sha256, "engine": "ONNX FP32"},
        "limitations": [
            "Selection uses automatically derived relevance and SIRR; does not demonstrate agronomic safety.",
            "Stage labels are morphology-derived weak labels; held-out results are not expert BBCH accuracy.",
            "Selecting beta on validation does not validate its utility against independent expert relevance.",
            "Fixed beta protocol and retrieval candidate pool match controlled historical rerun except beta.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected_beta": selected_beta, "abstention_threshold": threshold,
                       "test": test}, indent=2))


if __name__ == "__main__":
    main()
