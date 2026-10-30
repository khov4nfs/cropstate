"""Original-parent-group bootstrap/sign-flip inference and error-stratified retrieval analysis.

Addresses two reviewer-mandated (P0) fixes for the CropState-Vision paper:

  * P0-3.8  Paired CI and Holm-adjusted tests resample whole original parent
            photographs (not overlapping image crops or repeated topics).
            Parent IDs must come from the split manifest; missing IDs fail closed.
  * P0-3.3  The error-stratified table (correct / adjacent / non-adjacent) must
            include the strongest baseline B1 (query expansion) for a fair
            comparison against the proposed adaptive gating (P).

Input : CROPSTATE_RESULTS/retrieval/retrieval_evaluation_belief.json
Output: CROPSTATE_RESULTS/retrieval/retrieval_significance_clustered.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS = Path("CROPSTATE_RESULTS/retrieval")
IN_PATH = RESULTS / "retrieval_evaluation_belief.json"
OUT_PATH = RESULTS / "retrieval_significance_clustered.json"

STAGE_ORDER = {
    "establishment": 0,
    "tillering": 1,
    "stem_booting": 2,
    "reproductive": 3,
    "grain_filling": 4,
    "ripening": 5,
}
METHODS = [
    "B0_ungated",
    "B1_query_expansion",
    "B2_hard_filter",
    "B3_fixed_soft",
    "P_adaptive_soft",
    "oracle_reference",
]
METRICS = ["ndcg_at_k", "recall_at_k", "precision_at_k", "sirr_at_k"]
PROPOSED = "P_adaptive_soft"
BASELINES = ["B0_ungated", "B1_query_expansion", "B3_fixed_soft"]
N_BOOT = 5000
SEED = 42


def error_condition(row: dict) -> str:
    delta = abs(STAGE_ORDER[row["ground_truth_stage"]] - STAGE_ORDER[row["predicted_stage"]])
    if delta == 0:
        return "correct"
    if delta == 1:
        return "adjacent"
    return "nonadjacent"


def image_id(scenario_id: str) -> str:
    # scenario_id == f"{image_id}__{topic}"
    return scenario_id.split("__")[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=IN_PATH)
    parser.add_argument("--image-manifest", type=Path, required=True,
                        help="Mapping of source image_id to original parent_image_id")
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.n_bootstrap < 100:
        parser.error("n-bootstrap must be >=100")
    data = json.loads(args.input.read_text())
    rows = data["rows"]

    by_scenario: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_scenario[row["scenario_id"]][row["method"]] = row
    scenarios = list(by_scenario)

    # Original parent photographs, not their overlapping crops, are sampling units.
    with args.image_manifest.open(encoding="utf-8-sig", newline="") as stream:
        manifest_rows = list(csv.DictReader(stream))
    parent_map = {str(row["image_id"]): str(row["parent_image_id"]) for row in manifest_rows}
    if any(not row["parent_image_id"] for row in manifest_rows):
        raise ValueError("Blank original parent_image_id in image manifest")
    if len(parent_map) != len(manifest_rows):
        raise ValueError("Duplicated image_id in image manifest")
    clusters: dict[str, list[str]] = defaultdict(list)
    for scenario in scenarios:
        img = image_id(scenario)
        if img not in parent_map:
            raise ValueError(f"Missing original parent_image_id for image {img}")
        clusters[parent_map[img]].append(scenario)
    cluster_keys = sorted(clusters)

    rng = np.random.default_rng(args.seed)

    def paired_bootstrap(proposed: str, baseline: str, metric: str) -> dict:
        diffs = {
            s: by_scenario[s][proposed][metric] - by_scenario[s][baseline][metric]
            for s in scenarios
        }
        observed = float(np.mean([diffs[s] for s in scenarios]))
        boot = np.empty(args.n_bootstrap)
        n_clusters = len(cluster_keys)
        for b in range(args.n_bootstrap):
            picks = rng.integers(0, n_clusters, n_clusters)
            vals = [diffs[s] for idx in picks for s in clusters[cluster_keys[idx]]]
            boot[b] = np.mean(vals)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        # A bootstrap distribution centred on the observed effect does not
        # directly impose the null; use cluster-wise paired sign flips to
        # test H0: zero mean effect, then correct the family of 12 tests.
        # A +1 Monte-Carlo adjustment prevents the impossible p=0.0.
        cluster_sums = np.array([
            sum(diffs[s] for s in clusters[c]) for c in cluster_keys
        ], dtype=float)
        permutations = rng.choice((-1.0, 1.0), size=(args.n_bootstrap, n_clusters))
        null_stats = permutations @ cluster_sums / len(scenarios)
        p = (1 + int(np.count_nonzero(np.abs(null_stats) >= abs(observed)))) / (
            args.n_bootstrap + 1
        )
        return {"diff": observed, "ci95": [float(lo), float(hi)],
                "p_raw": min(p, 1.0), "p_test": "paired cluster sign-flip"}

    comparisons = {}
    flat = []
    for baseline in BASELINES:
        comparisons[baseline] = {}
        for metric in METRICS:
            res = paired_bootstrap(PROPOSED, baseline, metric)
            comparisons[baseline][metric] = res
            flat.append((baseline, metric, res))

    # Holm-Bonferroni across the whole family of tests.
    flat.sort(key=lambda t: t[2]["p_raw"])
    m = len(flat)
    running = 0.0
    for i, (baseline, metric, res) in enumerate(flat):
        adj = min(1.0, (m - i) * res["p_raw"])
        adj = max(adj, running)
        running = adj
        comparisons[baseline][metric]["p_holm"] = adj

    # Error-stratified nDCG/SIRR for every method (incl. B1).
    strata: dict[str, list[str]] = defaultdict(list)
    for scenario in scenarios:
        strata[error_condition(by_scenario[scenario]["B0_ungated"])].append(scenario)

    stratified = {}
    for condition, scs in strata.items():
        stratified[condition] = {"n": len(scs), "methods": {}}
        for method in METHODS:
            stratified[condition]["methods"][method] = {
                "ndcg_at_k": float(np.mean([by_scenario[s][method]["ndcg_at_k"] for s in scs])),
                "sirr_at_k": float(np.mean([by_scenario[s][method]["sirr_at_k"] for s in scs])),
            }

    out = {
        "n_scenarios": len(scenarios),
        "n_images": len({image_id(s) for s in scenarios}),
        "n_parent_groups": len(cluster_keys),
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "proposed": PROPOSED,
        "clustering": "original parent_image_id (all overlapping crops and topics grouped)",
        "multiple_comparison_correction": "holm",
        "p_value_test": "paired cluster sign-flip permutation with +1 correction",
        "n_signflip_permutations": args.n_bootstrap,
        "paired_bootstrap": comparisons,
        "error_stratified": stratified,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
