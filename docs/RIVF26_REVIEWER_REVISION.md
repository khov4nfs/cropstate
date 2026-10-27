# RIVF 2026 reviewer-driven revision: evidence and experiment integrity

Status: **work in progress**. Do not replace camera-ready numerical tables with outputs
from changed code until the complete experiment has been rerun on the fixed evaluation
artifacts. This document separates implemented engineering corrections from **unverified
scientific claims**. The main paper remains the source of the previously *reported* numbers.

## Changes included in this PR (not a new result)

1. `scripts/experiments/build_belief_scenarios.py` now uses the temperature-calibrated
   six-class probability vector in the evaluator's `stage_belief` column and computes
   the confidence mixture (normalized entropy 0.6 + top-two margin 0.4). Previously
   `stage_belief` was raw probabilities, `calibrated_stage_belief` was ignored by
   `evaluate_retrieval.py`, and `confidence` came from raw top-1 probability.
   Both raw and calibrated vectors, temperature, and confidence method are now
   included for traceability. `--confidence-method top1` provides an explicit
   comparison, not the default.
2. `scripts/evaluate_retrieval.py` now defaults to `--source-weight 0` to isolate
   stage-conditioning from a source-authority bonus that was previously present
   only for some ranking methods; the value is recorded in results. The general
   fallback is opt-in, because allowing unjudged `general_crop_care` chunks into
   a topic-specific evaluation silently changes the relevance universe. Do not
   enable it unless those chunks receive explicit judgments. `--fixed-beta` is
   configurable but **not validation-tuned automatically**.
3. A lightweight unit test verifies the calibrated-stage handoff; it cannot replace
   a rerun of the 420-scenario benchmark.

## Recompute before revising numbers

- Freeze the input image manifest and predicted-logit archive; record SHA256 sums.
- Fit temperature on validation logits only (never on test).
- Generate scenarios from test logits with `build_belief_scenarios.py`, explicitly
  recording the selected confidence method.
- Re-evaluate B0/B1/B2/B3/P/oracle on the **same** corpus, topic restrictions,
  candidate depth, queries, embeddings, and `source_weight=0`.
- Tune fixed-beta and any abstention threshold on **validation scenarios/images**,
  and apply frozen settings to test.
- Report P@5, R@5, nDCG@5, SIRR@5 and per-image-cluster paired bootstrap CIs;
  do not treat a nonsignificant difference as equivalence.
- Record why 263 test images yield only 199 contributing images and 420 answerable
  scenarios. Measure the selection bias and release the scenario-generation rules.
- An independent, blinded agronomist-labeled relevance set is still required for
  claims about substantive agronomic advice or safety. Metadata-derived SIRR measures
  **stage incompatibility**, not complete agricultural safety.

Example (paths to external artifacts must exist locally):

```bash
PYTHONPATH=src python scripts/experiments/build_belief_scenarios.py \
  --logits-index CROPSTATE_RESULTS/novelty/logits/index.json \
  --calibration CROPSTATE_RESULTS/novelty/calibration.json \
  --confidence-method combined \
  --corpus CROPSTATE_KNOWLEDGE_BASE/chunks/rice_knowledge_irri_en_nonrestricted.jsonl \
  --output CROPSTATE_RESULTS/retrieval/belief_scenarios_calibrated.csv

PYTHONPATH=src python scripts/evaluate_retrieval.py \
  --corpus CROPSTATE_KNOWLEDGE_BASE/chunks/rice_knowledge_irri_en_nonrestricted.jsonl \
  --scenarios CROPSTATE_RESULTS/retrieval/belief_scenarios_calibrated.csv \
  --source-weight 0 --fixed-beta 0.20 \
  --output CROPSTATE_RESULTS/retrieval/retrieval_evaluation_calibrated.json

PYTHONPATH=src python -m unittest discover -s tests -v
```

## Outstanding reviewer concerns (not solved by this PR)

- Six RiceSEG macro-stage labels are morphological **weak labels**, not an
  independently double-annotated BBCH reference. Describe thresholds and review
  flags in methods. Obtain expert review especially near transitions and for
  reproductive/grain-filling classes.
- The paper describes a DCT-based pHash while the current implementation uses
  grayscale 8x8 **average hash (aHash)**. Also, hash-connected groups are formed
  *before* the split at Hamming <=10; zero leakage at that radius follows from
  the construction and cannot be advertised as an independent empirical audit.
  Amend the paper after confirming the provenance of reported audit counts.
- Existing fixed-split, three-seed results measure variability **conditional on
  the split**; they do not estimate split variance or establish field/season
  generalization. The 24-image reproductive and 10-image grain-filling test
  subsets limit statistical conclusions.
- Report the entire risk-coverage curve and choose abstention thresholds only on
  validation. Temperature scaling had negligible reported improvement in ECE.
- Compare a validation-tuned fixed soft weight and a stage-belief vs top-1
  ablation; isolate entropy and margin terms. The current PR lays groundwork but
  does not claim those experiments are complete.
- Freeze and publish configuration, annotated metadata, manifests, derived test
  logits, scenario file and aggregate outputs where redistribution rights permit.
- Audit source licenses and scope before importing external candidate datasets
  (see `docs/DATASET_CANDIDATES.md`).

## Camera-ready language guardrails

Prefer: "On weakly labeled rice images, the classifier agrees with derived
stage labels; under automatic metadata-defined relevance, adaptive soft gating
lowers stage-incompatibility while trading off some retrieval precision."

Avoid: "Agronomist-verified BBCH accuracy," "proven advisory safety," "leak-free
across fields and seasons," "significantly higher rare-stage recall," or "ranking
equivalence" until supported by independent experiments.
