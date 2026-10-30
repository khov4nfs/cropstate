# Camera-ready correction — English queries and original-parent inference (29 Oct 2026)

**AUTHORITATIVE protocol for the revised paper; supersedes retrieval numeric results dated 27 Oct 2026.** All relevance and SIRR are automatically defined by compatibility metadata; no agronomic expert scores.

## Corrections
1. Corpus: 173 English IRRI chunks. Both BM25 and multilingual MiniLM queries now use **English**; old Vietnamese lexical queries had zero matching tokens in these chunks.
2. Inference: 420 topic scenarios over 199 crops from **63 parent images**. Parent-image resampling, not independent-crop resampling.
3. Hard filter: 42/420 cases underfilled Top-5 (2 empty), because stage filtering can remove all candidates. SIRR@5 divides by 5 even when fewer results are returned.
4. Tuning: fixed beta selected on 359 validation scenarios / 180 crops / 59 parent groups; test frozen.
5. Vision: checked-in config has balanced_sampler=false with class-weighted loss. Focal formula uses the weighted CE inside exp(-CE), unlike standard unweighted focal.

## Six-method test results
| Method | P@5 | R@5 | nDCG@5 | metadata SIRR@5 ↓ | full Top-5 |
|---|---:|---:|---:|---:|---:|
| B0 ungated | 0.277 | 0.235 | 0.283 | 0.218 | 100% |
| B1 expanded | 0.373 | 0.241 | 0.343 | 0.301 | 100% |
| B2 hard filter | 0.350 | 0.340 | 0.374 | 0.011 | 90% |
| B3 fixed β=.20 | 0.409 | 0.317 | 0.465 | 0.175 | 100% |
| P adaptive | 0.485 | 0.380 | 0.545 | 0.127 | 100% |
| Weak-stage reference oracle | 0.614 | 0.453 | 0.661 | 0.070 | 100% |

## Parent-group paired bootstrap/sign-flip (5,000 draws, Holm across 12 tests)
| Comparison | Metric | Delta | 95% CI | adjusted p |
|---|---|---:|---|---:|
| P−B0_ungated | ndcg_at_k | +0.262 | [+0.219, +0.308] | 0.0024 |
| P−B0_ungated | recall_at_k | +0.146 | [+0.109, +0.183] | 0.0024 |
| P−B0_ungated | precision_at_k | +0.208 | [+0.172, +0.244] | 0.0024 |
| P−B0_ungated | sirr_at_k | -0.091 | [-0.118, -0.064] | 0.0024 |
| P−B1_query_expansion | ndcg_at_k | +0.202 | [+0.165, +0.243] | 0.0024 |
| P−B1_query_expansion | recall_at_k | +0.139 | [+0.111, +0.168] | 0.0024 |
| P−B1_query_expansion | precision_at_k | +0.112 | [+0.081, +0.144] | 0.0024 |
| P−B1_query_expansion | sirr_at_k | -0.174 | [-0.195, -0.152] | 0.0024 |
| P−B3_fixed_soft | ndcg_at_k | +0.081 | [+0.057, +0.105] | 0.0024 |
| P−B3_fixed_soft | recall_at_k | +0.063 | [+0.045, +0.082] | 0.0024 |
| P−B3_fixed_soft | precision_at_k | +0.076 | [+0.056, +0.095] | 0.0024 |
| P−B3_fixed_soft | sirr_at_k | -0.048 | [-0.061, -0.036] | 0.0024 |

## Validation-selected ablations
- Selected fixed beta: 0.35.
- Adaptive combined nDCG/SIRR: 0.545 / 0.127.
- Fixed validation selected nDCG/SIRR: 0.612 / 0.101.
- Top-1 belief nDCG/SIRR: 0.617 / 0.098.
- Parent-cluster fixed−adaptive nDCG delta: +0.066, CI [+0.050, +0.083], exploratory Holm p=0.0012.

## Reproduction
Run these from repository root with the original local research artifacts and pinned ONNX model:

    PYTHONPATH=src:. .venv/bin/python scripts/reproduce_retrieval_benchmark.py --corpus CROPSTATE_KNOWLEDGE_BASE/chunks/rice_knowledge_irri_en_nonrestricted.jsonl --logits-index CROPSTATE_RESULTS/novelty/logits/index.json --calibration CROPSTATE_RESULTS/novelty/calibration.json --image-manifest data/riceseg_stage_manifest.csv --query-language en --embedding-backend onnx --output-dir CROPSTATE_RESULTS/retrieval/reverified/english-parent-20261029
    PYTHONPATH=src:. .venv/bin/python scripts/experiments/run_retrieval_ablations.py --corpus CROPSTATE_KNOWLEDGE_BASE/chunks/rice_knowledge_irri_en_nonrestricted.jsonl --validation-scenarios CROPSTATE_RESULTS/retrieval/independent_validation/validation_scenarios.csv --test-scenarios CROPSTATE_RESULTS/retrieval/reverified/english-parent-20261029/belief_scenarios_calibrated.csv --logits CROPSTATE_RESULTS/novelty/logits/resnet18.npz --image-manifest data/riceseg_stage_manifest.csv --calibration CROPSTATE_RESULTS/novelty/calibration.json --reference-evaluation CROPSTATE_RESULTS/retrieval/reverified/english-parent-20261029/retrieval_evaluation.json --output CROPSTATE_RESULTS/retrieval/reverified/english-parent-20261029/ablation_results.json

## Integrity
- Evaluation sha256: 76e63ea17b3467b53cf92a291eb6fd8e2c2bf89c9325679299d2a6315c50bc84.
- Parent significance sha256: 68ac7e19744990f8ce69821d5c2521854da37be1ca6c4a8658b271bdb2ea453f.
- Ablations sha256: ab21a5d91823816e4455abab17ebf05cd5eda24638d07e6eedc85e7b4aa973c4.
- Image manifest sha256: 775ab4b7e56a12d9ac5a0d0a18cf53cb0523b26a47646036f9395f498886c7cf.
- XeLaTeX camera-ready PDF sha256: dfc018474647242e675c864c8bb5c53cee0ef557c655fd6e2facb1e0daa1b3cd (6 A4 pages, PDF 1.6).
- Regression/fixture tests: 17 passed; reader-side paper comparisons refer only to the English-query, parent-group protocol.

Limitations: weak morphology-derived BBCH-inspired pseudo-stage labels; metadata-defined relevance and mismatch; no independent field-level transfer, agronomic effectiveness or safety claim.
