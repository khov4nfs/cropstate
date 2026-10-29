# CropState independent scientific validation — execution protocol (2026-10-27)

**State: PREPARED, NOT EXPERT VALIDATED.** The repository contains no verified human annotations. This phase separates three evidence tiers:

1. **Existing weak-label mechanism benchmark**: 420 test scenarios from 199 RiceSEG test images, 173 machine-curated IRRI chunks, six methods. Reproducible, but relevance and SIRR come from the same stage metadata as reranking.
2. **Validation-only metadata ablations**: 359 eligible validation scenarios from 180 validation images (disjoint original parent-image groups); they select beta and selective-prediction threshold without using the 420-scenario test results. Still *not* independent relevance.
3. **Blinded expert relevance study**: 60 distinct test images and 564 unique pooled (image, query, chunk) judgments, copied into neutral file names. **0 completed judgments at creation**, **0 complete consensus cases**. Must obtain two real, independent agronomic reviewers to advance.

## Exact runbook

All paths are relative to the repository root. Install the original project dependencies plus `requirements-onnx.txt`. The 4 research source artifacts and pinned ONNX/tokenizer model must be present **outside Git**, under `.gitignore`. Never commit non-redistributable RiceSEG images, reviewer sheets with annotations or the machine-curated IRRI corpus without checking source terms.

```bash
export PYTHONPATH=src:.

.venv/bin/python scripts/experiments/build_belief_scenarios.py \
  --split validation --logits-index CROPSTATE_RESULTS/novelty/logits/index.json \
  --calibration CROPSTATE_RESULTS/novelty/calibration.json \
  --corpus CROPSTATE_KNOWLEDGE_BASE/chunks/rice_knowledge_irri_en_nonrestricted.jsonl \
  --output CROPSTATE_RESULTS/retrieval/independent_validation/validation_scenarios.csv

.venv/bin/python scripts/experiments/run_retrieval_ablations.py \
  --corpus CROPSTATE_KNOWLEDGE_BASE/chunks/rice_knowledge_irri_en_nonrestricted.jsonl \
  --validation-scenarios CROPSTATE_RESULTS/retrieval/independent_validation/validation_scenarios.csv \
  --test-scenarios CROPSTATE_RESULTS/retrieval/reverified/onnx-fp32-20261027/belief_scenarios_calibrated.csv \
  --logits CROPSTATE_RESULTS/novelty/logits/resnet18.npz \
  --image-manifest data/riceseg_stage_manifest.csv \
  --calibration CROPSTATE_RESULTS/novelty/calibration.json \
  --reference-evaluation CROPSTATE_RESULTS/retrieval/reverified/onnx-fp32-20261027/retrieval_evaluation.json \
  --output CROPSTATE_RESULTS/retrieval/independent_validation/ablation_results.json

.venv/bin/python scripts/prepare_retrieval_expert_review.py prepare \
  --scenarios CROPSTATE_RESULTS/retrieval/reverified/onnx-fp32-20261027/belief_scenarios_calibrated.csv \
  --evaluation CROPSTATE_RESULTS/retrieval/reverified/onnx-fp32-20261027/retrieval_evaluation.json \
  --corpus CROPSTATE_KNOWLEDGE_BASE/chunks/rice_knowledge_irri_en_nonrestricted.jsonl \
  --image-manifest data/riceseg_stage_manifest.csv --data-root CROPSTATE_DATASET \
  --n-cases 60 --seed 42 \
  --output-dir CROPSTATE_RESULTS/retrieval/independent_validation/reviewer_packet_60
```

### Annotation packet distribution and ethics

The administrator must **keep** `admin_cases.csv`, `admin_items.csv`, `STATUS.json`, and the input evaluation files private. Give Reviewer A **only** `reviewer_1.csv`, the EXIF-stripped neutral `assets/` and this rubric; Reviewer B independently receives **only** `reviewer_2.csv` and the same assets/rubric. No predicted stage, metadata stage compatibility, auto-relevance label, model method, or other reviewer's ratings are shown. Preserve two independent signed/attested submissions and evidence of expert qualifications separately; the scorer cannot authenticate a human. A third expert adjudicates conflicts without silently treating adjudication as two-person consensus.

For each neutral `CASE0001.jpg` and each associated passage, use these reviewer fields:

- `relevance_grade`: **0** no substantive answer to the query, **1** partially relevant/general evidence, **2** directly informative for the specified query; do not infer from metadata.
- `stage_inappropriate`: **yes** if advice is inappropriate, contraindicated or mistimed for the *visibly supported crop stage*; **no** otherwise. Avoid inferring chemical/dosage safety from the passage; if unclear, mark uncertain.
- `image_evidence_sufficient`: **yes** only when the visible image justifies the stage-dependent judgment; **no** if cropped panicles, occlusion, lighting or mixed-stage plants prevent judgment.
- `uncertain`: **yes** for any unclear image or passage; leave scores blank and document why in `notes`.
- Do not use filenames, weak BBCH labels, season/country cues or the paper's expected stage distribution to decide correctness.

For genuine independent scientific validation, additional domain experts must adjudicate BBCH on corresponding image identities. The pre-existing **120-image BBCH packet is a separate unannotated task**; its sample does not automatically cover these 60 retrieval cases. Do **not** claim the retrieval scores validate vision classification or vice versa.

After both reviewers submit the completed files:

```bash
.venv/bin/python scripts/prepare_retrieval_expert_review.py score \
  --admin-dir CROPSTATE_RESULTS/retrieval/independent_validation/reviewer_packet_60 \
  --reviewer-1 /private/independent_reviewer_1.csv \
  --reviewer-2 /private/independent_reviewer_2.csv \
  --output CROPSTATE_RESULTS/retrieval/independent_validation/expert_scores.json
```

### Interpretation and stopping conditions

The scorer **never imputes missing labels** and excludes a case unless *every* item in the union of all six methods' top-five lists has identical, non-uncertain independent judgments. Stage-inappropriate comparisons use the **same subset** of cases where every pooled document has sufficient visible stage evidence. The scorer reports consensus-only precision@5, pooled-relative recall@5, pooled nDCG@5, and stage-inappropriate@5. Pooled recall is *not* retrieval recall against the entire corpus. Disagreements, uncertain judgments, and images with insufficient evidence must be disclosed with exact counts; compare methods with paired image-level estimates only once enough independently labeled cases exist. The 60-image subset is weak-stage balanced and deliberately nonrepresentative, so no population-level or field/season generalization is justified.

## Validation ablation design and constraints

The standalone ONNX FP32 ablation uses the same pinned multilingual MiniLM encoder and topic-exact candidate set as the controlled benchmark. The test P, B2 and B3=0.20 aggregates must match that benchmark to within 1e-9 or the run fails.

**Beta selection**: predeclared grid `{0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35}`; select on *validation scenarios only* by maximizing `nDCG@5 - metadata-SIRR@5` (ties choose lower beta), then freeze before test. This equal-weight utility is a diagnostic choice, not agronomic risk preference. **Confidence ablations**: entropy-only, margin-only, calibrated top-1 confidence, one-hot top-1 belief, raw uncalibrated full belief, and original calibrated full belief with 0.6 entropy / 0.4 margin. Held-out comparisons use 5,000 paired image-cluster bootstrap and sign-flip permutations with Holm adjustment over six exploratory tests. A selection of these ablations was hypothesized by reviewers; treat all new result claims as exploratory unless independently replicated.

**Abstention selection**: validation-only threshold chosen to approximate **70% coverage**, using entropy/margin combined confidence on validation logits. Apply the frozen cutoff to all 263 original test images; report the full risk-coverage curve and realized coverage/risk *against weak labels* only.

## Phase-gate before camera-ready scientific claims

- Two identifiable independent agronomic experts actually submit completed ratings (not LLM-generated labels).
- Independent BBCH reference is obtained and matched to original held-out image IDs where accuracy is claimed.
- Agreement/disagreement, uncertain images, image-source rights, and independent retrieval relevance are reported; no undocumented missing-label deletion.
- Sensitivity tests and limitations distinguish metadata-mechanism consistency from real agronomic effectiveness.
- Final IEEE layout and scholarly proofread pass before uploading the camera-ready PDF.

Until then, the current camera-ready source may report **the validated automatic ablation findings with limitations**, but not expert relevance, causal safety or new real-world BBCH accuracy.
