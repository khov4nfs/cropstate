# CropState-Vision — Camera-ready experimental verification, 2026-10-27

**Status: controlled-protocol retrieval benchmark recomputed, not expert judged.** The canonical camera-ready draft is paper/cropstate_camera_ready_ieee_short.tex. Legacy tables remain in paper/cropstate_paper.tex for comparison; do not combine them.

## Verification and provenance

- Recovered ResNet-18 stage logits from SourceNerve ignored workspace artifacts; validation count 237, test count 263, six classes, no ID overlap.
- Validation-fitted temperature: 0.8746500341029899. Stage belief is temperature-calibrated, and routing confidence is 0.6*(1-normalized entropy) + 0.4*(top1-top2 margin).
- Corpus: 173 nonrestricted IRRI Rice Knowledge Bank chunks, machine curated; research mode only, without expert sign-off.
- Recomputed exactly 420 scenario rows from 199 source test images (5 eligible topics), using weak derived stages and metadata-derived relevance.
- Stage-free BM25 and dense MiniLM embeddings combined with RRF, candidate depth 50, topic-exact retrieval (no unjudged general-care fallback), top-k=5.
- Rerank authority score disabled (source_weight=0.0); fixed-soft beta=0.20, adaptive beta range unchanged. No assertion of validation-tuned beta.
- The original PyTorch encoder was blocked by the SourceNerve sandbox (/dev/urandom inaccessible). The rerun uses an ONNX FP32 export of the **same** multilingual MiniLM-L12-v2 model, with max length 128, attention-masked mean pooling and L2 normalization.
- In a separate controlled historical-settings replay (raw probabilities, top-1 confidence, authority weight 0.10, general-care fallback enabled) **all 24 stored aggregate scores across six methods exactly matched** the original result file (absolute difference <1e-11). This is a model-backend cross-check, not a justification for comparing legacy and corrected scores as the same experimental protocol.

## Retrieval results — recomputed controlled protocol

| Method | P@5 | R@5 | nDCG@5 | SIRR@5 (lower) |
|---|---:|---:|---:|---:|
| B0 ungated | 0.521 | 0.256 | 0.500 | 0.306 |
| B1 query expansion | 0.534 | 0.316 | 0.510 | 0.285 |
| B2 hard filter | 0.575 | 0.360 | 0.611 | 0.002 |
| B3 fixed-soft | 0.601 | 0.349 | 0.617 | 0.182 |
| P adaptive | 0.640 | 0.386 | 0.667 | 0.124 |
| Oracle | 0.688 | 0.433 | 0.723 | 0.080 |

## P minus B1: 95% clustered bootstrap CI and Holm-adjusted p-values

Statistical analysis: 5,000 bootstrap resamples of the 199 source-image clusters for percentile 95% confidence intervals; 5,000 paired image-cluster sign-flip permutations (+1 Monte-Carlo correction) for p-values; Holm correction across 12 comparisons (P versus B0, B1, B3 on 4 metrics).

- ndcg_at_k: delta=+0.156, CI [+0.107, +0.207], adjusted p=0.0024.
- recall_at_k: delta=+0.070, CI [+0.024, +0.119], adjusted p=0.0062.
- precision_at_k: delta=+0.106, CI [+0.063, +0.151], adjusted p=0.0024.
- sirr_at_k: delta=-0.161, CI [-0.200, -0.123], adjusted p=0.0024.

B1 SIRR decreases by 56.4% relative under the corrected protocol; B2 still has the smallest SIRR.

Image-error strata: 330 correct-stage, 84 adjacent-stage, and 6 nonadjacent-stage scenarios. In the nonadjacent subgroup, P SIRR is 0.567 and B2 SIRR is 0.100. This rare subgroup is too small to generalize.

## SHA-256 digests (local SourceNerve workspace artifacts)

- Input logits NPZ: 17ee5f9273e3ef87031393c3f60c6d2996793e35e7d9ff4f3fbb07940415c549 (CROPSTATE_RESULTS/novelty/logits/resnet18.npz)
- Input index: c778b7969a589a3ed08791ed7188e1f35813cbf33e0131295dda4930009cccf8 (CROPSTATE_RESULTS/novelty/logits/index.json)
- Input calibration: e6ad9e57ec1c67699435dd57eda324457c1ae2536a9f209aa30800def9d9bc6a (CROPSTATE_RESULTS/novelty/calibration.json)
- Input knowledge corpus: 2d0a9b4f47a70103f52ef453849b265d9b77ef446b81a9fa69b324fd552f6497 (CROPSTATE_KNOWLEDGE_BASE/chunks/rice_knowledge_irri_en_nonrestricted.jsonl)
- Generated calibrated scenario CSV: bf45d2d686a421ba95c28337d212464524fded4428a42a2b12ad286e5c0125ed (CROPSTATE_RESULTS/retrieval/reverified/onnx-fp32-20261027/belief_scenarios_calibrated.csv)
- Generated retrieval evaluation: fc3b9be4a5c2daba7d2055c7032fe02c59ef53c263e69d3ab8bf91a328aa9b80 (CROPSTATE_RESULTS/retrieval/reverified/onnx-fp32-20261027/retrieval_evaluation.json)
- Generated significance evaluation: 4e921bc7cfef92c67ad013be4e7b81f6937df374381c88cde37f16c1f74a70f1 (CROPSTATE_RESULTS/retrieval/reverified/onnx-fp32-20261027/retrieval_significance.json)
- Generated run manifest: 8145593c7c83397d96aeccca07b2bb6665b4ff5664587f0671d1b5ab5856406f (CROPSTATE_RESULTS/retrieval/reverified/onnx-fp32-20261027/PROVENANCE.json)
- ONNX FP32 model weights: 10f7a088420252b26caf819236ca2c9d2987afd0fc06fec7553b542a5655a05a
- Tokenizer: 2c3387be76557bd40970cec13153b3bbf80407865484b209e655e5e4729076b8

## Reproduction and dependency notes

Install optional ONNX Runtime and Hugging Face tokenizers into the project virtual environment. The ONNX FP32 weights and tokenizer must be a pinned snapshot of sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2. Run from the CropState repo root using project-relative input paths:

    PYTHONPATH=src .venv/bin/python scripts/reproduce_retrieval_benchmark.py \
      --corpus CROPSTATE_KNOWLEDGE_BASE/chunks/rice_knowledge_irri_en_nonrestricted.jsonl \
      --logits-index CROPSTATE_RESULTS/novelty/logits/index.json \
      --calibration CROPSTATE_RESULTS/novelty/calibration.json \
      --embedding-backend onnx \
      --output-dir CROPSTATE_RESULTS/retrieval/reverified/onnx-fp32-20261027

The local run directory contains calibrated scenarios, per-scenario per-method results (2,520 rows), clustered significance output, camera-ready summary, and SHA-256 manifest. The data are intentionally git-ignored and have not been publicly committed.

## Strong limitations that remain

- RiceSEG did not supply native expert BBCH labels; morphological weak labels do not prove field-level accuracy. The 120-image blind reviewer packet has no completed human annotations.
- The retrieved relevance labels are generated from the same compatibility metadata consumed by the reranker; these metrics test internal metadata consistency, **not** agronomist-judged relevance or safety.
- The corrected candidate universe and source authority differ from the historical protocol, so magnitudes cannot be interpreted as a purely algorithmic gain.
- No hyperparameter tuning on validation scenarios is included for fixed beta or selective abstention thresholds.
- The same source-image clusters recur across topics; confidence intervals and paired tests account for image-level dependence but not field/season generalization.
- The IEEE draft must be compiled, page-counted, checked for layout and reviewed before conference upload. The 6-page constraint remains in force.

Vision results in the original manuscript remain historical, not rerun this turn. Do not label them newly reverified.
