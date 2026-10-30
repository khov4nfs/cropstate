# CropState-Vision — Camera-ready experimental verification, updated 2026-10-29

> **SUPERSEDED NUMERIC TABLE (2026-10-29):** The older controlled retrieval figures below (nDCG 0.667 Adaptive and image-level CIs) came from Vietnamese lexical queries against an English IRRI corpus and resampled 199 image crops rather than their 63 original parent photographs. Do **not** use those figures in the final paper. The authoritative corrected method, metrics, validation-only ablations, source hashes and six-page PDF QA are documented in [CAMERA_READY_PROTOCOL_CORRECTION_20261029.md](CAMERA_READY_PROTOCOL_CORRECTION_20261029.md). Final PDF sha256: `dfc018474647242e675c864c8bb5c53cee0ef557c655fd6e2facb1e0daa1b3cd` (6 A4 pages, PDF 1.6). This archival section remains only to show scientific provenance.


**Status: controlled retrieval reproduced; validation-only metadata ablations complete; A4/PDF 1.6 camera-ready compiled successfully at six pages.** Canonical source: `paper/cropstate_camera_ready_ieee_short.tex`; compiled PDF: `paper/cropstate_camera_ready_ieee_short.pdf`. The study deliberately limits its claims to weak labels and metadata-based retrieval, without claiming independent expert validation. Legacy tables in `paper/cropstate_paper.tex` describe a different protocol and must not be combined with the controlled results.

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
- The original six-method table uses untuned fixed beta=0.20; subsequent validation-only tuning and abstention selection are reported separately below and still use weak/metadata labels.
- The same source-image clusters recur across topics; confidence intervals and paired tests account for image-level dependence but not field/season generalization.
- The revised XeLaTeX manuscript was independently built in a Dockerized TeX Live 2025 environment on 2026-10-29: six A4 pages, PDF 1.6, embedded fonts, no overfull lines, unresolved references, missing glyphs or LaTeX errors. The official RIVF'26 author-facing limit is six A4 pages.

Vision results in the original manuscript remain historical, not rerun this turn. Do not label them newly reverified.

## Validation-only ablations: evidence and inference limits (2026-10-27)

**Not an expert-judged result.** The automated ablations are included only as metadata-defined experiments. `docs/INDEPENDENT_VALIDATION_PROTOCOL.md` preserves their reproducibility protocol; it does not supply independent agronomy truth.

- 359 automatically labeled **validation** scenarios from 180 distinct validation images; held-out **test** remains the same 420 scenarios from 199 images. Inputs confirmed disjoint at original parent-image group and image ID levels; no test metric is used for hyperparameter selection.
- Fixed beta grid 0.05–0.35, step 0.05: validation-only selection by max(nDCG@5 − metadata-SIRR@5) chooses **beta=0.35**. Equal metric weights are a predeclared diagnostic, NOT a domain-expert risk utility.
- Frozen beta=0.35: test nDCG@5 **0.6824**, metadata-SIRR@5 **0.1000**; original adaptive combined: **0.6668**, **0.1243**. Difference (fixed minus adaptive) clustered by 199 test images: nDCG +0.0156, 95% CI [+0.0074,+0.0248], Holm-adjusted exploratory p=0.0012; SIRR −0.0243, CI [−0.0338,−0.0161], adjusted p=0.0012. **Thus the original adaptive is NOT best under the validation-tuned metadata-defined objective.**
- Calibrated top-1 confidence: test nDCG/SIRR **0.6753/0.1100**; one-hot top-1 belief **0.6781/0.0986**; entropy-only **0.6683/0.1195**; margin-only **0.6650/0.1252**; raw belief **0.6630/0.1295**. These are exploratory ablations, not independent expert labels, and do not establish safety.
- Selective prediction threshold **0.5245412541** chosen **only on 237 validation logits** to target 70% coverage. On 263 held-out test images: coverage **0.7186** (189 retained), selective risk against **weak labels** **0.1217**, versus full risk **0.2053**. Full curve saved externally.
- New run cross-check asserts exact agreement to within 1e−9 for original adaptive, hard-filter and fixed beta=0.20 controlled benchmark.
- No independently verified BBCH stage references or agronomist-scored recommendation relevance are available in this study; this is a stated limit, not a pending camera-ready deliverable.

New ignored artifact hashes:
- Validation scenarios CSV SHA-256: `cca6ade6af69060c4220091057709e1110cef7b76011fec01dcd52811bdc9858`
- Ablation JSON SHA-256: `a0d1277b0998f496321b10ed094adbe7009eb9a65f343eb4b1169cfb84e8c349`
- Admin case manifest SHA-256: `883baede9775b7eee4307b4549165769ab775ed030dce891d57a07b6ab052dea`
- Admin pooled-item manifest SHA-256: `ad7746315357d4a79f61ece83cd3d4322a120f0646e2446391afe25c63390053`

**Camera-ready editorial decision:** Retain the six-method controlled table, label B3 as `fixed beta=0.20 (untuned)`, and report validation-selected fixed-beta experiments separately. Do not claim adaptive superiority over tuned fixed gating, expert-verified BBCH accuracy, agronomic utility or safety. Unavailable external validations are research limitations, not incomplete camera-ready tasks.

## Final build QA — 2026-10-29

- Source: `paper/cropstate_camera_ready_ieee_short.tex` using `IEEEtran[conference,a4paper]`, XeLaTeX, and xdvipdfmx PDF 1.6.
- Output: `paper/cropstate_camera_ready_ieee_short.pdf`, **6 pages**, A4 595.28 × 841.89 pt, **138,865 bytes**, SHA-256 `a35dc185b473d4454fb5d976f25ddf505f60b0c5f960e431dad5818ae6bf8d9d`.
- `pdfinfo`: A4, PDF version 1.6, six pages. `pdffonts`: all fonts embedded. Build log: no overfull hbox, undefined references, missing glyphs or undefined control sequences. The minor underfull hbox warnings do not clip content.
- Visual inspection of rendered pages 1, 3, 5 and 6: titles, tables, limitations, conclusion and references are inside page bounds.
- This verifies the build and layout, **not** independent scientific claims, IEEE PDF eXpress acceptance, EDAS submission, copyright transfer or conference registration.
