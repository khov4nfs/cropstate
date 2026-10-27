# CropState-Vision RIVF'26: camera-ready data provenance and status

**Cutoff**: 2026-10-27. Camera-ready due **2026-11-25**. IEEE format,
6 standard pages. The main \`paper/cropstate_paper.tex\` currently uses
\`llncs\`; it must be converted/compiled/visually reviewed under the actual
RIVF/IEEE conference template before submission.

## IEEE formatting check (non-final builds)

The first IEEEtran draft (`cropstate_camera_ready_ieee.tex`) compiled with
XeLaTeX but produced **8 pages** (GitHub Actions run 40485805734).
The condensed draft (`cropstate_camera_ready_ieee_compact.tex`) remained
8 pages; a more compact, reviewer-aware IEEE draft
(`paper/cropstate_camera_ready_ieee_short.tex`) **compiled successfully at
6 pages** (GitHub Actions run 40486682537). It retains the central vision,
retrieval, significance and error-stratified findings while shortening
redundant related-work/discussion text. Later title-alignment changes with
the acceptance letter must be recompiled and checked again. This is only a
formatting milestone: historical results have NOT been independently rerun,
and final visual/IEEE compliance inspection remains mandatory.

## Important: distinguish three levels of evidence

1. **Legacy manuscript numbers** (below): previously reported in
   \`paper/cropstate_paper.tex\`, **NOT recomputed with revised retrieval code**.
2. **Automated checks**: GitHub Actions regression tests and blank, stratified
   human review task-generation; these checks do not verify scientific results.
3. **Independent agronomy evidence**: expert-verified image BBCH labels and
   expert-judged relevance of knowledge chunks are **NOT yet available**.

Do not copy metrics from level (1) into a table representing level (2) or (3).
In particular, the corrected calibrated-belief + confidence pipeline, source
authority treatment, and retrieval candidate universe can change the result.
No table in this document may be described as a successful rerun.

## Historical / reported (not independently reverified) numbers

Vision: 2,900 images, fixed 2,400/237/263 group partition,
three training seeds; comparison against **heuristically derived stages**, not
agronomist-verified reference BBCH labels.

| Objective | Accuracy | Macro-F1 | ECE |
| --- | ---: | ---: | ---: |
| CE | 0.800 | 0.794 | 0.062 |
| Ordinal | 0.801 | 0.796 | 0.064 |
| Focal | 0.767 | 0.757 | 0.179 |

Reported retrieval (420 metadata-derived scenarios; 199 contributing images):

| Method | P@5 | R@5 | nDCG@5 | SIRR@5 |
| --- | ---: | ---: | ---: | ---: |
| B0 ungated | 0.225 | 0.118 | 0.165 | 0.136 |
| B1 query expansion | 0.370 | 0.192 | 0.335 | 0.159 |
| B2 hard filter | 0.266 | 0.180 | 0.259 | 0.002 |
| B3 fixed-soft | 0.260 | 0.163 | 0.265 | 0.073 |
| P adaptive | 0.300 | 0.198 | 0.314 | 0.038 |
| Oracle | 0.327 | 0.222 | 0.349 | 0.022 |

Reported P minus B1 clustered bootstrap:
nDCG -0.021 (95% CI [-0.065, +0.021]); precision -0.070
([-0.111, -0.027]); SIRR -0.121 ([-0.152, -0.090]).
These historical tests do **not** constitute equivalence testing.

## Verification performed in this PR

- Stage-belief scenario builder now consumes validation-temperature-calibrated
  probabilities for retrieval belief and entropy/margin routing confidence.
- Benchmark script now permits authority-free comparison, fixed-beta selection,
  and explicit inclusion/exclusion of general-topic fallback.
- Full reproduction orchestrator:
  \`scripts/reproduce_retrieval_benchmark.py\`, which **fails closed** if
  the original corpus, logits/calibration, expected 420 scenarios or 199 image
  identities are missing; it hashes input/output artifacts and generates a
  new Markdown results table *only after successful rerun*.
- Blinded expert-review preparation: \`scripts/prepare_bbch_expert_review.py\`,
  \`scripts/score_bbch_expert_review.py\`. A GitHub Actions run created **120
  unannotated tasks (20 per stage)** and reported **zero paired reviews and zero
  expert consensus labels**, exactly as intended. Synthetic unit-test labels
  are NOT expert evidence.
- Manuscript method corrections: pHash/DCT was corrected to grayscale
  **average-hash (aHash)** and hash-grouped split construction was made explicit;
  the false implication that RiceSEG supplies BBCH stage labels was corrected
  in Data Availability. All historical tables remain untouched.

## Why the 420-scenario benchmark has not yet been rerun

The repository has \`data/retrieval_scenarios.csv\` with only two illustrative
scenarios, not the 420-scenario source CSV. The old evaluation code refers to
external inputs such as:

- \`CROPSTATE_RESULTS/novelty/logits/index.json\` and the indexed model NPZ
  containing per-image validation and test logits;
- \`CROPSTATE_RESULTS/novelty/calibration.json\`, fitted using validation data;
- \`CROPSTATE_KNOWLEDGE_BASE/chunks/rice_knowledge_irri_en_nonrestricted.jsonl\`;
- the canonical fixed manifest and full knowledge corpus (plus embedding model).

None of the above exact artifact files is committed in GitHub. On 2026-10-27,
the connected Google Drive search showed named results and knowledge folders
but no accessible matching files inside the expected retrieval/logits/chunks
directories. Their prior existence is not evidence that the full benchmark
executed again today. No experimental results have been fabricated.

## Run the full benchmark when artifacts are mounted

\`\`\`bash
python -m pip install -r requirements.txt
PYTHONPATH=src python scripts/reproduce_retrieval_benchmark.py \
  --corpus CROPSTATE_KNOWLEDGE_BASE/chunks/rice_knowledge_irri_en_nonrestricted.jsonl \
  --logits-index CROPSTATE_RESULTS/novelty/logits/index.json \
  --calibration CROPSTATE_RESULTS/novelty/calibration.json \
  --output-dir CROPSTATE_RESULTS/retrieval/reverified
\`\`\`

Outputs: \`retrieval_evaluation.json\`, \`retrieval_significance.json\`,
\`CAMERA_READY_RETRIEVAL_TABLE.md\`, \`PROVENANCE.json\`, and regenerated
\`belief_scenarios_calibrated.csv\`. Each MUST be quality-reviewed before copying
numbers into the manuscript. If scenario counts change, investigate the change;
do not silently force the old 420/199 denominators.

## Next required evidence

1. Supply image assets with neutral filenames to the two reviewer sheets, and
   obtain annotations from real agricultural specialists. Run
   \`score_bbch_expert_review.py\`, record disagreements and handle adjudication.
2. Build independent query-to-chunk relevance judgments to evaluate actual
   agronomic usefulness. Auto-derived compatibility is not an independent gold set.
3. Select fixed-beta and abstention threshold on validation before test evaluation.
4. Run the updated retrieval benchmark, paired image-clustered CI, and ablations.
5. Migrate to the correct IEEE conference class, compile, inspect page count and
   verify references and all tables/figures before uploading to EDAS.

## Conservative camera-ready interpretation pending new evidence

The contributions are (i) a fixed-split evaluation of six weakly supervised rice
macro-stage labels, (ii) uncertainty-aware ranking using a full stage belief, and
(iii) preliminary metadata-defined stage incompatibility analysis. They do **not**
establish field-level generalization, independently verified BBCH accuracy,
equivalent ranking performance, or independently assessed agronomic safety.
