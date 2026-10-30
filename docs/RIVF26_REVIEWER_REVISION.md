# RIVF'26 — Official reviewer feedback and reviewer-driven revision register

**Scope:** Only the four official EDAS peer-review reports for paper **#1571322865** (2026-10-25) and the verified revisions that directly address their comments. Independent code-audit discoveries belong in the separate camera-ready protocol record, not this reviewer checklist.

**Evidence boundary:** This register contains only four official EDAS manuscript reviews and verifiable code–paper findings. Requirements that cannot be addressed with the available dataset are recorded as study **LIMITATIONS**, not unfinished validation tasks. The reviewers did not supply independent agronomic ground truth.

## External review scores (EDAS, author-provided)

| EDAS reviewer | Novelty / 5 | Technical rigor / 5 | Presentation / 5 | Relevance / 5 |
|---|---:|---:|---:|---:|
| Reviewer 1 | 4 | 4 | 5 | 5 |
| Reviewer 2 | 2 | 2 | 3 | 4 |
| Reviewer 3 | 3 | 3 | 4 | 4 |
| Reviewer 4 | 2 | 3 | 4 | 4 |

**Reviewer 1:** Praised the near-duplicate audit, multi-seed analysis, ordinal loss, soft gating, statistical tests and explicit limitations. Requested broader architecture ablation, BBCH expert checking, field/season independence, independent retrieval relevance, caution on the 10-image grain-filling class and validation-tuned abstention.

**Reviewer 2:** Criticized (R2.1) unverified heuristic stage labels and missing operational labeling evidence; (R2.2) metadata-circular relevance and SIRR mislabeled as safety; (R2.3) unsupported ranking-equivalence language and missing CIs/adjusted p-values; (R2.4) absence of a validation-tuned fixed-soft baseline and stage-belief/confidence ablations; (R2.5) reproductive n=24, fixed-split versus partition variance, post-hoc abstention and limited risk–coverage reporting; (R2.6) incomplete reproducibility details, source-group definitions, omission of 64 test images from retrieval and limits of pHash claims.

**Reviewer 3:** Recognized the application and experiments while requesting clearer novelty, independent BBCH/relevance labels, field/season external validation and a validation-fitted abstention threshold.

**Reviewer 4:** Requested actual agronomist-reviewed stage and retrieval data; distinct measures of relevance versus advice applicability; external fields/seasons/cultivars/devices; and statistical evidence plus honest explanation of the ranking–incompatibility trade-off.

## Reviewer-driven revisions and evidence

| ID | EDAS source | Implementation / reproducibility evidence | Final research disposition |
|---|---|---|---|
| CP-01 | R1–R4: BBCH reference labels | `scripts/experiments/relabel_riceseg_bbch.py` derives weak labels from mask morphology. | **LIMITATION:** all classification metrics reflect morphological weak labels; expert-verified BBCH accuracy is not claimed. No additional expert experiment is presented as pending. |
| CP-02 | R1–R4: retrieval circularity | `scripts/experiments/build_belief_scenarios.py` derives relevance from stage metadata also used by reranking. | **LIMITATION:** scores reflect metadata-consistency only; neither independent recommendation usefulness nor safety is claimed. |
| CP-03 | R2/R4: adaptive versus fixed | Corrected English-query test protocol: β=0.35 selected on 359 validation scenarios; held-out fixed nDCG 0.612 / SIRR 0.101 versus adaptive 0.545 / 0.127. Top-1 belief nDCG 0.617. | **VERIFIED AND REVISED:** β=0.20 was untuned; tuned fixed and top-1 belief are superior to combined adaptive on metadata-defined scores. Original Vietnamese-query metrics are superseded. |
| CP-04 | R1–R3: abstention selection | Validation-only confidence cutoff 0.524541 retains 189/263 test images, coverage 0.7186, weak-label risk 0.1217. | **CORRECTED IN CAMERA-READY SOURCE:** the 0.54-coverage point is explicitly illustrative; the validation-selected threshold 0.5245 and its held-out outcome are reported separately. |
| CP-05 | R2: 64 omitted test images | Scenario builder requires >=3 relevant and >=3 incompatible chunks per (stage, topic); only 199/263 test images have at least one qualifying topic, yielding 420 scenarios. | **CORRECTED IN CAMERA-READY SOURCE:** explicitly explain 64 exclusions as corpus/scenario eligibility, not an image-model failure. |
| CP-06 | R1–R4: pHash and field leakage | src/cropstate/splits.py and scripts/experiments/leakage_audit.py implement grayscale 8x8 **aHash** Hamming clustering before splitting. | **CORRECTED:** short paper now says aHash, acknowledges zero cross-split matches is partly by construction and does not prove field/season independence. |
| CP-07 | R2–R4: rare-stage reliability | Same 24 reproductive and 10 grain-filling test images across seeded runs under one fixed group split. | **CLAIM NARROWED:** identical recall across seeds is descriptive on the small shared subset, not proof the ordinal loss causally stabilizes recall across fields. |
| CP-08 | R2: configuration fidelity | The evaluator uses `source_weight=0`, while the reusable library default is 0.10. | **EXPLAINED:** paper explicitly identifies the zero-weight evaluated configuration, not arbitrary direct API calls. |
| CP-09 | R2: reproducibility | `scripts/reproduce_retrieval_benchmark.py`, ONNX checksums and 420-scenario aggregate metrics are recorded in `paper/CAMERA_READY_RESULTS_STATUS.md`. | **VERIFIED RETRIEVAL; VISION SCOPE LIMIT:** the retrieval rerun is reproducible; vision scores are historical and not newly retrained. Source assets remain subject to access terms. |
| CP-11 | R1–R4: generalization and novelty | ResNet-18 CE/ordinal/focal objectives on one fixed split; no external acquisition-domain benchmark. | **LIMITATION:** results do not demonstrate multiple-backbone or new-field/season/cultivar/device generalization. Novelty is confined to the evaluated uncertainty-aware retrieval comparison. |

## Strict provenance and evidence boundaries

- **Keep:** real EDAS reviewers' numeric scores and substantive criticism, code/paper cross-checks, verified corrections and clearly bounded study limitations.
- **Completed versus limited:** completed items have code, experiment and paper evidence; unavailable external expert labels, architecture comparisons or domain-shift data are classified as study limits, not active tasks.
- **Do not invent expert evidence:** human kappa, expert-reference BBCH accuracy and agronomist-judged relevance are not reported because no independently verified labels are available.
- **Publication boundary:** source edits do not themselves establish a freshly compiled PDF or final conference submission.

**Original reviewer source:** author-supplied RIVF'26 EDAS acceptance/decision letter dated 2026-10-25 for paper #1571322865, sections Review 1 through Review 4 (retained privately; not the AI agronomy report).

**Comparison targets:** paper/cropstate_camera_ready_ieee_short.tex, docs/INDEPENDENT_VALIDATION_PROTOCOL.md, paper/CAMERA_READY_RESULTS_STATUS.md, scripts/evaluate_retrieval.py, scripts/experiments/build_belief_scenarios.py, scripts/experiments/run_retrieval_ablations.py and src/cropstate/retrieval.py.
