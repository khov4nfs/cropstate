# RIVF'26 — Official reviewer feedback and code–paper discrepancy register

**Scope:** Only (1) the four peer-review reports in the RIVF'26 EDAS decision for paper **#1571322865**, dated **2026-10-25**, and (2) implementation–paper discrepancies verified in repository code and reproducibility artifacts.

**Evidence boundary:** This register contains only the four EDAS manuscript reviewers and verified code–paper findings. Their reviews do not replace the two required independent human agronomy annotations. No completed agronomist A/B forms are on record; GitHub PR #1 has no submitted review comments.

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

## Verified code–paper discrepancies and reviewer-driven status

| ID | EDAS source | Implementation / reproducibility evidence | Paper status / necessary action |
|---|---|---|---|
| CP-01 | R1–R4: BBCH reference labels | scripts/experiments/relabel_riceseg_bbch.py generates weak labels from mask morphology; scripts/score_bbch_expert_review.py requires actual human annotations. | **OPEN:** paper acknowledges unverified BBCH; no human two-digit BBCH reference or expert accuracy. Do not insert non-human labels. |
| CP-02 | R1–R4: retrieval circularity | scripts/experiments/build_belief_scenarios.py derives relevance and incompatibility from chunk stage_compatibility, reused by src/cropstate/retrieval.py. | **OPEN:** paper correctly labels scores as metadata-mechanism evaluation, not independent relevance or real-world safety. |
| CP-03 | R2/R4: adaptive versus fixed | scripts/experiments/run_retrieval_ablations.py and ignored ablation_results.json: beta 0.35 selected from 359 validation scenarios; test fixed nDCG 0.6824 / SIRR 0.1000 vs adaptive 0.6668 / 0.1243. | **CORRECTED IN CAMERA-READY SOURCE:** B3 beta 0.20 is explicitly untuned, and tuned fixed is stronger on these metadata metrics. |
| CP-04 | R1–R3: abstention selection | Validation-only confidence cutoff 0.524541 retains 189/263 test images, coverage 0.7186, weak-label risk 0.1217. | **CORRECTED IN CAMERA-READY SOURCE:** the 0.54-coverage point is explicitly illustrative; the validation-selected threshold 0.5245 and its held-out outcome are reported separately. |
| CP-05 | R2: 64 omitted test images | Scenario builder requires >=3 relevant and >=3 incompatible chunks per (stage, topic); only 199/263 test images have at least one qualifying topic, yielding 420 scenarios. | **CORRECTED IN CAMERA-READY SOURCE:** explicitly explain 64 exclusions as corpus/scenario eligibility, not an image-model failure. |
| CP-06 | R1–R4: pHash and field leakage | src/cropstate/splits.py and scripts/experiments/leakage_audit.py implement grayscale 8x8 **aHash** Hamming clustering before splitting. | **CORRECTED:** short paper now says aHash, acknowledges zero cross-split matches is partly by construction and does not prove field/season independence. |
| CP-07 | R2–R4: rare-stage reliability | Same 24 reproductive and 10 grain-filling test images across seeded runs under one fixed group split. | **CLAIM NARROWED:** identical recall across seeds is descriptive on the small shared subset, not proof the ordinal loss causally stabilizes recall across fields. |
| CP-08 | R2: configuration fidelity | scripts/evaluate_retrieval.py evaluation default source_weight=0; library API src/cropstate/retrieval.py defaults source_weight=0.10. | **OPEN CONFIGURATION-SCOPE HAZARD:** the paper's weight-zero result is true of the evaluation entrypoint, not of arbitrary library calls; pin exact CLI flags/versions. |
| CP-09 | R2: reproducibility | scripts/reproduce_retrieval_benchmark.py, ONNX encoder hashes, 420-scenario metrics, 5000 clustered resamples, paper/CAMERA_READY_RESULTS_STATUS.md. | **PARTIALLY CORRECTED:** retrieval rerun verified; vision metrics historical. Release only permitted configs/provenance and rebuild 6-page PDF on current code. |
| CP-10 | R2: manuscript completeness | The short paper described simulated temporal fusion as a contribution without corresponding methods/results in that short version; the temporal_fusion.py experiment exists but is not quantified in the 6-page manuscript. | **CORRECTED IN CAMERA-READY SOURCE:** omit the unreported contribution; recompile PDF. The table label tab:retrsig was checked and is not duplicated. |
| CP-11 | R1–R4: generalization and novelty | ResNet-18 CE/ordinal/focal objectives, no separately verified external field/season/device test. | **OPEN:** present uncertainty-aware retrieval trade-off as the contribution, not a new backbone or field-proven safety. |

## Strict provenance and evidence boundaries

- **Keep:** real EDAS reviewers' numeric scores and substantive criticism, code/paper cross-checks, verifiable correction status, unresolved reviewer requirements.
- **No unverified expert substitution:** only authentically submitted human judgments may support expert-label or inter-rater claims.
- **No inference from absent evidence:** two independent agronomy sheets have no completed human judgments, so human kappa, expert BBCH reference accuracy and expert-judged retrieval nDCG are **NOT ESTABLISHED**.
- **No delivery overclaim:** repository source updates and passing tests do not establish successful PDF recompilation, final conference submission or human expert validation.

**Original reviewer source:** author-supplied RIVF'26 EDAS acceptance/decision letter dated 2026-10-25 for paper #1571322865, sections Review 1 through Review 4 (retained privately; not the AI agronomy report).

**Comparison targets:** paper/cropstate_camera_ready_ieee_short.tex, docs/INDEPENDENT_VALIDATION_PROTOCOL.md, paper/CAMERA_READY_RESULTS_STATUS.md, scripts/evaluate_retrieval.py, scripts/experiments/build_belief_scenarios.py, scripts/experiments/run_retrieval_ablations.py and src/cropstate/retrieval.py.


## Reproducible integration verification

The integration check verifies reviewer-packet blinding, score eligibility and fixture-based double consensus. It does **not** create or certify genuine human ratings. From repository root with dependencies installed:

~~~bash
PYTHONPATH=src:. .venv/bin/python -m unittest discover -s tests -p 'test_integration_validation_pipeline.py' -v
~~~

The checked-in GitHub Actions workflow runs the same integration test with a fresh Python 3.11 environment after installing lightweight test dependencies. Its result is an engineering test, separate from RIVF reviewers' scientific findings.
