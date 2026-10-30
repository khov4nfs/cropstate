# BBCH expert validation: reviewer protocol and claim boundary

**Status as of 2026-10-27**: 120 review tasks prepared automatically
(20 per weak stage), but **0 human annotations received**, **0 expert-verified
labels**, and **no kappa estimate**. Do not report this as completed validation.

## Role separation and blinding

- The **administrator** receives \`admin_manifest.csv\`, which contains
  \`weak_stage\`, \`review_flag\`, image source path, parent ID, and the
  neutral image name. Never send this file to stage raters.
- Reviewer A receives **only** \`reviewer_1.csv\`, the neutral \`assets/\`
  directory and \`docs/ANNOTATION_GUIDELINE.md\`. Reviewer B receives
  \`reviewer_2.csv\` plus the same neutral images and rubric. Annotators
  should not see the other reviewer's file until both are submitted.
- Images are sampled one per \`parent_image_id\` to minimize overlapping
  crops being treated as independent votes.
- If a segmentation RGB image contains insufficient evidence for BBCH
  (especially booting vs tillering, or anthesis vs filling), record
  \`uncertain=yes\` and describe why. Do **not** infer from the original
  filenames, weak labels, stage-folder name, color alone or the paper's
  expected class proportions.

## Prepare the blinded image assets (required before human reviewing)

After downloading the GitHub Actions artifact
\`rice-bbch-review-packet-blank\`, stage the corresponding original RGB
images from the RiceSEG source (subject to its license and access terms):

\`\`\`bash
python scripts/prepare_bbch_expert_review.py \
  --manifest data/riceseg_stage_manifest.csv \
  --per-stage 20 --seed 42 \
  --data-root /path/to/CROPSTATE_DATASET \
  --output-dir /path/to/secured_bbch_review
\`\`\`

This command uses the same deterministic seed and parent sampling to
recreate the **120** reviewer rows, and copies each source to a neutral
\`assets/BBCH00001.jpg\`-style filename. Inspect at least several mapping
pairs before distributing. Avoid sharing original images publicly without
permission. Administrative paths are NOT included in the blinded sheets.

## What annotators must fill

On each row:

- \`reviewer_bbch\`: the finest visually defensible two-digit code in
  \`00..89\`; leave blank if fine sub-stage cannot be judged.
- \`reviewer_stage\`: one of
  \`establishment, tillering, stem_booting, reproductive, grain_filling, ripening\`.
  The code can also derive the stage from a supplied BBCH number, but both
  entries must agree when provided.
- \`uncertain\`: "yes" when visual evidence is insufficient, ambiguous,
  occluded or mixed-stage; such a row is withheld from automatic consensus.
- \`notes\`: observed morphology and evidence limitations.

Raters should be independent agricultural/phenology specialists, not an
AI model simulating a specialist.

## Score completed reviews only

\`\`\`bash
python scripts/score_bbch_expert_review.py \
  --admin /path/to/secured_bbch_review/admin_manifest.csv \
  --reviewer-1 /path/to/secured_bbch_review/reviewer_1.csv \
  --reviewer-2 /path/to/secured_bbch_review/reviewer_2.csv \
  --output-dir /path/to/secured_bbch_review/scored
\`\`\`

Outputs:
- \`expert_quality_report.json\`: number of paired completed rows, raw
  inter-rater agreement, Cohen's kappa on paired labels, count of agreed
  stage labels, and heuristic-vs-expert agreement **on consensus subset only**.
- \`expert_consensus.csv\`: an explicit \`two_independent_reviewers_agree\`
  provenance for paired consensus, not a replacement for full test labels.
- \`adjudication_queue.csv\`: disagreements and uncertain/missing rows
  for a third independent agronomist. The current scoring tool does not
  silently count adjudication outcomes as two-reviewer consensus.

## Analysis and limitations

1. Report exact sample method, per-class counts, expert credentials
   (without personal identifiers), and the number of insufficient-image
   cases. Treat unknown/uncertain as missing, not negative agreement.
2. The sample is balanced by *heuristically derived* stage labels, not a
   representative estimate of the real-world stage prevalence. Label
   agreement cannot be extrapolated to population prevalence.
3. Expert-review agreement with weak labels is NOT classification
   accuracy of a trained model unless predictions for these exact image
   identities are also evaluated against the independently verified
   expert labels with known train/test provenance. Do not evaluate on
   training images and call it a held-out test.
4. If field/session data is absent, do not claim field-independent
   generalization. Obtain original parent/plot identifiers and collect
   independent acquisition seasons for that purpose.
5. A second, independent annotation task is required for ranking
   relevance and agronomic recommendation appropriateness. Image BBCH
   validation alone cannot resolve metadata-circular retrieval labels.

## Preprinted review packet

The GitHub Actions job
\`BBCH independent expert review packet\` publishes a *blank* CSV artifact
and its explicit zero-expert-consensus status. It intentionally contains
no redistributed RiceSEG images.

Review protocol references: \`docs/ANNOTATION_GUIDELINE.md\`;
\`scripts/prepare_bbch_expert_review.py\`;
\`scripts/score_bbch_expert_review.py\`.
