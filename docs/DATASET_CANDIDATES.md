# Candidate external datasets for CropState experiments (research, not imports)

Checked: 2026-10-27. **No image/data download, relabeling, or automatic import
is part of this change.** Public availability, annotation provenance, terms and
geographic leakage must be checked again before acquisition. Dataset-derived
stage labels must **not** be described as expert BBCH ground truth without validation.

| Dataset | Link | Available annotation / modality | Intended use | Limitations and access |
| --- | --- | --- | --- | --- |
| RiceSEG (PheniX-Lab) | https://huggingface.co/datasets/PheniX-Lab/RiceSEG | RGB field images and 6 **segmentation pixel classes**, not 6 BBCH stages | Reproduce existing weak-label benchmark; audit labeling heuristics | Hugging Face dataset page currently shows **gated** access, MIT metadata; retain original source/citation, never infer stage ground truth from segmentation-only labels |
| RiceStageSeg | https://github.com/SHANOAL/RiceStageSeg and https://doi.org/10.3390/rs17162858 | RGB/multispectral **remote sensing**, stage-wise pixel segmentation | Independent, cross-modality study **only after stage-ontology mapping** | Repository README points to Baidu distribution; license and practical download must be verified; do not naively merge with ground-level RiceSEG |
| Multitemporal multispectral rice phenology (Dryad) | https://doi.org/10.5061/dryad.v41ns1s4z | UAS multispectral imagery, plot/date identifiers and plot-level phenology/yield; 2021–2022 | Domain shift / plot-group and temporal fusion experiments | ~293 MB of archives; remote imagery rather than RGB ground photos; inspect stage definitions, files, license before reuse |
| RiGaD / AgML rice seedlings | https://huggingface.co/datasets/Project-AgML/rice_seedling_classification and https://doi.org/10.5281/zenodo.10960906 | ~5,364 seedling patches; single, clustered, undefined classes | Out-of-distribution / seedling-only diagnostic | **NOT** six BBCH classes; different task and aerial view; avoid reporting six-stage accuracy |
| Rice seedling BBCH 11–13 (published study) | https://doi.org/10.1038/s41598-025-23499-6 | 458 greenhouse seedling images with BBCH 11/12/13 study labels | Investigate expert stage protocol and fine-grained establishment-stage evaluation | Access to raw images/annotation licenses is **not yet verified**; nursery domain, three early stages only |

## Dataset-plugin workflow checked

- **Hugging Face connector**: search and inspect dataset structure/configs/splits. The
  RiceSEG Hub dataset has a displayed image-only viewer schema; segmentation data
  and original parent identities may require inspecting the original archive rather
  than the auto-converted image column.
- **Adaption dataset connector**: can import from Hugging Face or Kaggle and perform
  processing / augmentation. This is designed primarily for adaptive training
  dataset preparation; synthetic/derived labels **cannot** substitute for
  agronomist-verified benchmark annotations. Do not launch costly imports,
  transformations, or synthetic data generation without a clear experiment plan.
- **Dewey Data**: searched for rice growth-stage imagery; no matching accessible
  records returned at the time of review.

## Recommended sequence

1. First build a 100–200 example, stratified, blinded **expert-review subset** of
   the existing RiceSEG benchmark (including image parent, source region, BBCH
   rubric and expert disagreement). This best addresses reviewer ground-truth concern.
2. Evaluate mapping/compatibility for RiceStageSeg and Dryad on **small metadata
   samples**. No automatic mapping from remote segmentation masks to the six
   macro-stages; document missing/ambiguous stage mapping explicitly.
3. Keep `train/val/test` independent by original image/plot/field and preferably
   acquisition campaign. Report domain shift separately from in-domain accuracy.
4. Build a separate expert relevance gold set for retrieved knowledge chunks.
   More imagery does **not** resolve retrieval label circularity.

## Minimal provenance schema for new acquisitions

Each entry should record: `source_id`, `canonical_url`, `license_verified`,
`download_authorization`, `original_id`, `parent_or_plot_id`,
`capture_date_or_session`, `country_or_region`, `sensor_modality`,
`label_type`, `label_provenance`, `bbch_mapping_method`, `split_group_id`,
`citation`, and `image_checksum`. Never copy third-party raw images into the
public repository without permission.

References:
- https://www.global-rice.com/
- https://www.mdpi.com/2072-4292/17/16/2858
- https://datadryad.org/dataset/doi:10.5061/dryad.v41ns1s4z
- https://huggingface.co/datasets/PheniX-Lab/RiceSEG
- https://huggingface.co/datasets/Project-AgML/rice_seedling_classification
