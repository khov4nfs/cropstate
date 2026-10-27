"""Prepare two independently blinded BBCH stage review sheets from a weak-label manifest.

The model's weak labels/review flags are only in admin_manifest.csv, never in either
reviewer sheet. One image per original parent scene is sampled to avoid redundant
overlapping crops. This creates annotation TASKS, not expert-verified labels.

Usage:
  python scripts/prepare_bbch_expert_review.py \
    --manifest data/riceseg_stage_manifest.csv --per-stage 20 \
    --output-dir results/bbch_review --seed 42
  # When images are mounted, add --data-root /path/to/CROPSTATE_DATASET to
  # copy neutral-named images to results/bbch_review/assets/.

Reviewers should see ONLY reviewer_1.csv / reviewer_2.csv and assets/, never
admin_manifest.csv. Expert annotation requires human agronomists.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

STAGES = (
    "establishment", "tillering", "stem_booting",
    "reproductive", "grain_filling", "ripening",
)
TASK_FIELDS = (
    "review_id", "image_filename", "reviewer_bbch", "reviewer_stage",
    "uncertain", "notes",
)


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for col in ("image_path", "parent_image_id", "macro_stage"):
        if not rows or col not in rows[0]:
            raise ValueError(f"Manifest missing required column: {col}")
    for row in rows:
        if row["macro_stage"] not in STAGES:
            raise ValueError(f"Unknown weak stage: {row['macro_stage']}")
        if not row["image_path"] or not row["parent_image_id"]:
            raise ValueError("Missing image_path or parent_image_id")
    return rows


def select_one_per_parent(rows: list[dict[str, str]], per_stage: int, seed: int) -> list[dict[str, str]]:
    if per_stage < 1:
        raise ValueError("per_stage must be >= 1")
    rng = random.Random(seed)
    parents: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        parents[row["parent_image_id"]].append(row)

    groups: dict[str, list[dict[str, str]]] = {stage: [] for stage in STAGES}
    for parent_id in sorted(parents):
        members = parents[parent_id]
        counts = Counter(row["macro_stage"] for row in members)
        # A parent with conflicting weak labels is still only sampled once.
        weak_stage = next(stage for stage in STAGES if counts[stage] == max(counts.values()))
        eligible = [row for row in members if row["macro_stage"] == weak_stage]
        representative = rng.choice(eligible).copy()
        representative["parent_weak_stage_conflict"] = str(len(counts) > 1).lower()
        groups[weak_stage].append(representative)

    selected = []
    for stage in STAGES:
        candidates = groups[stage]
        flagged = [row for row in candidates if str(row.get("review_flag", "0")) == "1"]
        unflagged = [row for row in candidates if str(row.get("review_flag", "0")) != "1"]
        rng.shuffle(flagged)
        rng.shuffle(unflagged)
        # Sample flagged and unflagged roughly equally when both are available.
        quota = min(per_stage, len(candidates))
        n_flagged = min(len(flagged), (quota + 1) // 2)
        n_unflagged = min(len(unflagged), quota - n_flagged)
        if n_flagged + n_unflagged < quota:
            n_flagged = min(len(flagged), quota - n_unflagged)
        if n_flagged + n_unflagged < quota:
            n_unflagged = min(len(unflagged), quota - n_flagged)
        selected.extend(flagged[:n_flagged] + unflagged[:n_unflagged])
    rng.shuffle(selected)
    return selected


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def prepare_review(
    rows: list[dict[str, str]],
    output_dir: Path,
    per_stage: int = 20,
    seed: int = 42,
    data_root: Path | None = None,
) -> dict:
    selected = select_one_per_parent(rows, per_stage=per_stage, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    if data_root is not None:
        # Validate all source paths before making any copy: do not silently create
        # misleading empty reviewer assets when images are missing.
        missing = [str(data_root / row["image_path"]) for row in selected
                   if not (data_root / row["image_path"]).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} image assets, first: {missing[0]}")
        (output_dir / "assets").mkdir(exist_ok=True)

    admin, tasks = [], []
    for idx, row in enumerate(selected, start=1):
        review_id = f"BBCH{idx:05d}"
        source_path = Path(row["image_path"])
        neutral_filename = f"{review_id}{source_path.suffix.lower()}"
        if data_root is not None:
            shutil.copy2(data_root / source_path, output_dir / "assets" / neutral_filename)
        source_key = row["parent_image_id"] + "/" + row["image_path"]
        admin.append({
            "review_id": review_id,
            "image_filename": neutral_filename,
            "source_image_path": row["image_path"],
            "parent_image_id": row["parent_image_id"],
            "weak_stage": row["macro_stage"],
            "review_flag": row.get("review_flag", ""),
            "split": row.get("split", ""),
            "source_key_sha256": hashlib.sha256(source_key.encode("utf-8")).hexdigest(),
            "parent_weak_stage_conflict": row["parent_weak_stage_conflict"],
        })
        tasks.append({
            "review_id": review_id, "image_filename": neutral_filename,
            "reviewer_bbch": "", "reviewer_stage": "", "uncertain": "",
            "notes": "",
        })
    write_csv(output_dir / "admin_manifest.csv", admin, list(admin[0]) if admin else [
        "review_id", "image_filename", "source_image_path", "parent_image_id",
        "weak_stage", "review_flag", "split", "source_key_sha256",
        "parent_weak_stage_conflict",
    ])
    write_csv(output_dir / "reviewer_1.csv", tasks, TASK_FIELDS)
    write_csv(output_dir / "reviewer_2.csv", tasks, TASK_FIELDS)
    return {"total": len(selected), "per_stage": dict(Counter(row["macro_stage"] for row in selected)),
            "image_assets_copied": data_root is not None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--per-stage", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, help="Copy neutral filenames for blinded image review.")
    args = parser.parse_args()
    result = prepare_review(load_manifest(args.manifest), args.output_dir,
                            args.per_stage, args.seed, args.data_root)
    print(result)
    if args.data_root is None:
        print("NOTE: Add --data-root with the original images to create neutral assets for reviewers.")


if __name__ == "__main__":
    main()
