"""Score independently completed, blinded BBCH annotation sheets.

Expert validation is ONLY established when real human reviewer_1.csv and reviewer_2.csv
have been completed independently. Empty sheets cannot produce expert results.

Usage:
 python scripts/score_bbch_expert_review.py \
    --admin results/bbch_review/admin_manifest.csv \
    --reviewer-1 results/bbch_review/reviewer_1.csv \
    --reviewer-2 results/bbch_review/reviewer_2.csv \
    --output-dir results/bbch_review/scored
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

STAGES = (
    "establishment", "tillering", "stem_booting",
    "reproductive", "grain_filling", "ripening",
)


def read_unique_csv(path: Path, key: str = "review_id") -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if rows and key not in rows[0]:
        raise ValueError(f"{path}: missing {key}")
    result = {}
    for row in rows:
        rid = row[key]
        if not rid or rid in result:
            raise ValueError(f"{path}: missing or duplicate {key}={rid!r}")
        result[rid] = row
    return result


def stage_for_bbch(value: str) -> str:
    code = value.strip().upper().removeprefix("BBCH").strip()
    if not code.isdigit() or len(code) > 2:
        raise ValueError(f"Invalid two-digit BBCH principal/sub-stage: {value!r}")
    n = int(code)
    if n < 20:
        return "establishment"
    if n < 30:
        return "tillering"
    if n < 50:
        return "stem_booting"
    if n < 70:
        return "reproductive"
    if n < 80:
        return "grain_filling"
    if n < 90:
        return "ripening"
    raise ValueError(f"BBCH {value!r} outside paper's mapped 00–89 interval; flag for adjudication")


def annotation_stage(row: dict[str, str]) -> str | None:
    if str(row.get("uncertain", "")).strip().lower() in {"1", "true", "yes", "y"}:
        return None
    explicit = row.get("reviewer_stage", "").strip().lower().replace("/", "_").replace("-", "_")
    bbch = row.get("reviewer_bbch", "").strip()
    derived = stage_for_bbch(bbch) if bbch else ""
    if explicit and explicit not in STAGES:
        raise ValueError(f"Invalid reviewer stage: {explicit!r}")
    if explicit and derived and explicit != derived:
        raise ValueError(f"Conflicting reviewer stage {explicit!r} vs BBCH {bbch!r}")
    return explicit or derived or None


def cohens_kappa(a: list[str], b: list[str]) -> float | None:
    if len(a) != len(b):
        raise ValueError("Paired labels must have equal lengths")
    if not a:
        return None
    n = len(a)
    obs = sum(x == y for x, y in zip(a, b)) / n
    counts_a, counts_b = Counter(a), Counter(b)
    expected = sum(counts_a[c] * counts_b[c] for c in STAGES) / (n * n)
    return (obs - expected) / (1.0 - expected) if expected < 1 else (1.0 if obs == 1.0 else None)


def write_csv(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        writer.writerows(rows)


def score_review(admin: dict[str, dict[str, str]],
                 reviewer_1: dict[str, dict[str, str]],
                 reviewer_2: dict[str, dict[str, str]]) -> tuple[dict, list[dict], list[dict]]:
    if set(admin) != set(reviewer_1) or set(admin) != set(reviewer_2):
        raise ValueError("Both review sheets must contain exactly the admin review IDs")
    paired_1, paired_2 = [], []
    confirmed = []
    adjudicate = []
    for rid, entry in admin.items():
        a, b = annotation_stage(reviewer_1[rid]), annotation_stage(reviewer_2[rid])
        if a is not None and b is not None:
            paired_1.append(a)
            paired_2.append(b)
        if a is not None and a == b:
            confirmed.append({
                "review_id": rid,
                "source_image_path": entry["source_image_path"],
                "parent_image_id": entry["parent_image_id"],
                "split": entry.get("split", ""),
                "weak_stage": entry["weak_stage"],
                "expert_consensus_stage": a,
                "review_provenance": "two_independent_reviewers_agree",
            })
        else:
            adjudicate.append({
                "review_id": rid,
                "image_filename": entry["image_filename"],
                "reviewer_1_stage": a or "",
                "reviewer_2_stage": b or "",
                "resolution_stage": "",
                "resolution_reviewer": "",
                "notes": "",
            })
    stats = {
        "task_count": len(admin),
        "paired_completed_count": len(paired_1),
        "expert_consensus_count": len(confirmed),
        "requires_review_or_adjudication": len(adjudicate),
        "interreviewer_agreement_on_complete_pairs": (
            sum(x == y for x, y in zip(paired_1, paired_2)) / len(paired_1) if paired_1 else None
        ),
        "interreviewer_cohens_kappa_on_complete_pairs": cohens_kappa(paired_1, paired_2),
        "weak_label_agreement_on_consensus_subset": (
            sum(r["weak_stage"] == r["expert_consensus_stage"] for r in confirmed) / len(confirmed)
            if confirmed else None
        ),
        "expert_consensus_stage_counts": dict(Counter(r["expert_consensus_stage"] for r in confirmed)),
        "note": "Agreement is on a sampled subset, not six-class test accuracy. No adjudicated labels are silently counted as consensus.",
    }
    return stats, confirmed, adjudicate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admin", required=True, type=Path)
    parser.add_argument("--reviewer-1", required=True, type=Path)
    parser.add_argument("--reviewer-2", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = score_review(read_unique_csv(args.admin),
                          read_unique_csv(args.reviewer_1),
                          read_unique_csv(args.reviewer_2))
    stats, confirmed, adjudicate = result
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "expert_quality_report.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "expert_consensus.csv", confirmed,
              ("review_id", "source_image_path", "parent_image_id", "split", "weak_stage",
               "expert_consensus_stage", "review_provenance"))
    write_csv(args.output_dir / "adjudication_queue.csv", adjudicate,
              ("review_id", "image_filename", "reviewer_1_stage",
               "reviewer_2_stage", "resolution_stage", "resolution_reviewer", "notes"))
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if not stats["paired_completed_count"]:
        print("BLOCKED: No independently completed human labels; expert validation is not established.")


if __name__ == "__main__":
    main()
