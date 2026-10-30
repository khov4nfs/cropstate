"""Prepare and score independent, blinded retrieval relevance review.

The reviewer receives image + query + passage without predicted/weak stage,
model method, metadata compatibility or automatic relevance flags.
The generated forms are BLANK. No expert validation is claimed until two
independent human raters submit completed, matching judgments.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from cropstate.knowledge import load_knowledge_chunks
from cropstate.retrieval import build_topic_query

METHODS = ("B0_ungated", "B1_query_expansion", "B2_hard_filter",
           "B3_fixed_soft", "P_adaptive_soft", "oracle_reference")
REVIEW_COLS = ("item_id", "case_id", "image_filename", "query", "passage",
               "source_title", "relevance_grade", "stage_inappropriate",
               "image_evidence_sufficient", "uncertain", "notes")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def context_hash(row: dict) -> str:
    context = {k: row[k] for k in ("item_id", "case_id", "image_filename",
                                  "query", "passage", "source_title")}
    return hashlib.sha256(json.dumps(context, ensure_ascii=False, sort_keys=True)
                          .encode("utf-8")).hexdigest()


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...] | list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path, key: str) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if key not in (reader.fieldnames or []):
            raise ValueError(f"{path} missing required {key}")
        rows = list(reader)
    values = {}
    for row in rows:
        rid = row[key]
        if not rid or rid in values:
            raise ValueError(f"Duplicate/empty {key}: {rid!r}")
        values[rid] = row
    return values


def select_cases(scenarios: list[dict], n_cases: int, seed: int) -> list[dict]:
    """Balance on *weak* stages only; pick at most one topic per source image."""
    if n_cases < 1:
        raise ValueError("n_cases must be positive")
    by_image = defaultdict(list)
    for row in scenarios:
        if row.get("scenario_id") and row.get("ground_truth_stage"):
            by_image[row["scenario_id"].rsplit("__", 1)[0]].append(row)
    rng = random.Random(seed)
    groups = defaultdict(list)
    for img in sorted(by_image):
        candidates = sorted(by_image[img], key=lambda x: x["scenario_id"])
        selected = rng.choice(candidates)
        groups[selected["ground_truth_stage"]].append(selected)
    for v in groups.values():
        rng.shuffle(v)
    selected = []
    # Round robin avoids exhausting common classes before rare ones.
    while len(selected) < n_cases and any(groups.values()):
        for stage in sorted(groups):
            if groups[stage] and len(selected) < n_cases:
                selected.append(groups[stage].pop())
    return selected


def prepare(args: argparse.Namespace) -> dict:
    cases = list(read_csv(args.scenarios, "scenario_id").values())
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    if evaluation.get("scenario_count") != len(cases):
        raise ValueError("Evaluation/scenario count mismatch")
    by_scenario = defaultdict(dict)
    for row in evaluation["rows"]:
        if row["method"] not in METHODS:
            raise ValueError("Unknown evaluation method")
        by_scenario[row["scenario_id"]][row["method"]] = row["top_ids"]
    chunks = {c.chunk_id: c for c in load_knowledge_chunks(args.corpus, mode="research")}
    manifest = read_csv(args.image_manifest, "image_id")
    selected = select_cases(cases, args.n_cases, args.seed)
    if len(selected) < args.n_cases:
        raise ValueError(f"Only {len(selected)} distinct images, need {args.n_cases}")
    # Fail closed BEFORE creating a partial image packet.
    sources = []
    for row in selected:
        image_id = row["scenario_id"].rsplit("__", 1)[0]
        meta = manifest.get(image_id)
        if not meta:
            raise ValueError(f"No manifest record for {image_id}")
        source = args.data_root / meta["image_path"]
        if not source.is_file():
            raise FileNotFoundError(source)
        methods = by_scenario.get(row["scenario_id"], {})
        if set(methods) != set(METHODS):
            raise ValueError(f"Expected all six method rankings for {row['scenario_id']}")
        sources.append(source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assets = args.output_dir / "assets"
    assets.mkdir(exist_ok=True)
    reviewer, admin_cases, admin_items = [], [], []
    for i, (scenario, source) in enumerate(zip(selected, sources), 1):
        cid = f"CASE{i:04d}"
        # Normalize orientation and re-encode RGB to remove source EXIF and GPS.
        from PIL import Image, ImageOps
        filename = cid + ".jpg"
        with Image.open(source) as original:
            ImageOps.exif_transpose(original).convert("RGB").save(
                assets / filename, format="JPEG", quality=95
            )
        scenario_id = scenario["scenario_id"]
        methods = by_scenario[scenario_id]
        pool = sorted({doc for ranking in methods.values() for doc in ranking})
        admin_cases.append({
            "case_id": cid, "scenario_id": scenario_id,
            "image_id": scenario_id.rsplit("__", 1)[0],
            "weak_stage": scenario["ground_truth_stage"],
            "predicted_stage": scenario["predicted_stage"],
            "topic": scenario["topic"], "source_image_sha256": digest(source),
            "pooled_documents": len(pool),
            "ranking_json": json.dumps(methods, sort_keys=True),
        })
        for j, doc_id in enumerate(pool, 1):
            if doc_id not in chunks:
                raise ValueError(f"Missing knowledge chunk {doc_id}")
            chunk = chunks[doc_id]
            item_id = f"{cid}D{j:03d}"
            reviewer.append({
                "item_id": item_id, "case_id": cid, "image_filename": filename,
                "query": build_topic_query(scenario["topic"]),
                "passage": chunk.text, "source_title": chunk.source_title,
                "relevance_grade": "", "stage_inappropriate": "",
                "image_evidence_sufficient": "", "uncertain": "", "notes": "",
            })
            admin_items.append({"item_id": item_id, "case_id": cid,
                                "chunk_id": doc_id, "reviewer_context_sha256": context_hash(reviewer[-1])})
    write_csv(args.output_dir / "admin_cases.csv", admin_cases, list(admin_cases[0]))
    write_csv(args.output_dir / "admin_items.csv", admin_items, list(admin_items[0]))
    for reviewer_num in (1, 2):
        write_csv(args.output_dir / f"reviewer_{reviewer_num}.csv", reviewer, REVIEW_COLS)
    status = {
        "human_completed_labels": 0, "expert_consensus_labels": 0,
        "stage": "blank_review_packet_not_expert_validation",
        "n_cases": len(admin_cases), "n_items": len(reviewer),
        "n_distinct_source_images": len(selected),
        "seed": args.seed, "sampling": "weak-stage balanced, one case per image",
        "selected_image_hashes": digest(args.output_dir / "admin_cases.csv"),
        "instructions": "Only send each reviewer their own sheet and neutral assets; never admin CSV.",
    }
    (args.output_dir / "STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
    return status


def info_hash(admin_item: dict) -> str:
    value = admin_item.get("reviewer_context_sha256", "")
    if len(value) != 64:
        raise ValueError("Missing/invalid reviewer context checksum in admin items")
    return value


def judgment(row: dict) -> tuple[int, bool, bool] | None:
    if row["uncertain"].strip().lower() in ("yes", "true", "1"):
        return None
    grade = row["relevance_grade"].strip()
    unsafe = row["stage_inappropriate"].strip().lower()
    sufficient = row["image_evidence_sufficient"].strip().lower()
    if not grade and not unsafe and not sufficient:
        return None
    if grade not in ("0", "1", "2") or unsafe not in ("yes", "no") or sufficient not in ("yes", "no"):
        raise ValueError(f"Invalid or partial expert judgment at {row['item_id']}")
    return int(grade), unsafe == "yes", sufficient == "yes"


def dcg(values: list[int], k: int) -> float:
    return sum((2 ** grade - 1) / math.log2(idx + 2)
               for idx, grade in enumerate(values[:k]))


def grade_case(ranking: list[str], graded: dict[str, tuple[int, bool, bool]],
               k: int = 5) -> dict[str, float | None]:
    relevant = sum(value[0] > 0 for value in graded.values())
    actual = [graded[c][0] for c in ranking[:k]]
    ideal = sorted((v[0] for v in graded.values()), reverse=True)
    ideal_dcg = dcg(ideal, k)
    sufficient = all(graded[c][2] for c in ranking[:k])
    return {
        "precision_at_5": sum(v > 0 for v in actual) / k,
        "pooled_recall_at_5": sum(v > 0 for v in actual) / relevant if relevant else 0.0,
        "pooled_ndcg_at_5": dcg(actual, k) / ideal_dcg if ideal_dcg else 0.0,
        "stage_inappropriate_at_5": (
            sum(graded[c][1] for c in ranking[:k]) / k if sufficient else None
        ),
    }


def score(args: argparse.Namespace) -> dict:
    cases = read_csv(args.admin_dir / "admin_cases.csv", "case_id")
    items = read_csv(args.admin_dir / "admin_items.csv", "item_id")
    a = read_csv(args.reviewer_1, "item_id")
    b = read_csv(args.reviewer_2, "item_id")
    if not items or set(a) != set(items) or set(b) != set(items):
        raise ValueError("Reviewer sheets must contain all and only packet items")
    # Verify reviewer-visible fields identical: no swapped query/passages.
    for rid in items:
        if (context_hash(a[rid]) != info_hash(items[rid]) or
                context_hash(b[rid]) != info_hash(items[rid])):
            raise ValueError(f"Rater context changed since packet generation: {rid}")
    by_case = defaultdict(dict)
    skipped = 0
    for rid, info in items.items():
        x, y = judgment(a[rid]), judgment(b[rid])
        if x is not None and x == y:
            by_case[info["case_id"]][info["chunk_id"]] = x
        else:
            skipped += 1
    per_method = defaultdict(list)
    eligible = []
    stage_safety_cases = 0
    for cid, case in cases.items():
        rankings = json.loads(case["ranking_json"])
        pool = set(doc for ranking in rankings.values() for doc in ranking)
        if not pool.issubset(by_case[cid]):
            continue  # Never treat an unjudged passage as grade zero.
        eligible.append(cid)
        # Enforce one common image-evidence subset across ALL method comparisons.
        image_sufficient = all(by_case[cid][doc][2] for doc in pool)
        if image_sufficient:
            stage_safety_cases += 1
        for method, ranking in rankings.items():
            result = grade_case(ranking, by_case[cid])
            if not image_sufficient:
                result["stage_inappropriate_at_5"] = None
            per_method[method].append(result)
    result = {
        "status": "insufficient_expert_evidence" if not eligible else "complete_consensus_subset_only",
        "packet_cases": len(cases),
        "packet_items": len(items),
        "paired_identical_items": len(items) - skipped,
        "unresolved_items": skipped,
        "fully_consensus_cases": len(eligible),
        "cases_with_sufficient_image_evidence": stage_safety_cases,
        "methods": {
            method: {
                key: (sum(d[key] for d in rows if d[key] is not None) /
                      len([d for d in rows if d[key] is not None])
                      if any(d[key] is not None for d in rows) else None)
                for key in ("precision_at_5", "pooled_recall_at_5", "pooled_ndcg_at_5",
                            "stage_inappropriate_at_5")
            } for method, rows in per_method.items()
        },
        "limitations": "Only double-consensus, fully pooled cases counted; selected weak-stage-balanced subset, not population or field-independent accuracy. Pooled recall is NOT corpus recall. Stage-inappropriateness only for cases where image evidence was judged sufficient.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    subs = ap.add_subparsers(dest="command", required=True)
    p = subs.add_parser("prepare")
    for key in ("scenarios", "evaluation", "corpus", "image_manifest", "data_root", "output_dir"):
        p.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    p.add_argument("--n-cases", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    s = subs.add_parser("score")
    for key in ("admin_dir", "reviewer_1", "reviewer_2", "output"):
        s.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    args = ap.parse_args()
    result = prepare(args) if args.command == "prepare" else score(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
