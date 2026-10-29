from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from cropstate.constants import STAGE_ALIASES, STAGE_NAMES
from cropstate.knowledge import load_knowledge_chunks
from cropstate.metrics import ndcg_at_k, precision_recall_at_k, sirr_at_k
from cropstate.retrieval import HybridRetriever, build_topic_query, hard_filter, minmax, reciprocal_rank_fusion, rerank


def parse_ids(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, list):
        return {str(item) for item in value}
    text = str(value).strip()
    if not text:
        return set()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return {str(item) for item in parsed}
    except json.JSONDecodeError:
        pass
    return {item.strip() for item in text.replace(",", "|").split("|") if item.strip()}


def read_scenarios(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_stage(stage: str) -> str:
    key = stage.strip().lower().replace("/", "_").replace("-", "_").replace(" ", "_")
    normalized = STAGE_ALIASES.get(key, key)
    if normalized not in STAGE_NAMES:
        raise ValueError(f"Unknown stage: {stage}")
    return normalized


def one_hot(stage: str) -> np.ndarray:
    normalized = normalize_stage(stage)
    vector = np.zeros(len(STAGE_NAMES), dtype=float)
    vector[STAGE_NAMES.index(normalized)] = 1.0
    return vector


def parse_belief(value: Any, fallback_stage: str) -> np.ndarray:
    if value in (None, ""):
        return one_hot(fallback_stage)
    if isinstance(value, list):
        belief = np.asarray(value, dtype=float)
    else:
        text = str(value).strip()
        try:
            belief = np.asarray(json.loads(text), dtype=float)
        except json.JSONDecodeError:
            belief = np.asarray([float(part) for part in text.split(",")], dtype=float)
    if belief.shape != (len(STAGE_NAMES),) or belief.sum() <= 0:
        raise ValueError("stage_belief must contain six non-negative values with a positive sum")
    return belief / belief.sum()


def parse_relevance(scenario: dict[str, Any]) -> dict[str, float]:
    raw = scenario.get("relevance_grades")
    if raw:
        parsed = raw if isinstance(raw, dict) else json.loads(str(raw))
        return {str(key): float(value) for key, value in parsed.items()}
    return {document_id: 1.0 for document_id in parse_ids(scenario.get("relevant_chunk_ids"))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CROPSTATE retrieval baselines.")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--mode", choices=["research", "production", "all"], default="research")
    parser.add_argument("--embedding-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--embedding-backend", choices=("sentence-transformers", "onnx"),
                        default="sentence-transformers", help="ONNX FP32 is an explicit alternative execution backend.")
    parser.add_argument("--onnx-model-dir", default=None, help="Pinned model snapshot path for ONNX backend.")
    parser.add_argument("--query-language", choices=("en", "vi"), default="en",
                        help="Use English lexical queries for the English IRRI corpus; legacy VI is explicit.")
    parser.add_argument("--depth", type=int, default=50)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--source-weight", type=float, default=0.0,
                        help="Authority bonus applied only to soft rerankers; default zero to isolate stage gating.")
    parser.add_argument("--fixed-beta", type=float, default=0.20,
                        help="Fixed-soft weight; tune on validation scenarios, not test.")
    parser.add_argument("--include-general-fallback", action="store_true",
                        help="Include general_crop_care chunks; evaluate their relevance explicitly if enabled.")
    parser.add_argument("--output", default="results/retrieval_evaluation.json")
    args = parser.parse_args()

    if not 0.0 <= args.source_weight < 1.0 or not 0.0 <= args.fixed_beta <= 1.0 - args.source_weight:
        parser.error("source-weight and fixed-beta must be non-negative and sum to <= 1")
    chunks = load_knowledge_chunks(args.corpus, mode=args.mode)
    chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
    encoder = None
    encoder_metadata = {"backend": args.embedding_backend, "model_id": args.embedding_model}
    if args.embedding_backend == "onnx":
        if args.embedding_model != "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2":
            parser.error("The ONNX adapter is pinned to multilingual MiniLM-L12-v2 only.")
        from cropstate.onnx_encoder import ONNXSentenceEncoder
        encoder = ONNXSentenceEncoder(model_dir=args.onnx_model_dir)
        encoder_metadata.update({
            "model_sha256": encoder.model_sha256,
            "tokenizer_sha256": encoder.tokenizer_sha256,
            "max_seq_length": 128,
            "pooling": "attention_masked_mean_plus_l2",
            "note": "Same pretrained model weights, distinct ONNX FP32 execution backend; may differ numerically from PyTorch.",
        })
    retriever = HybridRetriever(chunks, embedding_model=args.embedding_model, encoder=encoder)
    scenarios = read_scenarios(Path(args.scenarios))
    rows: list[dict[str, Any]] = []

    for scenario_index, scenario in enumerate(scenarios, start=1):
        scenario_id = scenario.get("scenario_id") or f"scenario_{scenario_index:04d}"
        topic = str(scenario["topic"])
        true_stage = normalize_stage(str(scenario["ground_truth_stage"]))
        predicted_stage = normalize_stage(str(scenario.get("predicted_stage") or true_stage))
        belief = parse_belief(scenario.get("stage_belief"), predicted_stage)
        # Zero is a valid low-confidence value; do not turn it into belief.max().
        confidence_raw = scenario.get("confidence")
        confidence = float(confidence_raw) if confidence_raw not in (None, "") else float(belief.max())
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Invalid routing confidence for scenario {scenario_id}: {confidence!r}")
        # B0/B2/B3/P share a stage-free query; only B1 (query expansion) gets a stage-bearing query.
        base_query = str(scenario.get("query") or build_topic_query(topic, language=args.query_language))
        expanded_query = build_topic_query(topic, predicted_stage, language=args.query_language)
        relevance = parse_relevance(scenario)
        relevant = {document_id for document_id, grade in relevance.items() if grade > 0}
        incompatible = parse_ids(scenario.get("incompatible_chunk_ids", scenario.get("nonmatching_chunk_ids")))

        bm25_ranked, dense_ranked = retriever.retrieve(
            base_query, depth=args.depth, topic=topic,
            include_general_fallback=args.include_general_fallback,
        )
        base_scores = minmax(reciprocal_rank_fusion([bm25_ranked, dense_ranked]))
        expanded_bm25, expanded_dense = retriever.retrieve(
            expanded_query, depth=args.depth, topic=topic,
            include_general_fallback=args.include_general_fallback,
        )
        expanded_scores = minmax(reciprocal_rank_fusion([expanded_bm25, expanded_dense]))
        methods = {
            "B0_ungated": sorted(base_scores, key=base_scores.get, reverse=True),
            "B1_query_expansion": sorted(expanded_scores, key=expanded_scores.get, reverse=True),
            "B2_hard_filter": hard_filter(base_scores, chunk_map, int(np.argmax(belief))),
            "B3_fixed_soft": rerank(
                base_scores, chunk_map, belief, confidence,
                source_weight=args.source_weight, fixed_beta=args.fixed_beta,
            ),
            "P_adaptive_soft": rerank(
                base_scores, chunk_map, belief, confidence,
                source_weight=args.source_weight,
            ),
            "oracle_reference": rerank(
                base_scores, chunk_map, one_hot(true_stage), 1.0,
                source_weight=args.source_weight,
            ),
        }
        for method, ranking in methods.items():
            precision, recall = precision_recall_at_k(ranking, relevant, args.k)
            rows.append({
                "scenario_id": scenario_id,
                "topic": topic,
                "ground_truth_stage": true_stage,
                "predicted_stage": predicted_stage,
                "method": method,
                "precision_at_k": precision,
                "recall_at_k": recall,
                "ndcg_at_k": ndcg_at_k(ranking, relevance, args.k),
                "sirr_at_k": sirr_at_k(ranking, incompatible, args.k),
                "returned_at_k": min(len(ranking), args.k),
                "top_k_complete": int(len(ranking) >= args.k),
                "top_ids": ranking[: args.k],
            })

    summary: dict[str, dict[str, float]] = {}
    for method in sorted({row["method"] for row in rows}):
        subset = [row for row in rows if row["method"] == method]
        summary[method] = {
            metric: float(np.mean([row[metric] for row in subset]))
            for metric in ["precision_at_k", "recall_at_k", "ndcg_at_k", "sirr_at_k", "returned_at_k", "top_k_complete"]
        }
    payload = {
        "k": args.k,
        "scenario_count": len(scenarios),
        "evaluation_config": {
            "source_weight": args.source_weight,
            "fixed_beta": args.fixed_beta,
            "include_general_fallback": args.include_general_fallback,
            "embedding_model": args.embedding_model,
            "embedding_backend": encoder_metadata,
            "depth": args.depth,
            "mode": args.mode,
            "query_language": args.query_language,
            "english_corpus": True,
        },
        "summary": summary, "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
