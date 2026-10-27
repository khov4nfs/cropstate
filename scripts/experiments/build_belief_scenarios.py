"""Build closed-loop retrieval scenarios from REAL vision stage-beliefs (Tier B#4).

This is the bridge the paper says it cannot build without agronomists. It avoids
human relevance judgments by deriving relevance PROGRAMMATICALLY from each chunk's
stage_compatibility vector:
  - relevant chunk   = topic chunk with compatibility[true_stage] >= rel_threshold
  - incompatible chunk = topic chunk with compatibility[true_stage] == 0

Each test image contributes one scenario per stage-discriminative topic, carrying
its real six-way belief and confidence. Feeding this to evaluate_retrieval.py shows
whether belief-driven soft gating lowers the stage-incompatible retrieval rate
(SIRR) toward the true-stage oracle — the closed loop, minus the humans.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from cropstate.constants import STAGE_NAMES
from cropstate.confidence import combined_confidence
from cropstate.knowledge import load_knowledge_chunks


def softmax_np(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def prediction_from_logits(logits, temperature=1.0, confidence_method="combined"):
    """Return the calibrated belief consumed by retrieval, predicted stage, and routing confidence.

    The same calibrated probabilities must drive both the stage score and the confidence
    mixture. "top1" is retained only as an explicit historical ablation.
    """
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and strictly positive")
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != (len(STAGE_NAMES),) or not np.all(np.isfinite(logits)):
        raise ValueError("logits must be a finite vector with one value per stage")
    calibrated = softmax_np((logits / temperature)[None, :])[0]
    if confidence_method == "combined":
        confidence = combined_confidence(calibrated)
    elif confidence_method == "top1":
        confidence = float(calibrated.max())
    else:
        raise ValueError("confidence_method must be 'combined' or 'top1'")
    return calibrated, int(calibrated.argmax()), float(confidence)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits-index", default="CROPSTATE_RESULTS/novelty/logits/index.json")
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--corpus", default="CROPSTATE_KNOWLEDGE_BASE/chunks/rice_knowledge_irri_en_nonrestricted.jsonl")
    parser.add_argument("--calibration", default="CROPSTATE_RESULTS/novelty/calibration.json",
                        help="If present, also emit a temperature-calibrated belief column.")
    parser.add_argument("--confidence-method", choices=("combined", "top1"), default="combined",
                        help="Confidence used by the adaptive ranker; top1 is a legacy ablation.")
    parser.add_argument("--rel-threshold", type=float, default=0.5)
    parser.add_argument("--min-relevant", type=int, default=3)
    parser.add_argument("--min-incompatible", type=int, default=3)
    parser.add_argument("--output", default="CROPSTATE_RESULTS/novelty/belief_scenarios.csv")
    args = parser.parse_args()

    index = json.loads(Path(args.logits_index).read_text())
    data = np.load(index[args.model]["path"], allow_pickle=True)
    logits = data["test_logits"].astype(np.float64)
    labels = data["test_labels"].astype(int)
    image_ids = [str(x) for x in data["test_image_ids"]]
    beliefs = softmax_np(logits)

    temperature = 1.0
    calib_path = Path(args.calibration)
    if calib_path.exists():
        calib = json.loads(calib_path.read_text())
        temperature = float(calib.get(args.model, {}).get("temperature", 1.0))
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Calibration temperature must be finite and positive")

    chunks = load_knowledge_chunks(args.corpus, mode="research")
    topics = sorted({c.topic for c in chunks})

    # Per (topic, true stage): relevant and incompatible chunk id sets.
    def sets_for(topic, stage_idx):
        rel, inc = [], []
        for c in chunks:
            if c.topic != topic:
                continue
            comp = c.stage_compatibility[stage_idx]
            if comp >= args.rel_threshold:
                rel.append(c.chunk_id)
            elif comp == 0.0:
                inc.append(c.chunk_id)
        return rel, inc

    rows = []
    for i, (img_id, y) in enumerate(zip(image_ids, labels)):
        raw_belief = beliefs[i]
        belief, pred, confidence = prediction_from_logits(
            logits[i], temperature, args.confidence_method
        )
        for topic in topics:
            rel, inc = sets_for(topic, int(y))
            if len(rel) < args.min_relevant or len(inc) < args.min_incompatible:
                continue
            rows.append({
                "scenario_id": f"{img_id}__{topic}",
                "topic": topic,
                "ground_truth_stage": STAGE_NAMES[int(y)],
                "predicted_stage": STAGE_NAMES[pred],
                "stage_belief": json.dumps([round(float(x), 6) for x in belief]),
                "raw_stage_belief": json.dumps([round(float(x), 6) for x in raw_belief]),
                "calibrated_stage_belief": json.dumps([round(float(x), 6) for x in belief]),
                "confidence": round(confidence, 6),
                "confidence_method": args.confidence_method,
                "calibration_temperature": round(float(temperature), 6),
                "relevant_chunk_ids": "|".join(rel),
                "incompatible_chunk_ids": "|".join(inc),
            })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No answerable (image, topic) retrieval scenarios were generated")
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[scenarios] wrote {len(rows)} scenarios over {len({r['topic'] for r in rows})} topics, "
          f"{len(image_ids)} test images, temperature={temperature:.3f}, "
          f"confidence={args.confidence_method}")


if __name__ == "__main__":
    main()
