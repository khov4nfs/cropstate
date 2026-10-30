"""Export full empirical selective-risk curves for the original RIVF'26 manuscript.

Usage (repository root):
    PYTHONPATH=src:. .venv/bin/python scripts/experiments/export_risk_coverage.py

The 263 *test* logits determine the curves; temperature and the combined-
confidence operating threshold are selected using validation data only.
Both curves represent ranks of all test crops, not interpolated measurements.
In particular, the old 0.54/0.099 point is from max-softmax confidence;
the validation threshold 0.524541 is from combined entropy/margin confidence.
Do NOT plot both points on a single unlabeled curve or reuse the historical
coarse threshold-sweep AURC=0.126 for exact all-coverage AURC.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from cropstate.confidence import combined_confidence
from scripts.experiments.run_retrieval_ablations import choose_abstention_threshold


DEFAULT_NPZ = Path("CROPSTATE_RESULTS/novelty/logits/resnet18.npz")
DEFAULT_CALIBRATION = Path("CROPSTATE_RESULTS/novelty/calibration.json")
DEFAULT_OUTPUT = Path(
    "CropState_Vision__A_Reliability_Oriented_Evaluation_of_Six_Stage_Rice_"
    "Growth_Classification_and_Confidence_Aware_Stage_Gated_Retrieval/images"
)


def probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Stable temperature-scaled probabilities, avoiding extra dependencies."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6 or not np.isfinite(values).all():
        raise ValueError("Expected finite N x 6 logits")
    values = values / temperature
    values -= values.max(axis=1, keepdims=True)
    ex = np.exp(values)
    return ex / ex.sum(axis=1, keepdims=True)


def exact_curve(scores: np.ndarray, correct: np.ndarray) -> list[dict[str, float | int]]:
    """All non-empty top-k confidence prefixes; tie ordering is stable.

    In the rare event of tied confidence, the interpolated points inside a
    tied block depend on original example ordering. We therefore sort
    deterministically and report the full ranked-prefix empirical curve.
    """
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    n = correct.size
    if n == 0 or scores.shape != (n,) or not np.isfinite(scores).all():
        raise ValueError("Scores/correct must be nonempty aligned 1-D arrays")
    order = np.argsort(-scores, kind="stable")
    mistakes = np.cumsum(~correct[order])
    return [
        {
            "coverage": float(k / n),
            "risk": float(mistakes[k - 1] / k),
            "retained": k,
            "confidence_threshold": float(scores[order[k - 1]]),
        }
        for k in range(1, n + 1)
    ]


def exact_aurc(curve: list[dict[str, float | int]]) -> float:
    """Integral with origin risk 0, using equal-width right-endpoint bins.

    This is the mean of risks over all 1..N ranked prefixes. It covers the
    entire [0, 1] range, unlike a coarse 14-threshold trapezoid.
    """
    if not curve:
        raise ValueError("Cannot integrate an empty curve")
    return float(np.mean([float(row["risk"]) for row in curve]))


def write_curve(path: Path, curve: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, ["coverage", "risk", "retained", "confidence_threshold"]
        )
        writer.writeheader()
        for row in curve:
            writer.writerow({
                "coverage": f'{row["coverage"]:.12f}',
                "risk": f'{row["risk"]:.12f}',
                "retained": row["retained"],
                "confidence_threshold": f'{row["confidence_threshold"]:.12f}',
            })


def selective_point(scores: np.ndarray, correct: np.ndarray, cutoff: float) -> dict:
    kept = scores >= cutoff
    if not kept.any():
        raise ValueError("No retained samples at the requested threshold")
    return {
        "threshold": float(cutoff),
        "retained": int(kept.sum()),
        "coverage": float(kept.mean()),
        "risk": float(1 - correct[kept].mean()),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(npz: Path, calibration: Path, output_dir: Path, summary_path: Path) -> dict:
    settings = json.loads(calibration.read_text(encoding="utf-8"))
    temperature = float(settings["resnet18"]["temperature"])
    with np.load(npz, allow_pickle=False) as source:
        validation_logits = np.array(source["validation_logits"], dtype=np.float64)
        test_logits = np.array(source["test_logits"], dtype=np.float64)
        test_labels = np.array(source["test_labels"], dtype=np.int64)
    if len(validation_logits) != 237 or len(test_logits) != 263:
        raise ValueError("Unexpected split sizes: expected 237 validation and 263 test")
    if np.any((test_labels < 0) | (test_labels >= 6)):
        raise ValueError("Unexpected 6-class weak labels")

    threshold = choose_abstention_threshold(validation_logits, temperature)
    beliefs = probabilities(test_logits, temperature)
    correct = beliefs.argmax(axis=1) == test_labels
    combined = np.array([combined_confidence(row) for row in beliefs])
    maxsoftmax = beliefs.max(axis=1)

    main_curve = exact_curve(combined, correct)
    comparison_curve = exact_curve(maxsoftmax, correct)
    main_point = selective_point(combined, correct, threshold)
    legacy_point = selective_point(maxsoftmax, correct, 0.8)
    full_point = {"coverage": 1.0, "risk": float((~correct).mean())}

    # Guard exact study claims. These are checks, not fabricated points.
    if main_point["retained"] != 189 or legacy_point["retained"] != 142:
        raise ValueError("Reproduction mismatch in historical coverage markers")
    if not np.isclose(main_point["risk"], 23 / 189, atol=1e-9):
        raise ValueError("Validation-selected weak-label selective risk mismatch")
    if not np.isclose(legacy_point["risk"], 14 / 142, atol=1e-9):
        raise ValueError("Historical max-softmax illustrative point mismatch")
    if not np.isclose(full_point["risk"], 54 / 263, atol=1e-9):
        raise ValueError("Full test risk mismatch")

    file_main = output_dir / "risk_coverage_points.csv"
    file_comparison = output_dir / "risk_coverage_maxsoftmax.csv"
    write_curve(file_main, main_curve)
    write_curve(file_comparison, comparison_curve)
    summary = {
        "n_validation": len(validation_logits),
        "n_test": len(test_logits),
        "temperature_from_validation": temperature,
        "combined_confidence": "0.6*(1-H(p)/ln(6)) + 0.4*(top1-top2)",
        "validation_target_coverage": 0.7,
        "validation_selected_point": main_point,
        "historical_maxsoftmax_point": legacy_point,
        "full_coverage_point": full_point,
        "exact_aurc_combined": exact_aurc(main_curve),
        "exact_aurc_maxsoftmax": exact_aurc(comparison_curve),
        "aurc_definition": "mean selective risk over all nonempty sorted prefixes, 1..N",
        "source_npz_sha256": sha256(npz),
        "calibration_json_sha256": sha256(calibration),
        "main_csv_sha256": sha256(file_main),
        "comparison_csv_sha256": sha256(file_comparison),
        "metadata_note": (
            "Weak labels, not agronomist BBCH ground truth. "
            "The older 0.126 AURC came from the coarse threshold sweep "
            "and is not comparable without recomputing from all prefixes."
        ),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-path", type=Path,
                        default=Path("CROPSTATE_RESULTS/novelty/risk_coverage_full_summary.json"))
    args = parser.parse_args()
    print(json.dumps(run(args.npz, args.calibration, args.output_dir, args.summary_path),
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
