"""Apply a frozen gold-free R0/R1 selector to candidate reports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .analyze_selector_diagnostics import load_selector_metadata
from .selector_features import inference_feature_rows


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def frozen_probabilities(feature_names: list[str], rows: list[dict], payload: dict) -> dict[str, float]:
    if feature_names != payload["features"]:
        raise ValueError("feature schema does not match the frozen selector")
    model = payload["final_model"]
    mean = np.asarray(model["scaler_mean"], dtype=float)
    scale = np.asarray(model["scaler_scale"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    if not (len(mean) == len(scale) == len(coefficients) == len(feature_names)):
        raise ValueError("frozen model dimensions do not match its feature schema")
    probabilities = {}
    for row in rows:
        normalized = (np.asarray(row["x"], dtype=float) - mean) / scale
        logit = float(model["intercept"] + np.dot(coefficients, normalized))
        probabilities[row["cell_id"]] = sigmoid(logit)
    return probabilities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--selector-metadata", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.model.read_text(encoding="utf-8"))
    metadata = load_selector_metadata(args.selector_metadata)
    feature_names, rows = inference_feature_rows(args.data, metadata)
    probabilities = frozen_probabilities(feature_names, rows, payload)
    threshold = float(payload["final_model"]["threshold"])

    counts = {"R0": 0, "R1": 0}
    for row in rows:
        cell_id = row["cell_id"]
        crisis = row["crisis"]
        winner = "R1" if probabilities[cell_id] >= threshold else "R0"
        source_path = args.source / winner / crisis / f"{cell_id}.report.json"
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        destination_dir = args.out / crisis
        destination_dir.mkdir(parents=True, exist_ok=True)
        report = json.loads(source_path.read_text(encoding="utf-8"))
        (destination_dir / f"{cell_id}.report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        decision = {
            "method": "frozen grouped logistic R0/R1 selector",
            "winner": winner,
            "probability_r1": probabilities[cell_id],
            "threshold": threshold,
            "training_c": payload["final_model"]["c"],
            "source_report": str(source_path),
            "gold_free_inference": True,
        }
        (destination_dir / f"{cell_id}.report.meta.json").write_text(
            json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        counts[winner] += 1

    print(json.dumps({"cells": len(rows), "selections": counts, "threshold": threshold}, indent=2))


if __name__ == "__main__":
    main()
