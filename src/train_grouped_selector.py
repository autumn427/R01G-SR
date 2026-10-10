"""Train and evaluate a gold-free R0/R1 selector with nested crisis CV.

Gold scores are used only as training labels. Every reported out-of-fold
decision for a crisis is produced by a model that was fitted without any cell
from that crisis.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .analyze_selector_diagnostics import load_selector_metadata
from .analyze_train_routes import compare_payloads
from .official_score import aggregate_rows
from .selector_features import inference_feature_rows


METRICS = (
    "primary",
    "bertscore_f1",
    "bleurt",
    "rouge1",
    "rouge2",
    "rougeL",
    "alignment_f1",
    "structure_recall",
)
C_VALUES = (0.03, 0.1, 0.3, 1.0, 3.0)
THRESHOLDS = (0.4, 0.5, 0.6)


def hierarchy_weights(rows: list[dict]) -> np.ndarray:
    crises = {row["crisis"] for row in rows}
    windows_by_crisis: dict[str, set[str]] = defaultdict(set)
    counts = Counter()
    for row in rows:
        windows_by_crisis[row["crisis"]].add(row["window"])
        counts[(row["crisis"], row["window"])] += 1
    weights = []
    for row in rows:
        crisis = row["crisis"]
        window = row["window"]
        weights.append(
            1.0 / len(crises) / len(windows_by_crisis[crisis]) / counts[(crisis, window)]
        )
    result = np.asarray(weights, dtype=float)
    return result / result.mean()


def feature_rows(data_root: Path, metadata: dict[str, dict], r0: dict, r1: dict) -> tuple[list[str], list[dict]]:
    """Attach official score labels to the otherwise gold-free feature rows."""
    r0_cells = {row["cell_id"]: row for row in r0["cells"]}
    r1_cells = {row["cell_id"]: row for row in r1["cells"]}
    expected = set(r0_cells) & set(r1_cells)
    if set(metadata) != expected:
        raise ValueError(
            "selector metadata mismatch; "
            f"missing={sorted(expected - set(metadata))}, extra={sorted(set(metadata) - expected)}"
        )
    names, rows = inference_feature_rows(data_root, metadata)
    for row in rows:
        cell_id = row["cell_id"]
        delta = r1_cells[cell_id]["primary"] - r0_cells[cell_id]["primary"]
        row.update({
            "label": int(delta > 0),
            "tie": delta == 0,
            "primary_delta": delta,
        })
    return names, rows


def fit_model(rows: list[dict], c_value: float) -> tuple[StandardScaler, LogisticRegression]:
    decided = [row for row in rows if not row["tie"]]
    x = np.asarray([row["x"] for row in decided], dtype=float)
    y = np.asarray([row["label"] for row in decided], dtype=int)
    base_weights = hierarchy_weights(decided)
    regret_weights = np.asarray([max(abs(row["primary_delta"]), 0.005) for row in decided])
    sample_weights = base_weights * regret_weights
    sample_weights = sample_weights / sample_weights.mean()
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=c_value, max_iter=2000, solver="lbfgs", random_state=20260826)
    model.fit(scaler.transform(x), y, sample_weight=sample_weights)
    return scaler, model


def predict_probabilities(rows: list[dict], scaler: StandardScaler, model: LogisticRegression) -> dict[str, float]:
    x = np.asarray([row["x"] for row in rows], dtype=float)
    probabilities = model.predict_proba(scaler.transform(x))[:, 1]
    return {row["cell_id"]: float(probability) for row, probability in zip(rows, probabilities)}


def select_score_payload(
    rows: list[dict], probabilities: dict[str, float], threshold: float, r0: dict, r1: dict
) -> dict:
    r0_cells = {row["cell_id"]: row for row in r0["cells"]}
    r1_cells = {row["cell_id"]: row for row in r1["cells"]}
    selected_rows = [
        dict(r1_cells[row["cell_id"]] if probabilities[row["cell_id"]] >= threshold else r0_cells[row["cell_id"]])
        for row in rows
    ]
    macro, crises, windows = aggregate_rows(selected_rows, METRICS)
    return {"macro_by_crisis": macro, "crises": crises, "windows": windows, "cells": selected_rows}


def choose_hyperparameters(rows: list[dict], r0: dict, r1: dict) -> tuple[float, float, list[dict]]:
    crises = sorted({row["crisis"] for row in rows})
    trials = []
    for c_value in C_VALUES:
        probabilities = {}
        for held_out in crises:
            train = [row for row in rows if row["crisis"] != held_out]
            validation = [row for row in rows if row["crisis"] == held_out]
            scaler, model = fit_model(train, c_value)
            probabilities.update(predict_probabilities(validation, scaler, model))
        for threshold in THRESHOLDS:
            payload = select_score_payload(rows, probabilities, threshold, r0, r1)
            trials.append({
                "c": c_value,
                "threshold": threshold,
                "primary": payload["macro_by_crisis"]["primary"],
            })
    winner = max(trials, key=lambda trial: (trial["primary"], -trial["c"], -abs(trial["threshold"] - 0.5)))
    return winner["c"], winner["threshold"], trials


def nested_oof(rows: list[dict], r0: dict, r1: dict) -> tuple[dict[str, float], dict[str, dict]]:
    probabilities = {}
    fold_metadata = {}
    for held_out in sorted({row["crisis"] for row in rows}):
        outer_train = [row for row in rows if row["crisis"] != held_out]
        outer_test = [row for row in rows if row["crisis"] == held_out]
        c_value, threshold, trials = choose_hyperparameters(outer_train, r0, r1)
        scaler, model = fit_model(outer_train, c_value)
        fold_probabilities = predict_probabilities(outer_test, scaler, model)
        probabilities.update(fold_probabilities)
        fold_metadata[held_out] = {
            "c": c_value,
            "threshold": threshold,
            "inner_best_primary": max(trial["primary"] for trial in trials),
            "test_cells": len(outer_test),
        }
    return probabilities, fold_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--r0", type=Path, required=True)
    parser.add_argument("--r1", type=Path, required=True)
    parser.add_argument("--selector-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--score-out", type=Path, required=True)
    args = parser.parse_args()

    r0 = json.loads(args.r0.read_text(encoding="utf-8"))
    r1 = json.loads(args.r1.read_text(encoding="utf-8"))
    metadata = load_selector_metadata(args.selector_root)
    feature_names, rows = feature_rows(args.data, metadata, r0, r1)
    oof_probabilities, folds = nested_oof(rows, r0, r1)

    # Each outer fold has its own threshold, selected without the held-out crisis.
    selected_rows = []
    r0_cells = {row["cell_id"]: row for row in r0["cells"]}
    r1_cells = {row["cell_id"]: row for row in r1["cells"]}
    decisions = {}
    for row in rows:
        threshold = folds[row["crisis"]]["threshold"]
        probability = oof_probabilities[row["cell_id"]]
        winner = "R1" if probability >= threshold else "R0"
        selected_rows.append(dict(r1_cells[row["cell_id"]] if winner == "R1" else r0_cells[row["cell_id"]]))
        decisions[row["cell_id"]] = {
            "winner": winner,
            "probability_r1": probability,
            "threshold": threshold,
            "actual_primary_delta_r1_minus_r0": row["primary_delta"],
        }
    macro, crises, windows = aggregate_rows(selected_rows, METRICS)
    score_payload = {"macro_by_crisis": macro, "crises": crises, "windows": windows, "cells": selected_rows}
    comparison = compare_payloads(r0, score_payload)

    final_c, final_threshold, final_trials = choose_hyperparameters(rows, r0, r1)
    scaler, model = fit_model(rows, final_c)
    result = {
        "exploratory": True,
        "validation": "nested leave-one-crisis-out",
        "features": feature_names,
        "folds": folds,
        "oof_comparison": comparison,
        "oof_decisions": decisions,
        "final_model": {
            "c": final_c,
            "threshold": final_threshold,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "coefficients": model.coef_[0].tolist(),
            "intercept": float(model.intercept_[0]),
            "selection_trials": final_trials,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.score_out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.score_out.write_text(json.dumps(score_payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "oof_primary": macro["primary"],
        "oof_primary_gain": comparison["delta_vs_baseline"]["primary"],
        "crisis_wins": comparison["primary_wins"],
        "gate": comparison["gate"],
        "final_c": final_c,
        "final_threshold": final_threshold,
    }, indent=2))


if __name__ == "__main__":
    main()
