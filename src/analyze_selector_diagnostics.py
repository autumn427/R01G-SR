"""Diagnose how much of a two-route oracle a gold-free selector recovers."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .analyze_pair_oracle import build_oracle


FEATURES = (
    "n_scored_bullets",
    "represented_subsections",
    "mean_lexical_support",
    "duplicate_bullets",
    "unsupported_number_bullets",
)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def load_selector_metadata(root: Path) -> dict[str, dict]:
    metadata = {}
    for path in root.rglob("*.report.meta.json"):
        cell_id = path.name.removesuffix(".report.meta.json")
        metadata[cell_id] = json.loads(path.read_text(encoding="utf-8"))
    return metadata


def analyze_selector(r0: dict, r1: dict, selected: dict, metadata: dict[str, dict]) -> dict:
    r0_cells = {row["cell_id"]: row for row in r0["cells"]}
    r1_cells = {row["cell_id"]: row for row in r1["cells"]}
    common = sorted(r0_cells.keys() & r1_cells.keys())
    if set(common) != set(metadata):
        missing = sorted(set(common) - set(metadata))
        extra = sorted(set(metadata) - set(common))
        raise ValueError(f"selector metadata mismatch; missing={missing}, extra={extra}")

    records = []
    grouped = defaultdict(list)
    feature_deltas = {feature: defaultdict(list) for feature in FEATURES}
    correct = incorrect = ties = 0
    for cell_id in common:
        base_primary = r0_cells[cell_id]["primary"]
        candidate_primary = r1_cells[cell_id]["primary"]
        actual_delta = candidate_primary - base_primary
        actual_winner = "R1" if actual_delta > 0 else "R0" if actual_delta < 0 else "tie"
        meta = metadata[cell_id]
        predicted = meta["winner"]
        if actual_winner == "tie":
            ties += 1
        elif predicted == actual_winner:
            correct += 1
        else:
            incorrect += 1

        selected_primary = candidate_primary if predicted == "R1" else base_primary
        oracle_primary = max(base_primary, candidate_primary)
        selector_margin = meta["candidate_scores"]["R1"] - meta["candidate_scores"]["R0"]
        crisis = cell_id.split(".")[0]
        window = cell_id.split(".")[1]
        record = {
            "cell_id": cell_id,
            "crisis": crisis,
            "window": window,
            "actual_winner": actual_winner,
            "predicted_winner": predicted,
            "actual_primary_delta_r1_minus_r0": actual_delta,
            "selector_score_margin_r1_minus_r0": selector_margin,
            "regret": oracle_primary - selected_primary,
        }
        records.append(record)
        grouped[crisis].append(record)

        if actual_winner != "tie":
            for feature in FEATURES:
                delta = (
                    meta["candidate_features"]["R1"][feature]
                    - meta["candidate_features"]["R0"][feature]
                )
                feature_deltas[feature][actual_winner].append(delta)

    oracle = build_oracle(r0, r1, "R0", "R1")
    baseline_primary = r0["macro_by_crisis"]["primary"]
    selector_gain = selected["macro_by_crisis"]["primary"] - baseline_primary
    oracle_gain = oracle["oracle_delta_vs_baseline"]["primary"]
    per_crisis = {}
    for crisis, crisis_records in sorted(grouped.items()):
        decided = [row for row in crisis_records if row["actual_winner"] != "tie"]
        per_crisis[crisis] = {
            "cells": len(crisis_records),
            "accuracy": mean([
                float(row["predicted_winner"] == row["actual_winner"]) for row in decided
            ]),
            "mean_regret": mean([row["regret"] for row in crisis_records]),
        }

    return {
        "diagnostic_only": True,
        "cells": len(records),
        "classification": {
            "correct": correct,
            "incorrect": incorrect,
            "ties": ties,
            "accuracy_excluding_ties": correct / (correct + incorrect),
        },
        "primary": {
            "baseline": baseline_primary,
            "selector": selected["macro_by_crisis"]["primary"],
            "oracle": oracle["oracle_macro"]["primary"],
            "selector_gain": selector_gain,
            "oracle_gain": oracle_gain,
            "fraction_of_oracle_gain_recovered": selector_gain / oracle_gain if oracle_gain else None,
        },
        "mean_cell_regret": mean([record["regret"] for record in records]),
        "feature_delta_by_actual_winner": {
            feature: {
                winner: mean(values) for winner, values in sorted(groups.items())
            }
            for feature, groups in feature_deltas.items()
        },
        "per_crisis": per_crisis,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r0", type=Path, required=True)
    parser.add_argument("--r1", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--selector-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    result = analyze_selector(
        json.loads(args.r0.read_text(encoding="utf-8")),
        json.loads(args.r1.read_text(encoding="utf-8")),
        json.loads(args.selected.read_text(encoding="utf-8")),
        load_selector_metadata(args.selector_root),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "classification": result["classification"],
        "primary": result["primary"],
        "mean_cell_regret": result["mean_cell_regret"],
    }, indent=2))


if __name__ == "__main__":
    main()
