"""Measure the gold-dependent per-cell selection upper bound for two runs.

The result is diagnostic only: the oracle reads official Primary scores and is
therefore not a legal submission system.  It answers whether improving a
gold-free selector between two already-generated routes can still be valuable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .official_score import aggregate_rows


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


def build_oracle(baseline: dict, candidate: dict, baseline_name: str, candidate_name: str) -> dict:
    baseline_cells = {row["cell_id"]: row for row in baseline["cells"]}
    candidate_cells = {row["cell_id"]: row for row in candidate["cells"]}
    if baseline_cells.keys() != candidate_cells.keys():
        missing_candidate = sorted(baseline_cells.keys() - candidate_cells.keys())
        missing_baseline = sorted(candidate_cells.keys() - baseline_cells.keys())
        raise ValueError(
            f"cell sets differ; missing from candidate={missing_candidate}, "
            f"missing from baseline={missing_baseline}"
        )

    selected_rows = []
    selections = {}
    candidate_wins = 0
    baseline_wins = 0
    ties = 0
    for cell_id in sorted(baseline_cells):
        base_row = baseline_cells[cell_id]
        candidate_row = candidate_cells[cell_id]
        if candidate_row["primary"] > base_row["primary"]:
            selected = candidate_row
            winner = candidate_name
            candidate_wins += 1
        elif candidate_row["primary"] < base_row["primary"]:
            selected = base_row
            winner = baseline_name
            baseline_wins += 1
        else:
            selected = base_row
            winner = baseline_name
            ties += 1
        selected_rows.append(dict(selected))
        selections[cell_id] = {
            "winner": winner,
            "baseline_primary": base_row["primary"],
            "candidate_primary": candidate_row["primary"],
            "oracle_gain": selected["primary"] - base_row["primary"],
        }

    macro, crises, windows = aggregate_rows(selected_rows, METRICS)
    baseline_macro = baseline["macro_by_crisis"]
    return {
        "diagnostic_only": True,
        "selection_uses_gold": True,
        "baseline": baseline_name,
        "candidate": candidate_name,
        "counts": {
            baseline_name: baseline_wins,
            candidate_name: candidate_wins,
            "ties": ties,
            "cells": len(selected_rows),
        },
        "baseline_macro": baseline_macro,
        "oracle_macro": macro,
        "oracle_delta_vs_baseline": {
            metric: macro[metric] - baseline_macro[metric] for metric in METRICS
        },
        "crises": crises,
        "windows": windows,
        "selections": selections,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-name", default="R0")
    parser.add_argument("--candidate-name", default="R1")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    result = build_oracle(baseline, candidate, args.baseline_name, args.candidate_name)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "counts": result["counts"],
        "baseline_primary": result["baseline_macro"]["primary"],
        "oracle_primary": result["oracle_macro"]["primary"],
        "oracle_primary_gain": result["oracle_delta_vs_baseline"]["primary"],
    }, indent=2))


if __name__ == "__main__":
    main()
