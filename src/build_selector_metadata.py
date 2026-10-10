"""Build gold-free R0/R1 selector metadata from existing report candidates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .data import iter_cells
from .methods import quality_score


def load_report(root: Path, run: str, crisis: str, cell_id: str) -> dict:
    path = root / run / crisis / f"{cell_id}.report.json"
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--r0-name", default="R0c")
    parser.add_argument("--r1-name", default="R1c")
    args = parser.parse_args()

    counts = Counter()
    for cell in iter_cells(args.data):
        crisis = cell.cell_id.split(".")[0]
        r0 = load_report(args.source, args.r0_name, crisis, cell.cell_id)
        r1 = load_report(args.source, args.r1_name, crisis, cell.cell_id)
        score0, features0 = quality_score(r0, cell)
        score1, features1 = quality_score(r1, cell)
        winner = "R1" if score1 > score0 else "R0"
        counts[winner] += 1
        payload = {
            "method": "gold_free_compliant_report_selector",
            "winner": winner,
            "candidate_scores": {"R0": score0, "R1": score1},
            "candidate_features": {"R0": features0, "R1": features1},
        }
        path = args.out / crisis / f"{cell.cell_id}.report.meta.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"cells": sum(counts.values()), "gold_free_winners": counts}, indent=2))


if __name__ == "__main__":
    main()
