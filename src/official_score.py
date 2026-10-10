from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path


def mean(values):
    values = [x for x in values if isinstance(x, (int, float))]
    return sum(values) / len(values) if values else None


def rouge_fmeasure(summary: dict, name: str):
    value = summary.get("rouge", {}).get(name)
    return value.get("fmeasure") if isinstance(value, dict) else value


def aggregate_rows(rows: list[dict], keys: tuple[str, ...]) -> tuple[dict, dict, dict]:
    """Apply the official replicate -> window -> crisis -> overall hierarchy."""
    by_crisis_window = defaultdict(lambda: defaultdict(list))
    for row in rows:
        parts = row["cell_id"].split(".")
        if len(parts) < 3:
            continue
        by_crisis_window[row["crisis"]][parts[1]].append(row)
    windows = {
        crisis: {
            window: {key: mean([row[key] for row in items]) for key in keys}
            for window, items in sorted(grouped.items())
        }
        for crisis, grouped in sorted(by_crisis_window.items())
    }
    crises = {
        crisis: {key: mean([values[key] for values in grouped.values()]) for key in keys}
        for crisis, grouped in windows.items()
    }
    macro = {key: mean([values[key] for values in crises.values()]) for key in keys}
    return macro, crises, windows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the official evaluator per released dev cell and macro-average by crisis.")
    parser.add_argument("--python", type=Path, default=Path(".venv-eval/Scripts/python.exe"))
    parser.add_argument("--evaluator", type=Path, default=Path("vendor/lt4cpr-evaluation/shared_task/tools/eval_tools"))
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True, help="One run directory, e.g. outputs/dev/R2")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true", help="Recompute cell scores even when cached results exist.")
    args = parser.parse_args()
    official_script = (args.evaluator / "src/eval_pair.py").resolve()
    compat_script = (Path(__file__).parent / "eval_pair_compat.py").resolve()
    script = compat_script if os.name == "nt" else official_script
    config = (args.evaluator / "config/evaluation.yaml").resolve()
    detail_dir = args.out.parent / (args.out.stem + "_cells")
    evaluator_env = os.environ.copy()
    evaluator_env.update({
        "TF_ENABLE_ONEDNN_OPTS": "0",
        "TF_NUM_INTRAOP_THREADS": "1",
        "TF_NUM_INTEROP_THREADS": "1",
    })
    rows = []
    for pred in sorted(args.pred.rglob("*.report.json")):
        cell_id = pred.name.removesuffix(".report.json")
        matches = list(args.gold.rglob(f"{cell_id}.report.json"))
        if len(matches) != 1:
            continue
        result_path = detail_dir / f"{cell_id}.eval.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        command = [str(args.python.resolve()), str(script), "--gold", str(matches[0].resolve()), "--system", str(pred.resolve()), "--config", str(config), "--out", str(result_path.resolve()), "--quiet"]
        if args.overwrite or not result_path.exists():
            subprocess.run(command, cwd=args.evaluator, env=evaluator_env, check=True)
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        summary = payload["summary"]
        bert = summary["bertscore"].get("f1")
        bleurt = summary["bleurt"].get("score")
        rows.append({
            "cell_id": cell_id, "crisis": cell_id.split(".")[0], "bertscore_f1": bert, "bleurt": bleurt,
            "primary": mean([bert, bleurt]), "rouge1": rouge_fmeasure(summary, "rouge1"),
            "rouge2": rouge_fmeasure(summary, "rouge2"), "rougeL": rouge_fmeasure(summary, "rougeL"),
            "alignment_f1": summary["weighted_alignment"].get("micro_soft_f1"),
            "structure_recall": summary["weighted_alignment"].get("structure_recall"),
        })
        print(cell_id, rows[-1]["primary"], flush=True)
    keys = ("primary", "bertscore_f1", "bleurt", "rouge1", "rouge2", "rougeL", "alignment_f1", "structure_recall")
    macro, crises, windows = aggregate_rows(rows, keys)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"macro_by_crisis": macro, "crises": crises, "windows": windows, "cells": rows}, indent=2), encoding="utf-8")
    print(json.dumps(macro, indent=2))


if __name__ == "__main__":
    main()
