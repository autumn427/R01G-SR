from __future__ import annotations

import argparse
import copy
import json
import platform
import time
from pathlib import Path

from .data import Cell, iter_cells
from .llm import LLMClient
from .methods import consolidate_ledger, extract_ledger, generate_report, prepare_r1, quality_score, raw_lines, validate_report


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def report_path(root: Path, run: str, cell: Cell) -> Path:
    return root / run / cell.path.parent.name / f"{cell.cell_id}.report.json"


def run_cell(client: LLMClient, cell: Cell, requested: set[str], out: Path, overwrite: bool) -> dict:
    summary: dict = {"cell_id": cell.cell_id, "runs": {}}
    groups, compressed, prep = prepare_r1(cell)
    summary["preprocessing"] = prep

    def persist(name: str, report: dict, meta: dict) -> None:
        path = report_path(out, name, cell)
        if overwrite or not path.exists():
            write_json(path, report)
        write_json(path.with_suffix(".meta.json"), meta)
        summary["runs"][name] = meta

    if "R0" in requested:
        started = time.perf_counter()
        report, call = generate_report(client, cell, raw_lines(cell))
        persist("R0", report, {"method": "direct", "wall_seconds": time.perf_counter() - started, "calls": [call]})

    if "R1" in requested:
        started = time.perf_counter()
        report, call = generate_report(client, cell, compressed)
        persist("R1", report, {"method": "filter_dedup", "wall_seconds": time.perf_counter() - started, "calls": [call], **prep})

    needs_ledger = bool(requested & {"R2", "R3", "R4"})
    ledger, map_calls = ([], [])
    if needs_ledger:
        ledger_cache = out / "_cache" / cell.path.parent.name / f"{cell.cell_id}.ledger.json"
        if ledger_cache.exists() and not overwrite:
            ledger = json.loads(ledger_cache.read_text(encoding="utf-8"))
            map_calls = [{"cache_hit": True}]
        else:
            chunk_checkpoints = out / "_cache_chunks" / cell.path.parent.name / cell.cell_id
            ledger, map_calls = extract_ledger(
                client,
                cell,
                compressed,
                checkpoint_dir=chunk_checkpoints,
                reuse_checkpoints=not overwrite,
            )
            write_json(ledger_cache, ledger)
        summary["ledger_facts"] = len(ledger)

    base_report = None
    base_call = None
    if needs_ledger:
        existing_r2 = report_path(out, "R2", cell)
        if "R2" not in requested and not overwrite and existing_r2.exists():
            base_report = json.loads(existing_r2.read_text(encoding="utf-8"))
            base_call = {"cache_hit": True, "source": str(existing_r2)}
        else:
            base_report, base_call = consolidate_ledger(client, cell, ledger, "factual")

    if "R2" in requested:
        persist("R2", base_report, {"method": "map_reduce", "calls": map_calls + [base_call], "ledger_facts": len(ledger), **prep})

    validated = None
    validation_stats = None
    if requested & {"R3", "R4"}:
        validated, validation_stats = validate_report(copy.deepcopy(base_report), cell)

    if "R3" in requested:
        persist("R3", validated, {"method": "map_reduce_validate", "calls": map_calls + [base_call], "validation": validation_stats, "ledger_facts": len(ledger), **prep})

    if "R4" in requested:
        alternative, alt_call = consolidate_ledger(client, cell, ledger, "coverage", temperature=0.25)
        alternative, alt_validation = validate_report(alternative, cell)
        score_a, features_a = quality_score(validated, cell)
        score_b, features_b = quality_score(alternative, cell)
        winner = "factual" if score_a >= score_b else "coverage"
        selected = validated if winner == "factual" else alternative
        persist("R4", selected, {
            "method": "validated_candidate_rerank", "calls": map_calls + [base_call, alt_call], "winner": winner,
            "candidate_scores": {"factual": score_a, "coverage": score_b},
            "candidate_features": {"factual": features_a, "coverage": features_b},
            "validation": {"factual": validation_stats, "coverage": alt_validation}, "ledger_facts": len(ledger), **prep,
        })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", default="R0,R1,R2,R3,R4")
    parser.add_argument("--cells", default="")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="Qwen3-4B-Instruct-2507")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    requested = {x.strip().upper() for x in args.runs.split(",") if x.strip()}
    if not requested <= {"R0", "R1", "R2", "R3", "R4"}:
        raise SystemExit("--runs must contain only R0 through R4")
    selected = {x.strip() for x in args.cells.split(",") if x.strip()} or None
    client = LLMClient(args.base_url, args.model)
    manifest = {"data": str(args.data.resolve()), "runs": sorted(requested), "model": args.model, "platform": platform.platform(), "cells": []}
    for cell in iter_cells(args.data, selected):
        cell_requested = requested if args.overwrite else {run for run in requested if not report_path(args.out, run, cell).exists()}
        if not cell_requested:
            print(f"[{time.strftime('%H:%M:%S')}] {cell.cell_id} (complete; skipped)", flush=True)
            manifest["cells"].append({"cell_id": cell.cell_id, "skipped_complete": True})
            continue
        print(f"[{time.strftime('%H:%M:%S')}] {cell.cell_id} ({','.join(sorted(cell_requested))})", flush=True)
        try:
            manifest["cells"].append(run_cell(client, cell, cell_requested, args.out, args.overwrite))
        except Exception as exc:
            manifest["cells"].append({"cell_id": cell.cell_id, "error": f"{type(exc).__name__}: {exc}"})
            write_json(args.out / "manifest.partial.json", manifest)
            raise
        write_json(args.out / "manifest.partial.json", manifest)
    write_json(args.out / "manifest.json", manifest)


if __name__ == "__main__":
    main()
