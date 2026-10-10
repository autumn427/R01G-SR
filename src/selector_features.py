"""Gold-free R0/R1 selector features shared by training and inference."""

from __future__ import annotations

import math
from pathlib import Path

from .analyze_selector_diagnostics import FEATURES
from .data import iter_cells
from .methods import prepare_r1


def inference_feature_rows(data_root: Path, metadata: dict[str, dict]) -> tuple[list[str], list[dict]]:
    names = []
    for feature in FEATURES:
        names.extend((f"r0_{feature}", f"r1_{feature}", f"delta_{feature}"))
    names.extend((
        "quality_r0",
        "quality_r1",
        "quality_margin",
        "log_input_tweets",
        "relevant_ratio",
        "compressed_ratio",
        "window_index",
    ))

    rows = []
    data_cell_ids = set()
    for cell in iter_cells(data_root):
        data_cell_ids.add(cell.cell_id)
        if cell.cell_id not in metadata:
            continue
        meta = metadata[cell.cell_id]
        values = []
        for feature in FEATURES:
            value0 = float(meta["candidate_features"]["R0"][feature])
            value1 = float(meta["candidate_features"]["R1"][feature])
            values.extend((value0, value1, value1 - value0))
        quality0 = float(meta["candidate_scores"]["R0"])
        quality1 = float(meta["candidate_scores"]["R1"])
        _, _, prep = prepare_r1(cell)
        input_tweets = prep["input_tweets"]
        values.extend((
            quality0,
            quality1,
            quality1 - quality0,
            math.log1p(input_tweets),
            prep["relevant_tweets"] / max(1, input_tweets),
            prep["compressed_groups"] / max(1, input_tweets),
            float(cell.cell_id.split(".")[1].removeprefix("W")),
        ))
        rows.append({
            "cell_id": cell.cell_id,
            "crisis": cell.cell_id.split(".")[0],
            "window": cell.cell_id.split(".")[1],
            "x": values,
        })
    expected = set(metadata)
    if expected != data_cell_ids:
        raise ValueError(
            "selector metadata/data mismatch; "
            f"missing={sorted(data_cell_ids - expected)}, extra={sorted(expected - data_cell_ids)}"
        )
    found = {row["cell_id"] for row in rows}
    if found != expected:
        raise ValueError(f"feature coverage mismatch; missing={sorted(expected - found)}")
    return names, rows
