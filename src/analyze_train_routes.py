"""Compare train-crisis routes against a frozen baseline and decision gate."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


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


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take a percentile of an empty sequence")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_bootstrap(deltas: list[float], samples: int = 100_000, seed: int = 20260826) -> dict:
    rng = random.Random(seed)
    means = [sum(rng.choice(deltas) for _ in deltas) / len(deltas) for _ in range(samples)]
    return {
        "samples": samples,
        "seed": seed,
        "ci95": [percentile(means, 0.025), percentile(means, 0.975)],
        "probability_positive": sum(value > 0 for value in means) / samples,
    }


def one_sided_sign_p(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / (2 ** n)


def compare_payloads(
    baseline: dict,
    candidate: dict,
    min_gain: float = 0.005,
    min_crisis_wins: int = 5,
) -> dict:
    crises = sorted(set(baseline["crises"]) & set(candidate["crises"]))
    if not crises:
        raise ValueError("baseline and candidate share no crises")
    crisis_deltas = {
        crisis: candidate["crises"][crisis]["primary"] - baseline["crises"][crisis]["primary"]
        for crisis in crises
    }
    wins = sum(delta > 0 for delta in crisis_deltas.values())
    losses = sum(delta < 0 for delta in crisis_deltas.values())
    ties = len(crises) - wins - losses
    macro_deltas = {
        metric: candidate["macro_by_crisis"][metric] - baseline["macro_by_crisis"][metric]
        for metric in METRICS
    }
    primary_gain = macro_deltas["primary"]
    return {
        "macro": {metric: candidate["macro_by_crisis"][metric] for metric in METRICS},
        "delta_vs_baseline": macro_deltas,
        "crisis_primary_delta": crisis_deltas,
        "primary_wins": wins,
        "primary_losses": losses,
        "primary_ties": ties,
        "one_sided_sign_p": one_sided_sign_p(wins, losses),
        "paired_crisis_bootstrap": paired_bootstrap(list(crisis_deltas.values())),
        "gate": {
            "minimum_macro_primary_gain": min_gain,
            "minimum_crisis_wins": min_crisis_wins,
            "passes_gain": primary_gain >= min_gain,
            "passes_wins": wins >= min_crisis_wins,
            "passes": primary_gain >= min_gain and wins >= min_crisis_wins,
        },
    }


def markdown_report(baseline_name: str, baseline: dict, comparisons: dict) -> str:
    lines = [
        "# Train-crisis route analysis",
        "",
        "The decision gate was frozen before aggregate train scores were available.",
        "",
        "| Route | Primary | Delta vs baseline | Crisis wins | Bootstrap 95% CI | Gate |",
        "|---|---:|---:|---:|---:|---|",
        f"| {baseline_name} | {baseline['macro_by_crisis']['primary']:.6f} | 0.000000 | -- | -- | baseline |",
    ]
    for name, result in comparisons.items():
        interval = result["paired_crisis_bootstrap"]["ci95"]
        lines.append(
            f"| {name} | {result['macro']['primary']:.6f} | "
            f"{result['delta_vs_baseline']['primary']:+.6f} | "
            f"{result['primary_wins']}/{result['primary_wins'] + result['primary_losses'] + result['primary_ties']} | "
            f"[{interval[0]:+.6f}, {interval[1]:+.6f}] | "
            f"{'PASS' if result['gate']['passes'] else 'FAIL'} |"
        )
    lines.extend(["", "## Crisis-level Primary deltas", ""])
    crises = sorted(next(iter(comparisons.values()))["crisis_primary_delta"]) if comparisons else []
    lines.append("| Crisis | " + " | ".join(comparisons) + " |")
    lines.append("|---|" + "---:|" * len(comparisons))
    for crisis in crises:
        lines.append("| " + crisis + " | " + " | ".join(
            f"{comparisons[name]['crisis_primary_delta'][crisis]:+.6f}" for name in comparisons
        ) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="NAME=score.json")
    parser.add_argument("--candidate", action="append", default=[], help="NAME=score.json")
    parser.add_argument("--min-gain", type=float, default=0.005)
    parser.add_argument("--min-crisis-wins", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args()

    def load_spec(spec: str) -> tuple[str, dict]:
        name, raw_path = spec.split("=", 1)
        return name, json.loads(Path(raw_path).read_text(encoding="utf-8"))

    baseline_name, baseline = load_spec(args.baseline)
    comparisons = {}
    for spec in args.candidate:
        name, candidate = load_spec(spec)
        comparisons[name] = compare_payloads(
            baseline,
            candidate,
            min_gain=args.min_gain,
            min_crisis_wins=args.min_crisis_wins,
        )
    payload = {
        "baseline": baseline_name,
        "baseline_macro": baseline["macro_by_crisis"],
        "comparisons": comparisons,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = markdown_report(baseline_name, baseline, comparisons)
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
