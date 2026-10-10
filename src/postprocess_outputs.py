"""Apply deterministic schema normalization to completed run outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import normalize_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", default="R0,R1,R2,R3,R4")
    args = parser.parse_args()
    runs = {item.strip() for item in args.runs.split(",") if item.strip()}
    written = 0
    for path in sorted(args.source.rglob("*.report.json")):
        relative = path.relative_to(args.source)
        if not relative.parts or relative.parts[0] not in runs:
            continue
        report = normalize_report(json.loads(path.read_text(encoding="utf-8")))
        ids = [
            (str(section["id"]), str(subsection["id"]))
            for section in report["sections"]
            for subsection in section.get("subsections", [])
        ]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate subsection IDs remain in {path}")
        target = args.out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        written += 1
    print(f"wrote {written} normalized reports to {args.out}")


if __name__ == "__main__":
    main()
