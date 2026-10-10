from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .data import SECTION_TITLES, SUBSECTION_VOCAB, iter_cells


REPORT_SUFFIX = ".report.json"
TWEETS_SUFFIX = ".tweets.jsonl"


class SubmissionError(ValueError):
    """Raised when a submission violates the released task format."""


@dataclass(frozen=True)
class CellSpec:
    cell_id: str
    crisis: str
    tweet_ids: frozenset[int]


def discover_cells(data_root: Path) -> dict[str, CellSpec]:
    cells: dict[str, CellSpec] = {}
    for cell in iter_cells(data_root):
        expected_id = cell.path.name.removesuffix(TWEETS_SUFFIX)
        if cell.cell_id != expected_id:
            raise SubmissionError(
                f"{cell.path}: window cell_id {cell.cell_id!r} does not match filename {expected_id!r}"
            )
        if cell.cell_id in cells:
            raise SubmissionError(f"duplicate input cell_id: {cell.cell_id}")
        ids = [tweet.get("id") for tweet in cell.tweets]
        if any(isinstance(value, bool) or not isinstance(value, int) for value in ids):
            raise SubmissionError(f"{cell.path}: tweet ids must be integers")
        if len(ids) != len(set(ids)):
            raise SubmissionError(f"{cell.path}: duplicate tweet ids")
        cells[cell.cell_id] = CellSpec(
            cell_id=cell.cell_id,
            crisis=cell.path.parent.name,
            tweet_ids=frozenset(ids),
        )
    if not cells:
        raise SubmissionError(f"no *{TWEETS_SUFFIX} files found under {data_root}")
    return cells


def _validate_report(payload: object, cell: CellSpec, source: Path) -> None:
    def fail(message: str) -> None:
        raise SubmissionError(f"{source}: {message}")

    if not isinstance(payload, dict):
        fail("report root must be a JSON object")
    if payload.get("meta") != {"schema_version": "1.2"}:
        fail('meta must be exactly {"schema_version": "1.2"}')
    sections = payload.get("sections")
    if not isinstance(sections, list):
        fail("sections must be a list")

    seen_sections: set[str] = set()
    previous_section = 0
    seen_bullet_ids: set[str] = set()
    for section in sections:
        if not isinstance(section, dict):
            fail("every section must be an object")
        sid = str(section.get("id", ""))
        if sid not in SECTION_TITLES:
            fail(f"illegal section id {sid!r}")
        if sid in seen_sections:
            fail(f"duplicate section id {sid}")
        if int(sid) <= previous_section:
            fail("sections must be in ascending numeric id order")
        previous_section = int(sid)
        seen_sections.add(sid)
        if section.get("title") != SECTION_TITLES[sid]:
            fail(f"section {sid} has a non-canonical title")
        subsections = section.get("subsections")
        if not isinstance(subsections, list) or not subsections:
            fail(f"section {sid} must contain at least one subsection")

        seen_titles: set[str] = set()
        seen_subsection_ids: set[str] = set()
        previous_subsection_letter = ""
        for subsection in subsections:
            if not isinstance(subsection, dict):
                fail(f"section {sid} contains a non-object subsection")
            subsection_id = str(subsection.get("id", ""))
            match = re.fullmatch(re.escape(sid) + r"([a-z])", subsection_id)
            if not match:
                fail(f"illegal subsection id {subsection_id!r} for section {sid}")
            if subsection_id in seen_subsection_ids:
                fail(f"duplicate subsection id {subsection_id}")
            if match.group(1) <= previous_subsection_letter:
                fail(f"subsections in section {sid} must be in ascending id order")
            previous_subsection_letter = match.group(1)
            seen_subsection_ids.add(subsection_id)
            title = subsection.get("title")
            if title not in SUBSECTION_VOCAB[sid]:
                fail(f"subsection {subsection_id} has illegal title {title!r}")
            if title in seen_titles:
                fail(f"section {sid} repeats subsection title {title!r}")
            seen_titles.add(title)
            bullets = subsection.get("bullets")
            if not isinstance(bullets, list) or not bullets:
                fail(f"subsection {subsection_id} must contain at least one bullet")

            previous_bullet_number = 0
            for bullet in bullets:
                if not isinstance(bullet, dict):
                    fail(f"subsection {subsection_id} contains a non-object bullet")
                bullet_id = str(bullet.get("id", ""))
                match = re.fullmatch(re.escape(subsection_id) + r"\.([1-9][0-9]*)", bullet_id)
                if not match:
                    fail(f"illegal bullet id {bullet_id!r} for subsection {subsection_id}")
                bullet_number = int(match.group(1))
                if bullet_number <= previous_bullet_number:
                    fail(f"bullets in subsection {subsection_id} must be in ascending id order")
                previous_bullet_number = bullet_number
                if bullet_id in seen_bullet_ids:
                    fail(f"duplicate bullet id {bullet_id}")
                seen_bullet_ids.add(bullet_id)
                if not isinstance(bullet.get("text"), str) or not bullet["text"].strip():
                    fail(f"bullet {bullet_id} has empty text")
                if bullet.get("confidence") not in {"confirmed", "unconfirmed"}:
                    fail(f"bullet {bullet_id} has illegal confidence label")
                evidence = bullet.get("tweet_ids")
                if not isinstance(evidence, list):
                    fail(f"bullet {bullet_id} tweet_ids must be a list")
                if any(isinstance(value, bool) or not isinstance(value, int) for value in evidence):
                    fail(f"bullet {bullet_id} tweet_ids must contain integers only")
                if len(evidence) != len(set(evidence)):
                    fail(f"bullet {bullet_id} contains duplicate tweet_ids")
                if sid == "1" and evidence:
                    fail(f"section 1 bullet {bullet_id} must not contain evidence links")
                if sid != "1" and not evidence:
                    fail(f"section {sid} bullet {bullet_id} needs at least one evidence link")
                invalid_ids = sorted(set(evidence) - cell.tweet_ids)
                if invalid_ids:
                    fail(f"bullet {bullet_id} cites ids absent from its cell: {invalid_ids}")


def validate_reports(data_root: Path, reports_root: Path) -> dict:
    cells = discover_cells(data_root)
    report_paths = sorted(reports_root.rglob(f"*{REPORT_SUFFIX}"))
    by_cell: dict[str, list[Path]] = {}
    for path in report_paths:
        by_cell.setdefault(path.name.removesuffix(REPORT_SUFFIX), []).append(path)

    missing = sorted(set(cells) - set(by_cell))
    extra = sorted(set(by_cell) - set(cells))
    duplicates = sorted(cell_id for cell_id, paths in by_cell.items() if len(paths) != 1)
    if missing or extra or duplicates:
        raise SubmissionError(
            f"cell coverage mismatch: missing={missing}, extra={extra}, duplicates={duplicates}"
        )

    validated: list[dict[str, str]] = []
    for cell_id in sorted(cells):
        path = by_cell[cell_id][0]
        cell = cells[cell_id]
        if path.parent.name != cell.crisis:
            raise SubmissionError(
                f"{path}: expected crisis directory {cell.crisis!r}, found {path.parent.name!r}"
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SubmissionError(f"{path}: invalid UTF-8 JSON: {exc}") from exc
        _validate_report(payload, cell, path)
        validated.append({"cell_id": cell_id, "relative_path": f"{cell.crisis}/{path.name}"})
    return {"cells": len(cells), "reports": validated}


def sanitize_reports(data_root: Path, reports_root: Path, output_root: Path) -> dict:
    """Clean evidence links and drop unsupported bullets from sections 2--11.

    The released format requires evidence for every non-overview bullet. This
    narrow sanitation deliberately does not repair other schema errors: the
    strict validator below must still accept the result.
    """
    cells = discover_cells(data_root)
    report_paths = sorted(reports_root.rglob(f"*{REPORT_SUFFIX}"))
    by_cell: dict[str, list[Path]] = {}
    for path in report_paths:
        by_cell.setdefault(path.name.removesuffix(REPORT_SUFFIX), []).append(path)
    missing = sorted(set(cells) - set(by_cell))
    extra = sorted(set(by_cell) - set(cells))
    duplicates = sorted(cell_id for cell_id, paths in by_cell.items() if len(paths) != 1)
    if missing or extra or duplicates:
        raise SubmissionError(
            f"cell coverage mismatch: missing={missing}, extra={extra}, duplicates={duplicates}"
        )

    dropped = 0
    removed_invalid_evidence_links = 0
    affected_cells: list[str] = []
    for cell_id in sorted(cells):
        source = by_cell[cell_id][0]
        cell = cells[cell_id]
        if source.parent.name != cell.crisis:
            raise SubmissionError(f"{source}: expected crisis directory {cell.crisis!r}")
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SubmissionError(f"{source}: invalid UTF-8 JSON: {exc}") from exc
        cell_dropped = 0
        if isinstance(payload, dict) and isinstance(payload.get("sections"), list):
            kept_sections = []
            for section in payload["sections"]:
                if not isinstance(section, dict) or not isinstance(section.get("subsections"), list):
                    kept_sections.append(section)
                    continue
                sid = str(section.get("id", ""))
                kept_subsections = []
                for subsection in section["subsections"]:
                    if not isinstance(subsection, dict) or not isinstance(subsection.get("bullets"), list):
                        kept_subsections.append(subsection)
                        continue
                    original = subsection["bullets"]
                    if sid == "1":
                        kept = [{**bullet, "tweet_ids": []} if isinstance(bullet, dict) else bullet for bullet in original]
                    else:
                        kept = []
                        for bullet in original:
                            if not isinstance(bullet, dict):
                                continue
                            evidence = bullet.get("tweet_ids")
                            if not isinstance(evidence, list):
                                kept.append(bullet)
                                continue
                            valid_evidence = [
                                value
                                for value in evidence
                                if isinstance(value, int)
                                and not isinstance(value, bool)
                                and value in cell.tweet_ids
                            ]
                            removed_invalid_evidence_links += len(evidence) - len(valid_evidence)
                            if valid_evidence:
                                kept.append({**bullet, "tweet_ids": valid_evidence})
                    cell_dropped += len(original) - len(kept)
                    if kept:
                        kept_subsections.append({**subsection, "bullets": kept})
                if kept_subsections:
                    kept_sections.append({**section, "subsections": kept_subsections})
            payload = {**payload, "sections": kept_sections}
        destination = output_root / cell.crisis / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
        if cell_dropped:
            affected_cells.append(cell_id)
            dropped += cell_dropped

    validation = validate_reports(data_root, output_root)
    return {
        "cells": validation["cells"],
        "dropped_zero_evidence_bullets": dropped,
        "removed_invalid_evidence_links": removed_invalid_evidence_links,
        "affected_cells": affected_cells,
        "output_root": str(output_root.resolve()),
    }


def _check_methods(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise SubmissionError(f"cannot read methods file {path}: {exc}") from exc
    if not text:
        raise SubmissionError("methods.md must not be empty")
    if not re.search(r"\bindependent(?:ly)?\b", text, flags=re.IGNORECASE):
        raise SubmissionError("methods.md must confirm that cells were processed independently")
    return text + "\n"


def _write_deterministic_zip(source_root: Path, output_zip: Path) -> list[str]:
    members: list[str] = []
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(p for p in source_root.rglob("*") if p.is_file()):
            name = path.relative_to(source_root).as_posix()
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)
            members.append(name)
    return members


def package_submission(
    data_root: Path,
    reports_root: Path,
    methods_path: Path,
    output_zip: Path,
    route: str,
) -> dict:
    validation = validate_reports(data_root, reports_root)
    methods = _check_methods(methods_path)
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="lt4cpr-stage-") as temporary:
        stage = Path(temporary)
        (stage / "methods.md").write_text(methods, encoding="utf-8", newline="\n")
        submission_root = stage / "submission"
        for item in validation["reports"]:
            source = reports_root / item["relative_path"]
            destination = submission_root / item["relative_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        members = _write_deterministic_zip(stage, output_zip)

    with tempfile.TemporaryDirectory(prefix="lt4cpr-verify-") as temporary:
        extracted = Path(temporary)
        with zipfile.ZipFile(output_zip) as archive:
            archive.extractall(extracted)
        if sorted(path.relative_to(extracted).as_posix() for path in extracted.rglob("*") if path.is_file()) != members:
            raise SubmissionError("archive member list changed after extraction")
        _check_methods(extracted / "methods.md")
        extracted_validation = validate_reports(data_root, extracted / "submission")
        if extracted_validation["cells"] != validation["cells"]:
            raise SubmissionError("archive validation count differs from source validation")

    digest = hashlib.sha256(output_zip.read_bytes()).hexdigest()
    manifest = {
        "format": "LT4CPR submission rehearsal",
        "route": route,
        "data_root": str(data_root.resolve()),
        "reports_root": str(reports_root.resolve()),
        "archive": str(output_zip.resolve()),
        "sha256": digest,
        "cells": validation["cells"],
        "members": members,
        "checks": [
            "source reports validated without gold",
            "cell coverage is complete and exact",
            "methods.md confirms independent cell processing",
            "archive extracted and revalidated",
        ],
    }
    manifest_path = output_zip.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and rehearse an LT4CPR submission without gold reports.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="validate a report directory")
    validate_parser.add_argument("--data", type=Path, required=True)
    validate_parser.add_argument("--reports", type=Path, required=True)

    sanitize_parser = subparsers.add_parser("sanitize", help="drop zero-evidence non-overview bullets, then validate")
    sanitize_parser.add_argument("--data", type=Path, required=True)
    sanitize_parser.add_argument("--reports", type=Path, required=True)
    sanitize_parser.add_argument("--out", type=Path, required=True)

    package_parser = subparsers.add_parser("package", help="validate, package, extract, and revalidate")
    package_parser.add_argument("--data", type=Path, required=True)
    package_parser.add_argument("--reports", type=Path, required=True)
    package_parser.add_argument("--methods", type=Path, required=True)
    package_parser.add_argument("--out", type=Path, required=True)
    package_parser.add_argument("--route", required=True)

    args = parser.parse_args()
    try:
        if args.command == "validate":
            result = validate_reports(args.data, args.reports)
        elif args.command == "sanitize":
            result = sanitize_reports(args.data, args.reports, args.out)
        else:
            result = package_submission(args.data, args.reports, args.methods, args.out, args.route)
    except SubmissionError as exc:
        parser.exit(2, f"submission validation failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
