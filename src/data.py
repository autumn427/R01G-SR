from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SECTION_TITLES = {
    "1": "Situation overview",
    "2": "Timeline",
    "3": "Casualties and human impact",
    "4": "Infrastructure and service impact",
    "5": "Displacement and movement",
    "6": "Hazard assessment",
    "7": "Response actions",
    "8": "Communication and information",
    "9": "Aid and relief",
    "10": "Organizational and administrative activity",
    "11": "Social and community response",
}

SUBSECTION_VOCAB = {
    "1": ["Overview"],
    "2": ["Chronological timeline"],
    "3": ["Fatalities", "Injuries", "Cases", "Missing persons", "Trapped persons", "Rescued persons", "Found persons", "Treatment and admissions"],
    "4": ["Damage", "Destruction", "Service disruptions", "Restoration progress"],
    "5": ["Evacuations", "Displacement", "Returns", "Access constraints"],
    "6": ["Hazard assessment"],
    "7": ["Operational response"],
    "8": ["Warnings and alerts", "Information updates", "Requests for help"],
    "9": ["Aid provided and distributed", "Aid received and needs coverage", "Donations and financial transfers"],
    "10": ["Meetings and decisions", "Administrative and economic measures"],
    "11": ["Social support", "Public sentiment"],
}

SOURCE_RANK = {"Government": 5, "NGOs": 4, "Media": 3, "Eyewitness": 2, "Outsiders": 1, "Not labeled": 0}


@dataclass(frozen=True)
class Cell:
    cell_id: str
    crisis: dict
    window: dict
    tweets: list[dict]
    path: Path


def load_cell(path: Path) -> Cell:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    crisis = next(x for x in records if x["record_type"] == "crisis")
    window = next(x for x in records if x["record_type"] == "window")
    tweets = [x for x in records if x["record_type"] == "tweet"]
    return Cell(window["cell_id"], crisis, window, tweets, path)


def iter_cells(root: Path, selected: set[str] | None = None) -> Iterable[Cell]:
    for path in sorted(root.rglob("*.tweets.jsonl")):
        cell = load_cell(path)
        if selected is None or cell.cell_id in selected:
            yield cell


def normalize_text(text: str) -> str:
    text = re.sub(r"^RT\s+@\w+:\s*", "", text, flags=re.I)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[@#](\w+)", r"\1", text)
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return " ".join(text.split())


def content_tokens(text: str) -> set[str]:
    stop = {"the", "a", "an", "and", "or", "of", "to", "in", "at", "for", "on", "is", "are", "was", "were", "with", "this", "that", "from", "by", "it", "be", "has", "have", "as"}
    return {w for w in normalize_text(text).split() if len(w) > 2 and w not in stop}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(1, len(a | b))


def relevant_tweets(cell: Cell) -> list[dict]:
    context = " ".join(str(cell.crisis.get(k, "")) for k in ("title", "hazard", "location"))
    context_tokens = content_tokens(context)
    crisis_terms = {"emergency", "evacuation", "evacuated", "injured", "dead", "fatal", "damage", "destroyed", "warning", "alert", "rescue", "hospital", "shelter", "aid", "relief", "outage", "closed", "fire", "flood", "quake", "spill", "leak", "storm", "hazard"}
    kept = []
    for tweet in cell.tweets:
        tokens = content_tokens(tweet["text"])
        labeled = tweet.get("information_source") != "Not labeled"
        topical = bool(tokens & (context_tokens | crisis_terms))
        if labeled or topical:
            kept.append(tweet)
    return kept


def compress_tweets(tweets: list[dict], threshold: float = 0.70) -> list[dict]:
    """Greedy near-duplicate grouping, preserving every member as evidence."""
    groups: list[dict] = []
    for tweet in tweets:
        tokens = content_tokens(tweet["text"])
        match = next((g for g in groups if jaccard(tokens, g["tokens"]) >= threshold), None)
        if match is None:
            groups.append({"representative": tweet, "tokens": tokens, "members": [tweet]})
        else:
            match["members"].append(tweet)
            old = match["representative"]
            if SOURCE_RANK.get(tweet.get("information_source", ""), 0) > SOURCE_RANK.get(old.get("information_source", ""), 0):
                match["representative"] = tweet
                match["tokens"] = tokens
    return groups


def evidence_lines(groups: list[dict]) -> list[str]:
    lines = []
    for group in groups:
        rep = group["representative"]
        ids = [m["id"] for m in group["members"]]
        lines.append(f"[{','.join(map(str, ids))}] {rep['timestamp']} | {rep.get('information_source','')} | {rep['text']}")
    return lines


def normalize_report(report: dict) -> dict:
    """Keep legal titles, omit empty groups, and deterministically renumber IDs."""
    if isinstance(report.get("report"), dict) and "sections" not in report:
        report = {**report["report"], "applicable_subsections": report.get("applicable_subsections", {})}
    applicable_raw = report.get("subsection_profile", report.get("applicable_subsections", {}))
    applicable: dict[str, list[str]] = {}
    if isinstance(applicable_raw, dict):
        for sid, titles in applicable_raw.items():
            sid = str(sid)
            if sid in SUBSECTION_VOCAB and isinstance(titles, list):
                applicable[sid] = list(dict.fromkeys(
                    str(x) for x in titles if str(x) in SUBSECTION_VOCAB[sid]
                ))
    by_section: dict[str, list[dict]] = {}
    sections_raw = report.get("sections", [])
    if isinstance(sections_raw, dict):
        sections = []
        for key, value in sections_raw.items():
            if isinstance(value, dict):
                sections.append({"id": value.get("id", key), **value})
    else:
        sections = sections_raw if isinstance(sections_raw, list) else []
    for section in sections:
        if not isinstance(section, dict):
            continue
        sid = str(section.get("id", ""))
        if sid not in SECTION_TITLES:
            continue
        legal = SUBSECTION_VOCAB[sid]
        found = []
        subsections_raw = section.get("subsections", [])
        if isinstance(subsections_raw, dict):
            subsections = []
            for key, value in subsections_raw.items():
                if isinstance(value, dict):
                    subsections.append({"title": value.get("title", key), **value})
        else:
            subsections = subsections_raw if isinstance(subsections_raw, list) else []
        for subsection in subsections:
            if not isinstance(subsection, dict):
                continue
            title = str(subsection.get("title", ""))
            if title not in legal:
                continue
            bullets_raw = subsection.get("bullets", [])
            if isinstance(bullets_raw, dict):
                bullets_raw = list(bullets_raw.values())
            bullets = [b for b in bullets_raw if isinstance(b, dict) and str(b.get("text", "")).strip()]
            if bullets:
                found.append({"title": title, "bullets": bullets, "original_id": str(subsection.get("id", ""))})
        merged: dict[str, dict] = {}
        for subsection in found:
            title = subsection["title"]
            if title in merged:
                merged[title]["bullets"].extend(subsection["bullets"])
            else:
                merged[title] = subsection
        found = list(merged.values())
        order = applicable.get(sid) or legal
        original_ids = [x["original_id"] for x in found]
        preserve_ids = (
            not applicable.get(sid)
            and len(original_ids) == len(set(original_ids))
            and all(re.fullmatch(re.escape(sid) + r"[a-z]", value) for value in original_ids)
        )
        found.sort(key=lambda x: ord(x["original_id"][-1]) if preserve_ids else (order.index(x["title"]) if x["title"] in order else len(order) + legal.index(x["title"])))
        if found:
            by_section[sid] = found

    sections = []
    for sid in map(str, range(1, 12)):
        if sid not in by_section:
            continue
        subsections = []
        order = applicable.get(sid) or [x["title"] for x in by_section[sid]]
        original_ids = [x["original_id"] for x in by_section[sid]]
        preserve_ids = (
            not applicable.get(sid)
            and len(original_ids) == len(set(original_ids))
            and all(re.fullmatch(re.escape(sid) + r"[a-z]", value) for value in original_ids)
        )
        for index, subsection in enumerate(by_section[sid]):
            letter_index = ord(subsection["original_id"][-1]) - ord("a") if preserve_ids else (order.index(subsection["title"]) if subsection["title"] in order else index)
            ssid = f"{sid}{chr(97 + letter_index)}"
            bullets = []
            source_bullets = subsection["bullets"][:4] if sid == "1" else (subsection["bullets"][:8] if sid == "2" else subsection["bullets"])
            for n, bullet in enumerate(source_bullets, 1):
                ids = bullet.get("tweet_ids", [])
                ids = sorted({int(x) for x in ids if str(x).isdigit()})
                bullets.append({
                    "id": f"{ssid}.{n}",
                    "text": str(bullet["text"]).strip(),
                    "confidence": bullet.get("confidence") if bullet.get("confidence") in {"confirmed", "unconfirmed"} else "unconfirmed",
                    "tweet_ids": [] if sid == "1" else ids,
                })
            subsections.append({"id": ssid, "title": subsection["title"], "bullets": bullets})
        sections.append({"id": sid, "title": SECTION_TITLES[sid], "subsections": subsections})
    return {"meta": {"schema_version": "1.2"}, "sections": sections}
