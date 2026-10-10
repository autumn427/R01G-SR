from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from .data import Cell, SUBSECTION_VOCAB, compress_tweets, content_tokens, evidence_lines, jaccard, normalize_report, relevant_tweets
from .llm import LLMClient, parse_json


SYSTEM = """You are an expert humanitarian situation-report analyst. Work only from the supplied cell evidence. Never use outside knowledge. Return one valid JSON object and no prose."""

ALL_SUBSECTION_TITLES = [title for titles in SUBSECTION_VOCAB.values() for title in titles]

REPORT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "subsection_profile": {
            "type": "object",
            "properties": {
                str(section_id): {
                    "type": "array",
                    "items": {"type": "string", "enum": SUBSECTION_VOCAB[str(section_id)]},
                    "maxItems": 8,
                }
                for section_id in range(1, 12)
            },
            "required": [str(section_id) for section_id in range(1, 12)],
            "additionalProperties": False,
        },
        "sections": {"type": "array", "minItems": 1, "maxItems": 11, "items": {"type": "object", "properties": {
            "id": {"type": "string", "pattern": "^(1|2|3|4|5|6|7|8|9|10|11)$"}, "title": {"type": "string"},
            "subsections": {"type": "array", "minItems": 1, "maxItems": 8, "items": {"type": "object", "properties": {
                "id": {"type": "string", "pattern": "^(1|2|3|4|5|6|7|8|9|10|11)[a-z]$"}, "title": {"type": "string", "enum": ALL_SUBSECTION_TITLES},
                "bullets": {"type": "array", "minItems": 1, "maxItems": 8, "items": {"type": "object", "properties": {
                    "text": {"type": "string"}, "confidence": {"type": "string", "enum": ["confirmed", "unconfirmed"]},
                    "tweet_ids": {"type": "array", "items": {"type": "integer"}, "maxItems": 8},
                }, "required": ["text", "confidence", "tweet_ids"]}},
            }, "required": ["id", "title", "bullets"]}},
        }, "required": ["id", "title", "subsections"]}},
    }, "required": ["subsection_profile", "sections"],
}

LEDGER_JSON_SCHEMA = {
    "type": "object", "properties": {"facts": {"type": "array", "minItems": 1, "maxItems": 300, "items": {"type": "object", "properties": {
        "claim": {"type": "string"}, "subsection": {"type": "string", "enum": ALL_SUBSECTION_TITLES}, "tweet_ids": {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 8},
        "time": {"type": "string"}, "status": {"type": "string", "enum": ["asserted", "planned", "negated", "corrected"]},
    }, "required": ["claim", "subsection", "tweet_ids", "time", "status"]}}}, "required": ["facts"],
}


def schema_text() -> str:
    return "\n".join(f"Section {sid}: " + "; ".join(titles) for sid, titles in SUBSECTION_VOCAB.items())


REPORT_RULES = f"""
Return {{"subsection_profile":{{"3":[...],"4":[...],...}},"sections":[...]}}
following schema version 1.2. In subsection_profile, list—in expected
report order—every title structurally applicable to this crisis across its
whole lifecycle, including titles with no fact in the current window. This
list establishes stable subsection letters across cumulative windows.
Allowed subsection titles are exactly:
{schema_text()}
Each bullet must be one atomic, complete sentence with fields text, confidence
(confirmed or unconfirmed), and tweet_ids. Cite only integer IDs shown in the
evidence. Section 1 has exactly four overview bullets when evidence permits and
its tweet_ids must be empty. Section 2 is chronological. Omit empty subsections.
Confirmed means corroborated by at least two independent reports, not merely an
official source. Prefer the latest supported revision of changing figures;
record consequential corrections under Information updates. Do not infer facts.
Sections 3–11 are the primary output: populate every one that has evidence and
do not stop after sections 1–2. Section 1 has at most 4 bullets and section 2
has at most 8 bullets so they cannot consume the report budget.
"""


def cell_header(cell: Cell) -> str:
    return json.dumps({"crisis": cell.crisis, "window": cell.window}, ensure_ascii=False)


def raw_lines(cell: Cell, max_chars: int = 85000) -> list[str]:
    lines = [f"[{t['id']}] {t['timestamp']} | {t.get('information_source','')} | {t['text']}" for t in cell.tweets]
    # Preserve beginning and end when a long cell exceeds the direct baseline's context budget.
    while sum(map(len, lines)) > max_chars and len(lines) > 2:
        lines.pop(len(lines) // 2)
    return lines


def generate_report(
    client: LLMClient,
    cell: Cell,
    lines: list[str],
    mode: str = "factual",
    temperature: float = 0.0,
    seed: int | None = None,
    preparation_text: str = "",
) -> tuple[dict, dict]:
    emphasis = {
        "factual": "Maximize faithful atomic coverage and avoid redundant paraphrases.",
        "coverage": "Systematically inspect every allowed subsection and recover less salient but supported facts; still omit unsupported claims.",
    }[mode]
    preparation = ""
    if preparation_text.strip():
        preparation = (
            "\nNAVIGATION INDEX (deterministically derived from the evidence; use it to "
            "find information, but never cite it or treat it as independent evidence):\n"
            + preparation_text.strip()
            + "\n"
        )
    prompt = (
        f"{REPORT_RULES}\n{emphasis}\nCELL:\n{cell_header(cell)}"
        f"{preparation}\nEVIDENCE:\n" + "\n".join(lines)
    )
    text, meta = client.chat(SYSTEM, prompt, temperature=temperature, max_tokens=6144, json_schema=REPORT_JSON_SCHEMA, seed=seed)
    parsed = parse_json(text)
    if parsed.get("_recovered_truncation"):
        meta["recovered_truncation"] = True
    report = normalize_report(parsed)
    if not report["sections"]:
        raise ValueError(f"model returned no usable sections: {text[:1000]}")
    return report, meta


def extract_ledger(
    client: LLMClient,
    cell: Cell,
    lines: list[str],
    chunk_size: int = 150,
    checkpoint_dir: Path | None = None,
    reuse_checkpoints: bool = True,
) -> tuple[list[dict], list[dict]]:
    facts, calls = [], []
    instruction = f"""
Extract all distinct reportable atomic facts from this evidence chunk.
Return {{"facts":[{{"claim":"...","subsection":"exact allowed title",
"tweet_ids":[1,2],"time":"timestamp or empty","status":"asserted|planned|negated|corrected"}}]}}.
Use only these subsection titles:\n{schema_text()}
Merge paraphrases only when they express the same event and value. Preserve
contradictions and revisions as separate facts. Cite every supporting ID.
    """
    for start in range(0, len(lines), chunk_size):
        chunk = lines[start:start + chunk_size]
        checkpoint = checkpoint_dir / f"chunk_{start:05d}.json" if checkpoint_dir else None
        if checkpoint and reuse_checkpoints and checkpoint.exists():
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            parsed = {"facts": saved["facts"]}
            meta = {"cache_hit": True, "source_meta": saved.get("meta", {})}
        else:
            prompt = instruction + "\nCELL:\n" + cell_header(cell) + "\nEVIDENCE:\n" + "\n".join(chunk)
            text, meta = client.chat(SYSTEM, prompt, max_tokens=6144, json_schema=LEDGER_JSON_SCHEMA)
            parsed = parse_json(text)
            if parsed.get("_recovered_truncation"):
                meta["recovered_truncation"] = True
            if checkpoint:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_text(json.dumps({"facts": parsed.get("facts", []), "meta": meta}, ensure_ascii=False, indent=2), encoding="utf-8")
        for fact in parsed.get("facts", []):
            if fact.get("subsection") in {x for values in SUBSECTION_VOCAB.values() for x in values} and fact.get("claim"):
                facts.append(fact)
        calls.append(meta)
    if not facts:
        raise ValueError("fact extraction returned an empty usable ledger")
    return facts, calls


def consolidate_ledger(client: LLMClient, cell: Cell, ledger: list[dict], mode: str = "factual", temperature: float = 0.0) -> tuple[dict, dict]:
    emphasis = {
        "factual": "Prefer precise, strongly supported claims. Merge duplicate facts and resolve revisions using time and correction status.",
        "coverage": "Cover every subsection represented in the ledger, including supported minority facts, while merging true duplicates.",
    }[mode]
    prompt = f"{REPORT_RULES}\n{emphasis}\nCELL:\n{cell_header(cell)}\nATOMIC FACT LEDGER:\n" + json.dumps(ledger, ensure_ascii=False)
    text, meta = client.chat(SYSTEM, prompt, temperature=temperature, max_tokens=6144, json_schema=REPORT_JSON_SCHEMA)
    parsed = parse_json(text)
    if parsed.get("_recovered_truncation"):
        meta["recovered_truncation"] = True
    report = normalize_report(parsed)
    if not report["sections"]:
        raise ValueError(f"model returned no usable sections: {text[:1000]}")
    return report, meta


def _numbers(text: str) -> set[str]:
    return {x.replace(",", "") for x in re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", text)}


def _factual_numbers(text: str) -> set[str]:
    """Numbers that should be grounded in tweet text, excluding date/time notation."""
    months = r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
    cleaned = re.sub(rf"\b(?:{months})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?\b", " ", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", " ", cleaned)
    cleaned = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\s*[ap]m)?\b", " ", cleaned, flags=re.IGNORECASE)
    return {number for number in _numbers(cleaned) if not (1900 <= float(number) <= 2100 and number.isdigit())}


def _all_bullets(report: dict):
    for section in report.get("sections", []):
        for subsection in section.get("subsections", []):
            for bullet in subsection.get("bullets", []):
                yield section["id"], subsection, bullet


def validate_report(report: dict, cell: Cell) -> tuple[dict, dict]:
    valid_ids = {t["id"] for t in cell.tweets}
    by_id = {t["id"]: t for t in cell.tweets}
    tweet_tokens = {t["id"]: content_tokens(t["text"]) for t in cell.tweets}
    stats = defaultdict(int)
    for sid, subsection, bullet in list(_all_bullets(report)):
        if sid == "1":
            bullet["tweet_ids"] = []
            continue
        claim_tokens = content_tokens(bullet["text"])
        ranked = sorted(valid_ids, key=lambda tid: jaccard(claim_tokens, tweet_tokens[tid]), reverse=True)
        cited = [tid for tid in bullet.get("tweet_ids", []) if tid in valid_ids]
        if not cited or max((jaccard(claim_tokens, tweet_tokens[x]) for x in cited), default=0) < 0.08:
            cited = [tid for tid in ranked[:3] if jaccard(claim_tokens, tweet_tokens[tid]) >= 0.08]
            stats["citations_repaired"] += 1
        evidence_text = " ".join(by_id[x]["text"] for x in cited)
        required_numbers = _factual_numbers(bullet["text"])
        unsupported = required_numbers - _numbers(evidence_text)
        if unsupported:
            numeric_support = [
                tid for tid in ranked
                if unsupported <= _numbers(by_id[tid]["text"])
            ]
            repaired = list(dict.fromkeys(numeric_support[:3] + cited))[:8]
            if repaired != cited:
                cited = repaired
                evidence_text = " ".join(by_id[x]["text"] for x in cited)
                unsupported = required_numbers - _numbers(evidence_text)
                stats["numeric_citations_repaired"] += 1
        bullet["_drop"] = not cited or bool(unsupported)
        if unsupported:
            stats["unsupported_number_bullets"] += 1
        if not cited:
            stats["unsupported_text_bullets"] += 1
        bullet["tweet_ids"] = cited[:8]
        # Corroboration is counted over normalized independent phrasings.
        groups = compress_tweets([by_id[x] for x in cited], threshold=0.82)
        bullet["confidence"] = "confirmed" if len(groups) >= 2 else "unconfirmed"

    for section in report.get("sections", []):
        for subsection in section.get("subsections", []):
            before = len(subsection["bullets"])
            subsection["bullets"] = [{k: v for k, v in b.items() if k != "_drop"} for b in subsection["bullets"] if not b.get("_drop")]
            stats["bullets_dropped"] += before - len(subsection["bullets"])
        section["subsections"] = [s for s in section["subsections"] if s["bullets"]]
    report["sections"] = [s for s in report["sections"] if s["subsections"]]
    return normalize_report(report), dict(stats)


def quality_score(report: dict, cell: Cell) -> tuple[float, dict]:
    by_id = {t["id"]: t for t in cell.tweets}
    scored = [(sid, b) for sid, _, b in _all_bullets(report) if sid not in {"1", "2"}]
    support, unsupported_numbers, duplicates = [], 0, 0
    texts = []
    for _, bullet in scored:
        claim = content_tokens(bullet["text"])
        cited = [x for x in bullet["tweet_ids"] if x in by_id]
        support.append(max((jaccard(claim, content_tokens(by_id[x]["text"])) for x in cited), default=0))
        evidence = " ".join(by_id[x]["text"] for x in cited)
        unsupported_numbers += bool(_numbers(bullet["text"]) - _numbers(evidence))
        texts.append(claim)
    for i, tokens in enumerate(texts):
        duplicates += any(jaccard(tokens, earlier) > 0.75 for earlier in texts[:i])
    represented = len({ss["title"] for s in report.get("sections", []) if s["id"] not in {"1", "2"} for ss in s["subsections"]})
    components = {
        "n_scored_bullets": len(scored),
        "represented_subsections": represented,
        "mean_lexical_support": sum(support) / max(1, len(support)),
        "duplicate_bullets": duplicates,
        "unsupported_number_bullets": unsupported_numbers,
    }
    score = 0.30 * min(len(scored), 50) / 50 + 0.25 * min(represented, 12) / 12 + 0.45 * components["mean_lexical_support"] - 0.03 * duplicates - 0.08 * unsupported_numbers
    return score, components


def prepare_r1(cell: Cell) -> tuple[list[dict], list[str], dict]:
    relevant = relevant_tweets(cell)
    groups = compress_tweets(relevant)
    return groups, evidence_lines(groups), {"input_tweets": len(cell.tweets), "relevant_tweets": len(relevant), "compressed_groups": len(groups)}
