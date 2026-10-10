from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import re


class LLMClient:
    def __init__(self, base_url: str, model: str, timeout: int = 3600):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.timeout = timeout

    def chat(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 8192, json_schema: dict | None = None, seed: int | None = None) -> tuple[str, dict]:
        if json_schema and os.environ.get("LLM_SCHEMA_FORMAT") == "json_object":
            response_format = {"type": "json_object", "schema": json_schema}
        elif json_schema:
            response_format = {"type": "json_schema", "json_schema": {"name": "response", "strict": True, "schema": json_schema}}
        else:
            response_format = {"type": "json_object"}
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
        }
        if seed is not None:
            body["seed"] = seed
        request = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"LLM HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc
        elapsed = time.perf_counter() - started
        choice = payload["choices"][0]["message"]["content"]
        return choice, {"elapsed_seconds": elapsed, "usage": payload.get("usage", {}), "seed": seed}


def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        recovered = _recover_complete_sections(text) or _recover_complete_facts(text)
        if recovered is not None:
            return recovered
        raise


def _recover_complete_sections(text: str) -> dict | None:
    """Recover only fully closed section objects from a length-truncated report."""
    sections = _recover_complete_array_objects(text, "sections")
    return {"sections": sections, "_recovered_truncation": True} if sections else None


def _recover_complete_facts(text: str) -> dict | None:
    """Recover only fully closed fact objects from a length-truncated ledger."""
    facts = _recover_complete_array_objects(text, "facts")
    return {"facts": facts, "_recovered_truncation": True} if facts else None


def _recover_complete_array_objects(text: str, key: str) -> list[dict]:
    """Return complete top-level objects from a named JSON array."""
    match = re.search(rf'"{re.escape(key)}"\s*:\s*\[', text)
    if not match:
        return []
    array_start = text.find("[", match.start())
    stack: list[str] = []
    in_string = False
    escaped = False
    section_start: int | None = None
    objects: list[dict] = []
    for index in range(array_start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            if char == "{" and stack == ["["]:
                section_start = index
            stack.append(char)
        elif char in "]}":
            if not stack:
                break
            expected = "[" if char == "]" else "{"
            if stack[-1] != expected:
                break
            stack.pop()
            if char == "}" and stack == ["["] and section_start is not None:
                try:
                    section = json.loads(text[section_start:index + 1])
                except json.JSONDecodeError:
                    section_start = None
                    continue
                if isinstance(section, dict):
                    objects.append(section)
                section_start = None
    return objects
