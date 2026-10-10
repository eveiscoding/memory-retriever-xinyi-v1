"""Small format helpers shared by public benchmark adapters."""
from __future__ import annotations

import ast
import datetime as dt
import json
from pathlib import Path
from typing import Any


def read_records(path: str | Path) -> list[dict]:
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        return [
            json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"{source} must contain JSON objects")


def timestamp_ms(value: object, fallback: int) -> int:
    text = str(value or "").strip()
    if not text:
        return fallback
    if text.isdigit():
        number = int(text)
        return number if number > 10_000_000_000 else number * 1000
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return int(parsed.timestamp() * 1000)
    except ValueError:
        return fallback


def text_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return str(value.get("content", value.get("text", ""))).strip()
    if isinstance(value, list):
        return "\n".join(filter(None, (text_value(item) for item in value)))
    return str(value or "").strip()


def parse_maybe_literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            return value

