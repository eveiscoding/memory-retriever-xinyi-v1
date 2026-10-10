"""Adapter for Mohammadta/BEAM JSONL exports."""
from __future__ import annotations

from pathlib import Path

from .common import parse_maybe_literal, read_records, text_value, timestamp_ms
from ..schema import BenchmarkCase, BenchmarkMessage, BenchmarkProbe, BenchmarkSession


def _unwrap_batches(batch_dicts) -> list[list[dict]]:
    batches = []
    for batch in batch_dicts or []:
        if not isinstance(batch, dict):
            continue
        turns = []
        for item in batch.get("turns", []):
            if isinstance(item, list):
                turns.extend(turn for turn in item if isinstance(turn, dict))
            elif isinstance(item, dict):
                turns.append(item)
        if turns:
            batches.append(turns)
    return batches


def _chat_batches(value) -> list[list[dict]]:
    """Normalize the three structures published by the official BEAM datasets."""
    if not isinstance(value, list) or not value:
        return []
    first = value[0]
    if isinstance(first, list):
        return [[turn for turn in batch if isinstance(turn, dict)] for batch in value]
    if not isinstance(first, dict):
        return []
    if "turns" in first:
        return _unwrap_batches(value)
    if "role" in first or "content" in first:
        return [[turn for turn in value if isinstance(turn, dict)]]
    batches = []
    for session in value:
        if not isinstance(session, dict):
            continue
        for key in sorted(session, key=lambda item: (
                int(item.rsplit("-", 1)[-1]) if item.rsplit("-", 1)[-1].isdigit() else 0)):
            batches.extend(_unwrap_batches(session.get(key)))
    return batches


def _questions(item: dict):
    raw = parse_maybe_literal(item.get("probing_questions", {}))
    if isinstance(raw, list):
        raw = {"unknown": raw}
    if not isinstance(raw, dict):
        return
    for dimension, values in raw.items():
        for question in values or []:
            if isinstance(question, dict):
                yield str(dimension), question


def load_beam(path: str | Path, *, limit: int | None = None) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    probe_count = 0
    for row_index, item in enumerate(read_records(path)):
        case_id = str(item.get("conversation_id", row_index))
        sessions = []
        for session_index, raw_messages in enumerate(_chat_batches(item.get("chat", []))):
            base = timestamp_ms(raw_messages[0].get("time_anchor") if raw_messages else None,
                                1_700_000_000_000 + session_index * 86_400_000)
            messages = []
            for message_index, raw in enumerate(raw_messages):
                body = text_value(raw.get("content", raw.get("text")))
                if not body:
                    continue
                role = str(raw.get("role", "user"))
                messages.append(BenchmarkMessage(
                    f"{case_id}:{session_index}:{message_index}",
                    role if role in {"user", "assistant"} else "user", body,
                    timestamp_ms(raw.get("time_anchor"), base + message_index * 1000),
                ))
            if messages:
                sessions.append(BenchmarkSession(f"{case_id}:session-{session_index}", tuple(messages)))
        probes = []
        for question_index, (dimension, question) in enumerate(_questions(item) or []):
            if limit is not None and probe_count >= limit:
                break
            options = question.get("options") or []
            probes.append(BenchmarkProbe(
                probe_id=f"{case_id}:q{question_index}",
                query=text_value(question.get("question_text", question.get("question"))),
                capability=f"beam-{dimension}",
                options=tuple(text_value(option) for option in options),
            ))
            probe_count += 1
        if sessions and probes:
            case = BenchmarkCase(case_id, "beam", tuple(sessions), tuple(probes), {
                "scoring": "answer-model-required", "scale": item.get("_scale", "unknown"),
            })
            case.validate()
            cases.append(case)
        if limit is not None and probe_count >= limit:
            break
    return cases

