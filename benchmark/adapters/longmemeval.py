"""Adapter for LongMemEval-S and compatible cleaned variants."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

from ..json_stream import iter_json_array
from ..schema import BenchmarkCase, BenchmarkMessage, BenchmarkProbe, BenchmarkSession


def _date_timestamp(value: object, fallback: int) -> int:
    text = str(value or "").strip()
    for fmt in ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            parsed = dt.datetime.strptime(text, fmt).replace(tzinfo=dt.timezone.utc)
            return int(parsed.timestamp() * 1000)
        except ValueError:
            continue
    return fallback


def load_longmemeval(path: str | Path, *, limit: int | None = None) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for row_index, item in enumerate(iter_json_array(path)):
        if limit is not None and len(cases) >= limit:
            break
        if not isinstance(item, dict):
            raise ValueError(f"LongMemEval row {row_index} must be an object")
        question_id = str(item.get("question_id", row_index))
        raw_sessions = item.get("haystack_sessions") or []
        session_ids = item.get("haystack_session_ids") or []
        session_dates = item.get("haystack_dates") or []
        answer_session_ids = {str(value) for value in item.get("answer_session_ids") or []}
        sessions: list[BenchmarkSession] = []
        marked_evidence: list[str] = []
        answer_session_fallbacks: list[str] = []

        for session_index, raw_messages in enumerate(raw_sessions):
            source_session_id = str(
                session_ids[session_index] if session_index < len(session_ids)
                else f"session-{session_index}"
            )
            base_timestamp = _date_timestamp(
                session_dates[session_index] if session_index < len(session_dates) else None,
                1_700_000_000_000 + session_index * 86_400_000,
            )
            messages: list[BenchmarkMessage] = []
            for message_index, raw_message in enumerate(raw_messages or []):
                if not isinstance(raw_message, dict):
                    continue
                content = str(raw_message.get("content", "")).strip()
                if not content:
                    continue
                role = str(raw_message.get("role", "user")).strip() or "user"
                message = BenchmarkMessage(
                    message_id=f"{question_id}:{source_session_id}:{message_index}",
                    role=role,
                    content=content,
                    timestamp=base_timestamp + message_index * 1000,
                )
                messages.append(message)
                if raw_message.get("has_answer") is True:
                    marked_evidence.append(content)
                if source_session_id in answer_session_ids:
                    answer_session_fallbacks.append(content)
            if messages:
                sessions.append(BenchmarkSession(source_session_id, tuple(messages)))

        evidence = tuple(dict.fromkeys(marked_evidence))
        used_session_fallback = False
        if not evidence and answer_session_fallbacks:
            evidence = (answer_session_fallbacks[0],)
            used_session_fallback = True
        probe = BenchmarkProbe(
            probe_id=question_id,
            query=str(item.get("question", "")).strip(),
            capability=str(item.get("question_type", "unknown")),
            gold_evidence=evidence,
        )
        if sessions and probe.query:
            case = BenchmarkCase(
                case_id=question_id,
                dataset="longmemeval-s",
                sessions=tuple(sessions),
                probes=(probe,),
                metadata={
                    "question_date": item.get("question_date"),
                    "has_answer": bool(evidence),
                    "session_fallback": used_session_fallback,
                },
            )
            case.validate()
            cases.append(case)
    return cases
