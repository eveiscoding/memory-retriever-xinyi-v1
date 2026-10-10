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


def load_longmemeval(path: str | Path, *, limit: int | None = None,
                     per_capability: int | None = None,
                     dataset: str = "longmemeval-s") -> list[BenchmarkCase]:
    if per_capability is not None and per_capability < 1:
        raise ValueError("per_capability must be positive")
    cases: list[BenchmarkCase] = []
    capability_counts: dict[str, int] = {}
    for row_index, item in enumerate(iter_json_array(path)):
        if limit is not None and len(cases) >= limit:
            break
        if not isinstance(item, dict):
            raise ValueError(f"LongMemEval row {row_index} must be an object")
        capability = str(item.get("question_type", "unknown"))
        if per_capability is not None and capability_counts.get(capability, 0) >= per_capability:
            continue
        question_id = str(item.get("question_id", row_index))
        raw_sessions = item.get("haystack_sessions") or []
        session_ids = item.get("haystack_session_ids") or []
        session_dates = item.get("haystack_dates") or []
        answer_session_ids = {str(value) for value in item.get("answer_session_ids") or []}
        sessions: list[BenchmarkSession] = []
        marked_evidence: list[str] = []
        answer_session_fallbacks: dict[str, list[str]] = {}

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
                    message_id=f"{question_id}:{source_session_id}:{session_index}:{message_index}",
                    role=role,
                    content=content,
                    timestamp=base_timestamp + message_index * 1000,
                )
                messages.append(message)
                if raw_message.get("has_answer") is True:
                    marked_evidence.append(content)
                if source_session_id in answer_session_ids:
                    answer_session_fallbacks.setdefault(source_session_id, []).append(content)
            if messages:
                sessions.append(BenchmarkSession(
                    f"{source_session_id}:{session_index}", tuple(messages)
                ))

        evidence = tuple(dict.fromkeys(marked_evidence))
        used_session_fallback = False
        evidence_groups: tuple[tuple[str, ...], ...] = ()
        if not evidence and answer_session_fallbacks:
            evidence_groups = tuple(
                tuple(dict.fromkeys(markers))
                for session_id, markers in answer_session_fallbacks.items()
                if session_id in answer_session_ids and markers
            )
            used_session_fallback = True
        probe = BenchmarkProbe(
            probe_id=question_id,
            query=str(item.get("question", "")).strip(),
            capability=capability,
            gold_evidence=evidence,
            gold_evidence_groups=evidence_groups,
        )
        if sessions and probe.query:
            case = BenchmarkCase(
                case_id=question_id,
                dataset=dataset,
                sessions=tuple(sessions),
                probes=(probe,),
                metadata={
                    "question_date": item.get("question_date"),
                    "has_answer": bool(evidence or evidence_groups),
                    "session_fallback": used_session_fallback,
                },
            )
            case.validate()
            cases.append(case)
            capability_counts[capability] = capability_counts.get(capability, 0) + 1
    return cases
