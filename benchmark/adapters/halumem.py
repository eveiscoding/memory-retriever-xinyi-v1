"""Evidence-aware adapter for IAAR-Shanghai/HaluMem JSONL."""
from __future__ import annotations

from pathlib import Path

from .common import read_records, text_value, timestamp_ms
from ..schema import BenchmarkCase, BenchmarkMessage, BenchmarkProbe, BenchmarkSession


def _strings(value) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(_strings(item))
        return result
    if isinstance(value, dict):
        for key in ("content", "text", "memory_content", "memory_source", "evidence"):
            if key in value:
                return _strings(value[key])
    return []


def load_halumem(path: str | Path, *, limit: int | None = None) -> list[BenchmarkCase]:
    cases = []
    probe_count = 0
    for row_index, user in enumerate(read_records(path)):
        case_id = str(user.get("uuid", row_index))
        sessions = []
        evidence_by_index: dict[str, tuple[str, ...]] = {}
        probes = []
        for session_index, raw_session in enumerate(user.get("sessions", [])):
            base = timestamp_ms(raw_session.get("start_time", raw_session.get("end_time")),
                                1_700_000_000_000 + session_index * 86_400_000)
            messages = []
            for message_index, raw in enumerate(raw_session.get("dialogue", [])):
                body = text_value(raw.get("content", raw.get("text"))) if isinstance(raw, dict) else text_value(raw)
                if not body:
                    continue
                role = str(raw.get("role", "user")) if isinstance(raw, dict) else "user"
                messages.append(BenchmarkMessage(
                    f"{case_id}:{session_index}:{message_index}",
                    role if role in {"user", "assistant"} else "user", body,
                    timestamp_ms(raw.get("timestamp") if isinstance(raw, dict) else None,
                                 base + message_index * 1000),
                ))
            if messages:
                sessions.append(BenchmarkSession(f"{case_id}:session-{session_index}", tuple(messages)))
            for point in raw_session.get("memory_points", []):
                if not isinstance(point, dict):
                    continue
                alternatives = tuple(dict.fromkeys(_strings(point.get("memory_content"))))
                evidence_by_index[str(point.get("index"))] = alternatives
            for question_index, question in enumerate(raw_session.get("questions", [])):
                if limit is not None and probe_count >= limit:
                    break
                groups = []
                for evidence in question.get("evidence", []) if isinstance(question, dict) else []:
                    alternatives = evidence_by_index.get(str(evidence), tuple(_strings(evidence)))
                    if alternatives:
                        groups.append(alternatives)
                probes.append(BenchmarkProbe(
                    probe_id=f"{case_id}:{session_index}:q{question_index}",
                    query=text_value(question.get("question")),
                    capability=f"halumem-{question.get('question_type', 'unknown')}",
                    gold_evidence_groups=tuple(groups),
                    after_session=len(sessions) - 1 if sessions else None,
                ))
                probe_count += 1
        if sessions and probes:
            case = BenchmarkCase(case_id, "halumem", tuple(sessions), tuple(probes), {
                "scoring": "evidence-when-available",
            })
            case.validate()
            cases.append(case)
        if limit is not None and probe_count >= limit:
            break
    return cases

