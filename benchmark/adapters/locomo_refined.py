"""Adapters for mem-eval-suite/LoCoMo_refined public and annotated formats."""
from __future__ import annotations

import dataclasses
from pathlib import Path

from .common import read_records, text_value, timestamp_ms
from .locomo import load_locomo
from ..schema import BenchmarkCase, BenchmarkMessage, BenchmarkProbe, BenchmarkSession


def _public_sessions(item: dict, case_id: str) -> tuple[BenchmarkSession, ...]:
    raw_sessions = item.get("sessions") or item.get("conversation", {}).get("sessions") or []
    sessions: list[BenchmarkSession] = []
    for session_index, raw_session in enumerate(raw_sessions):
        messages_source = raw_session.get("messages", raw_session) if isinstance(raw_session, dict) else raw_session
        if not isinstance(messages_source, list):
            continue
        base = timestamp_ms(
            raw_session.get("date_time", raw_session.get("date", raw_session.get("created_at")))
            if isinstance(raw_session, dict) else None,
            1_700_000_000_000 + session_index * 86_400_000,
        )
        messages: list[BenchmarkMessage] = []
        for message_index, raw in enumerate(messages_source):
            if not isinstance(raw, dict):
                continue
            body = text_value(raw.get("text", raw.get("content")))
            if not body:
                continue
            speaker = str(raw.get("speaker", raw.get("role", "user")))
            role = str(raw.get("role") or ("assistant" if speaker.lower() == "assistant" else "user"))
            content = body if role in {"user", "assistant"} else f"{speaker}: {body}"
            ident = str(raw.get("dia_id", raw.get("id", f"{case_id}:{session_index}:{message_index}")))
            messages.append(BenchmarkMessage(ident, role if role in {"user", "assistant"} else "user",
                                             content, base + message_index * 1000))
        if messages:
            session_id = str(raw_session.get("session_id", f"{case_id}:session-{session_index}")) \
                if isinstance(raw_session, dict) else f"{case_id}:session-{session_index}"
            sessions.append(BenchmarkSession(session_id, tuple(messages)))
    return tuple(sessions)


def load_locomo_refined(path: str | Path, *, questions_path: str | Path | None = None) -> list[BenchmarkCase]:
    """Load the full annotated JSON or the two-file public JSONL export.

    Public questions may carry evidence IDs and resolved evidence messages. When
    present, those annotations are used for retrieval scoring; otherwise the run
    still exercises Add/Search and latency without inventing gold evidence.
    """
    if questions_path is None and Path(path).suffix.lower() == ".json":
        cases = load_locomo(path)
        return [dataclasses.replace(case, dataset="locomo-refined") for case in cases]

    conversations = read_records(path)
    questions = read_records(questions_path or Path(path).with_name("questions.jsonl"))
    by_sample: dict[str, list[dict]] = {}
    for row in questions:
        by_sample.setdefault(str(row.get("sample_id", row.get("conversation_id", ""))), []).append(row)
    cases: list[BenchmarkCase] = []
    for index, item in enumerate(conversations):
        case_id = str(item.get("sample_id", item.get("conversation_id", index)))
        sessions = _public_sessions(item, case_id)
        evidence_by_id = {
            message.message_id: message.content
            for session in sessions for message in session.messages
        }
        probes_list = []
        has_public_evidence = False
        for i, row in enumerate(by_sample.get(case_id, [])):
            query = text_value(row.get("question"))
            if not query:
                continue
            markers = []
            for evidence in row.get("evidence_messages", []) or []:
                marker = text_value(evidence)
                if marker:
                    markers.append(marker)
            for evidence_id in row.get("evidence", []) or []:
                marker = evidence_by_id.get(str(evidence_id))
                if marker:
                    markers.append(marker)
            markers = list(dict.fromkeys(markers))
            has_public_evidence = has_public_evidence or bool(markers)
            probes_list.append(BenchmarkProbe(
                probe_id=str(row.get("qa_id", row.get("question_id", f"{case_id}:q{i}"))),
                query=query,
                capability=f"locomo-refined-{row.get('category', 'open-ended')}",
                gold_evidence=tuple(markers),
            ))
        probes = tuple(probes_list)
        if sessions and probes:
            case = BenchmarkCase(case_id, "locomo-refined", sessions, probes, {
                "scoring": "evidence-when-available" if has_public_evidence
                else "answer-model-required",
                "public_evidence": has_public_evidence,
            })
            case.validate()
            cases.append(case)
    return cases

