"""Adapter for snap-research/locomo ``data/locomo10.json``."""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

from ..schema import BenchmarkCase, BenchmarkMessage, BenchmarkProbe, BenchmarkSession

_SESSION = re.compile(r"^session_(\d+)$")


def _timestamp(value: object, fallback: int) -> int:
    text = str(value or "").strip()
    formats = (
        "%I:%M %p on %d %B, %Y", "%I:%M %p on %d %B %Y",
        "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M",
    )
    for fmt in formats:
        try:
            parsed = dt.datetime.strptime(text, fmt).replace(tzinfo=dt.timezone.utc)
            return int(parsed.timestamp() * 1000)
        except ValueError:
            continue
    return fallback


def load_locomo(path: str | Path) -> list[BenchmarkCase]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("LoCoMo file must contain a JSON array")
    cases: list[BenchmarkCase] = []
    for sample_index, sample in enumerate(raw):
        conversation = sample["conversation"]
        case_id = str(sample.get("sample_id", sample_index))
        speaker_a = str(conversation.get("speaker_a", "speaker_a"))
        speaker_b = str(conversation.get("speaker_b", "speaker_b"))
        sessions: list[BenchmarkSession] = []
        evidence: dict[str, str] = {}
        keys = sorted(
            (key for key in conversation if _SESSION.match(key)),
            key=lambda key: int(_SESSION.match(key).group(1)),
        )
        for session_offset, key in enumerate(keys):
            session_number = _SESSION.match(key).group(1)
            base = _timestamp(
                conversation.get(f"session_{session_number}_date_time"),
                1_700_000_000_000 + session_offset * 86_400_000,
            )
            messages: list[BenchmarkMessage] = []
            for turn_index, turn in enumerate(conversation[key]):
                dialog_id = str(turn.get("dia_id", f"{key}-{turn_index}"))
                speaker = str(turn.get("speaker", "speaker"))
                text = str(turn.get("text", "")).strip()
                if not text:
                    continue
                content = f"{speaker}: {text}"
                evidence[dialog_id] = content
                role = "user" if speaker == speaker_a else "assistant" if speaker == speaker_b else "user"
                messages.append(BenchmarkMessage(dialog_id, role, content, base + turn_index * 1000))
            if messages:
                sessions.append(BenchmarkSession(f"{case_id}:{key}", tuple(messages)))

        probes: list[BenchmarkProbe] = []
        for question_index, question in enumerate(sample.get("qa", [])):
            evidence_ids = question.get("evidence") or []
            markers = tuple(evidence[str(item)] for item in evidence_ids if str(item) in evidence)
            probes.append(BenchmarkProbe(
                probe_id=f"{case_id}:q{question_index}",
                query=str(question["question"]),
                capability=f"locomo-category-{question.get('category', 'unknown')}",
                gold_evidence=markers,
            ))
        if sessions and probes:
            case = BenchmarkCase(case_id, "locomo", tuple(sessions), tuple(probes), {
                "speaker_a": speaker_a, "speaker_b": speaker_b,
            })
            case.validate()
            cases.append(case)
    return cases
