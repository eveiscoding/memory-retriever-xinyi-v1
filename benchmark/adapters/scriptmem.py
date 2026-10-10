"""Adapter for ScriptMem bundles that include legally obtained source dialogue."""
from __future__ import annotations

import re
from pathlib import Path

from .common import read_records, text_value, timestamp_ms
from ..schema import BenchmarkCase, BenchmarkMessage, BenchmarkProbe, BenchmarkSession

_SESSION = re.compile(r"^session_(\d+)$")
_FILES = ("angry.json", "enemy.json", "friends.json", "man_earth.json")


def load_scriptmem(path: str | Path) -> list[BenchmarkCase]:
    source = Path(path)
    paths = [source / name for name in _FILES if (source / name).exists()] if source.is_dir() else [source]
    cases: list[BenchmarkCase] = []
    placeholder_only = False
    for file in paths:
        for sample_index, sample in enumerate(read_records(file)):
            case_id = str(sample.get("sample_id", f"{file.stem}-{sample_index}"))
            conversation = sample.get("conversation") or {}
            session_keys = sorted(
                (key for key in conversation if _SESSION.match(key)),
                key=lambda key: int(_SESSION.match(key).group(1)),
            )
            if not session_keys and conversation.get("format_example"):
                placeholder_only = True
                continue
            sessions: list[BenchmarkSession] = []
            for session_index, key in enumerate(session_keys):
                raw_messages = conversation.get(key) or []
                base = timestamp_ms(conversation.get(f"{key}_date_time"),
                                    1_700_000_000_000 + session_index * 86_400_000)
                messages = []
                for message_index, raw in enumerate(raw_messages):
                    body = text_value(raw.get("text", raw.get("content"))) if isinstance(raw, dict) else text_value(raw)
                    if not body:
                        continue
                    role = str(raw.get("role") or ("assistant" if raw.get("type") == "narration" else "user"))
                    messages.append(BenchmarkMessage(
                        str(raw.get("dia_id", f"{case_id}:{session_index}:{message_index}")),
                        role if role in {"user", "assistant"} else "user", body,
                        base + message_index * 1000,
                    ))
                if messages:
                    sessions.append(BenchmarkSession(f"{case_id}:{key}", tuple(messages)))
            probes = tuple(
                BenchmarkProbe(
                    probe_id=f"{file.stem}:{case_id}:q{index:04d}",
                    query=text_value(qa.get("question")),
                    capability=f"scriptmem-{qa.get('qa_type', 'unknown')}",
                    options=tuple(text_value(option) for option in qa.get("option", qa.get("options", []))),
                )
                for index, qa in enumerate(sample.get("qa", []))
                if text_value(qa.get("question"))
            )
            if sessions and probes:
                case = BenchmarkCase(case_id, "scriptmem", tuple(sessions), probes, {
                    "scoring": "answer-model-required", "source": file.stem,
                })
                case.validate()
                cases.append(case)
    if not cases and placeholder_only:
        raise ValueError(
            "ScriptMem public files omit the copyrighted source dialogue. "
            "Provide a local, legally obtained bundle containing conversation.session_N fields."
        )
    return cases

