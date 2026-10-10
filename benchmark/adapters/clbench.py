"""Adapter for AML-style and Tencent CL-bench JSONL records."""
from __future__ import annotations

from pathlib import Path

from .common import read_records, text_value
from ..schema import BenchmarkCase, BenchmarkMessage, BenchmarkProbe, BenchmarkSession


def load_clbench(path: str | Path, *, limit: int | None = None) -> list[BenchmarkCase]:
    cases = []
    for row_index, item in enumerate(read_records(path)):
        if limit is not None and len(cases) >= limit:
            break
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        case_id = str(item.get("idx", item.get("id", metadata.get("task_id", row_index))))
        query = text_value(item.get("question", item.get("task")))
        context = item.get("context", item.get("memories"))
        raw_messages = item.get("messages") if isinstance(item.get("messages"), list) else []
        memory_messages = []
        if context is not None:
            body = text_value(context)
            if body:
                memory_messages.append(BenchmarkMessage(f"{case_id}:context", "user", body,
                                                        1_700_000_000_000))
        else:
            if not query and raw_messages:
                query = text_value(raw_messages[-1])
                raw_messages = raw_messages[:-1]
            for message_index, raw in enumerate(raw_messages):
                if not isinstance(raw, dict):
                    continue
                body = text_value(raw.get("content", raw.get("text")))
                if not body:
                    continue
                role = str(raw.get("role", "user"))
                if role == "system":
                    body = f"[system] {body}"
                    role = "user"
                memory_messages.append(BenchmarkMessage(
                    f"{case_id}:{message_index}", role if role in {"user", "assistant"} else "user",
                    body, 1_700_000_000_000 + message_index * 1000,
                ))
        if not query and raw_messages:
            query = text_value(raw_messages[-1])
        if memory_messages and query:
            case = BenchmarkCase(
                case_id, "clbench",
                (BenchmarkSession(f"{case_id}:context", tuple(memory_messages)),),
                (BenchmarkProbe(
                    probe_id=f"{case_id}:q0", query=query,
                    capability=f"clbench-{metadata.get('context_category', item.get('category', 'unknown'))}",
                    options=tuple(text_value(option) for option in item.get("options", [])),
                ),),
                {"scoring": "answer-model-required", "rubrics": item.get("rubrics", metadata.get("rubrics"))},
            )
            case.validate()
            cases.append(case)
    return cases

