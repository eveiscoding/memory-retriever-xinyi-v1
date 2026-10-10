"""PersonaMem-v2 32K adapter (retrieval pipeline; final scoring needs Answer)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .common import parse_maybe_literal, text_value
from ..schema import BenchmarkCase, BenchmarkMessage, BenchmarkProbe, BenchmarkSession


def load_personamem(csv_path: str | Path, *, chat_history_dir: str | Path,
                    limit: int | None = None) -> list[BenchmarkCase]:
    rows_by_persona: dict[str, list[tuple[int, dict]]] = {}
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        for row_index, row in enumerate(csv.DictReader(handle)):
            rows_by_persona.setdefault(str(row["persona_id"]), []).append((row_index, row))
    root = Path(chat_history_dir)
    cases: list[BenchmarkCase] = []
    probe_count = 0
    for persona_id, rows in rows_by_persona.items():
        link = str(rows[0][1].get("chat_history_32k_link", "")).replace("\\", "/")
        candidates = [root / Path(link).name, root / f"chat_history_32k_persona{persona_id}.json"]
        history_path = next((candidate for candidate in candidates if candidate.exists()), None)
        if history_path is None:
            continue
        payload = json.loads(history_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            raw_history = payload
        elif isinstance(payload, dict):
            raw_history = payload.get("chat_history", payload.get("messages",
                                      payload.get("conversations", [])))
            if not isinstance(raw_history, list):
                raw_history = next((value.get("conversations", [])
                                    for value in payload.values()
                                    if isinstance(value, dict) and isinstance(
                                        value.get("conversations"), list)), [])
        else:
            raw_history = []
        messages = []
        for index, raw in enumerate(raw_history):
            if not isinstance(raw, dict) or str(raw.get("role")) == "system":
                continue
            body = text_value(raw.get("content", raw.get("text")))
            if body:
                role = str(raw.get("role", "user"))
                messages.append(BenchmarkMessage(
                    f"persona-{persona_id}:{index}", role if role in {"user", "assistant"} else "user",
                    body, 1_700_000_000_000 + index * 1000,
                ))
        probes = []
        for row_index, row in rows:
            if limit is not None and probe_count >= limit:
                break
            parsed_query = parse_maybe_literal(row.get("user_query", ""))
            query = text_value(parsed_query)
            incorrect = parse_maybe_literal(row.get("incorrect_answers", "[]"))
            if not isinstance(incorrect, list):
                incorrect = []
            correct = row.get("correct_answer", row.get("answer", ""))
            options = [str(correct), *(str(value) for value in incorrect)]
            probes.append(BenchmarkProbe(
                probe_id=f"persona-{persona_id}:q{row_index}", query=query,
                capability=f"personamem-{row.get('pref_type', 'unknown')}",
                options=tuple(option for option in options if option),
            ))
            probe_count += 1
        if messages and probes:
            case = BenchmarkCase(
                f"persona-{persona_id}", "personamem-v2",
                (BenchmarkSession(f"persona-{persona_id}:history", tuple(messages)),),
                tuple(probes), {"scoring": "answer-model-required", "persona_id": persona_id},
            )
            case.validate()
            cases.append(case)
        if limit is not None and probe_count >= limit:
            break
    return cases

