"""Deterministic approximation of the public AML textual chunk contract."""
from __future__ import annotations

import re

from .schema import BenchmarkMessage

_WORD = re.compile(r"[A-Za-z0-9_]+|[\u3400-\u9fff]")


def adapter_word_count(text: str) -> int:
    """Count Latin words/numbers and individual CJK characters deterministically."""
    return len(_WORD.findall(text))


def chunk_messages(messages: tuple[BenchmarkMessage, ...], *, max_messages: int = 20,
                   max_words: int = 2000) -> list[tuple[BenchmarkMessage, ...]]:
    if max_messages < 1 or max_words < 1:
        raise ValueError("chunk limits must be positive")
    chunks: list[tuple[BenchmarkMessage, ...]] = []
    current: list[BenchmarkMessage] = []
    words = 0
    for message in messages:
        message_words = adapter_word_count(message.content)
        if current and (len(current) >= max_messages or words + message_words > max_words):
            chunks.append(tuple(current))
            current = []
            words = 0
        current.append(message)
        words += message_words
    if current:
        chunks.append(tuple(current))
    return chunks
