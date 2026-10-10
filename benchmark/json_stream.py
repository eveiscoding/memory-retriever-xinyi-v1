"""Incrementally read a top-level JSON array using only the standard library."""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path


def iter_json_array(path: str | Path, *, chunk_size: int = 1024 * 1024) -> Iterator[object]:
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    started = False
    finished = False

    with Path(path).open("r", encoding="utf-8") as handle:
        while not finished:
            chunk = handle.read(chunk_size)
            eof = not chunk
            buffer += chunk

            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if not started:
                    if position >= len(buffer):
                        break
                    if buffer[position] != "[":
                        raise ValueError(f"{path}: expected a top-level JSON array")
                    position += 1
                    started = True
                    continue

                while position < len(buffer) and (buffer[position].isspace() or buffer[position] == ","):
                    position += 1
                if position < len(buffer) and buffer[position] == "]":
                    position += 1
                    finished = True
                    break
                if position >= len(buffer):
                    break
                try:
                    value, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError as error:
                    if eof:
                        raise ValueError(f"{path}: incomplete or invalid JSON array") from error
                    break
                yield value
                position = end

            if position:
                buffer = buffer[position:]
                position = 0
            if eof and not finished:
                raise ValueError(f"{path}: JSON array did not terminate")
