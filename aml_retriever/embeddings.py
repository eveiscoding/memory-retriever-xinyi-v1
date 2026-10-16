"""Pluggable dense embedding backends with a zero-dependency default path.

No backend is imported or constructed unless ``dense_backend`` is configured.
The built-in hashing backend is intentionally small and deterministic; it is
useful for smoke tests and offline ablations, while production-quality models
can be supplied as ``module:factory`` plugins.
"""
from __future__ import annotations

import hashlib
import importlib
import math
import re
from dataclasses import dataclass
from typing import Protocol, Sequence


class EmbeddingBackend(Protocol):
    identity: str
    model_hash: str
    dimension: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one fixed-width vector per input text."""


_WORD_RE = re.compile(r"[\w]+|[\u3400-\u9fff]", re.UNICODE)


@dataclass
class HashingEmbeddingBackend:
    """Deterministic signed feature hashing, enabled only when requested."""

    dimension: int = 256

    def __post_init__(self) -> None:
        if self.dimension < 8:
            raise ValueError("dense_dimension must be at least 8")
        self.identity = f"builtin:hashing-v1:{self.dimension}"
        self.model_hash = hashlib.sha256(self.identity.encode("utf-8")).hexdigest()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimension
            tokens = _WORD_RE.findall((text or "").lower())
            # Word/unigram features plus adjacent token pairs provide a compact
            # deterministic dense baseline without importing a model runtime.
            features = tokens + [f"{a}\x1f{b}" for a, b in zip(tokens, tokens[1:])]
            for feature in features:
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=16).digest()
                index = int.from_bytes(digest[:8], "little") % self.dimension
                vector[index] += 1.0 if digest[8] & 1 else -1.0
            norm = math.sqrt(sum(value * value for value in vector))
            if norm:
                vector = [value / norm for value in vector]
            vectors.append(vector)
        return vectors


def _validate_backend(backend) -> EmbeddingBackend:
    for attr in ("identity", "model_hash", "dimension", "embed"):
        if not hasattr(backend, attr):
            raise ValueError(f"embedding backend is missing {attr}")
    if int(backend.dimension) < 1:
        raise ValueError("embedding backend dimension must be positive")
    if not str(backend.identity).strip() or not str(backend.model_hash).strip():
        raise ValueError("embedding backend identity and model_hash are required")
    return backend


def load_embedding_backend(spec: str, *, dimension: int = 256) -> EmbeddingBackend | None:
    """Load a configured backend without touching optional dependencies otherwise.

    ``hashing-v1`` selects the built-in deterministic backend. Any other value
    must be ``module:factory``; the factory is called with ``dimension=...`` and
    must return an object satisfying :class:`EmbeddingBackend`.
    """
    name = (spec or "").strip()
    if not name:
        return None
    if name in {"hashing", "hashing-v1", "builtin:hashing-v1"}:
        return HashingEmbeddingBackend(dimension=int(dimension))
    if ":" not in name:
        raise ValueError("dense_backend must be hashing-v1 or module:factory")
    module_name, factory_name = name.rsplit(":", 1)
    factory = getattr(importlib.import_module(module_name), factory_name)
    return _validate_backend(factory(dimension=int(dimension)))


__all__ = [
    "EmbeddingBackend",
    "HashingEmbeddingBackend",
    "load_embedding_backend",
]
