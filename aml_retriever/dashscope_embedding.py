"""Alibaba Cloud Model Studio ``text-embedding-v4`` backend.

The backend uses the OpenAI-compatible embeddings endpoint with only Python's
standard library. Credentials are read lazily when this backend is selected.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Sequence


_MODEL = "text-embedding-v4"
_ALLOWED_DIMENSIONS = {64, 128, 256, 512, 768, 1024, 1536, 2048}
_MAX_BATCH_SIZE = 10


@dataclass
class DashScopeEmbeddingBackend:
    """Generate dense vectors through Model Studio's compatible endpoint."""

    dimension: int = 1024
    api_key: str | None = None
    base_url: str | None = None
    timeout_seconds: float = 30.0
    max_retries: int = 3

    def __post_init__(self) -> None:
        self.dimension = int(self.dimension)
        if self.dimension not in _ALLOWED_DIMENSIONS:
            allowed = ", ".join(str(value) for value in sorted(_ALLOWED_DIMENSIONS))
            raise ValueError(f"text-embedding-v4 dimension must be one of: {allowed}")

        self.api_key = (self.api_key or os.environ.get("DASHSCOPE_API_KEY", "")).strip()
        self.base_url = (
            self.base_url or os.environ.get("DASHSCOPE_BASE_URL", "")
        ).strip().rstrip("/")
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY is required for text-embedding-v4")
        if not self.base_url:
            raise ValueError(
                "DASHSCOPE_BASE_URL is required; use the workspace-specific "
                "OpenAI-compatible /v1 base URL"
            )
        if self.timeout_seconds <= 0:
            raise ValueError("embedding timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("embedding max_retries cannot be negative")

        self.identity = f"dashscope:{_MODEL}:{self.dimension}"
        self.model_hash = hashlib.sha256(self.identity.encode("utf-8")).hexdigest()

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/embeddings"

    @staticmethod
    def _error_message(exc: urllib.error.HTTPError) -> str:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
            error = payload.get("error", {})
            message = error.get("message") or payload.get("message")
            code = error.get("code") or payload.get("code")
            if message and code:
                return f"{code}: {message}"
            if message:
                return str(message)
        except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
            pass
        return exc.reason or "request failed"

    def _request_batch(self, texts: list[str]) -> list[list[float]]:
        body = json.dumps(
            {
                "model": _MODEL,
                "input": texts,
                "dimensions": self.dimension,
                "encoding_format": "float",
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    payload = json.load(response)
                break
            except urllib.error.HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt >= self.max_retries:
                    detail = self._error_message(exc)
                    raise RuntimeError(
                        f"DashScope embedding request failed with HTTP {exc.code}: {detail}"
                    ) from exc
            except urllib.error.URLError as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"DashScope embedding request failed: {exc.reason}"
                    ) from exc
            time.sleep(0.5 * (2**attempt))

        data = payload.get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            raise RuntimeError("DashScope returned an unexpected embedding count")

        indexed: dict[int, list[float]] = {}
        for item in data:
            if not isinstance(item, dict) or "index" not in item:
                raise RuntimeError("DashScope returned an invalid embedding item")
            index = int(item["index"])
            raw_vector = item.get("embedding")
            if not isinstance(raw_vector, list) or len(raw_vector) != self.dimension:
                raise RuntimeError("DashScope returned an unexpected embedding dimension")
            if index in indexed or not 0 <= index < len(texts):
                raise RuntimeError("DashScope returned an invalid embedding index")
            indexed[index] = [float(value) for value in raw_vector]

        if len(indexed) != len(texts):
            raise RuntimeError("DashScope returned incomplete embedding indices")
        return [indexed[index] for index in range(len(texts))]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        inputs = [str(text or "") for text in texts]
        vectors: list[list[float]] = []
        for offset in range(0, len(inputs), _MAX_BATCH_SIZE):
            vectors.extend(self._request_batch(inputs[offset : offset + _MAX_BATCH_SIZE]))
        return vectors


def create_backend(*, dimension: int = 1024) -> DashScopeEmbeddingBackend:
    """Factory used by ``AML_DENSE_BACKEND=module:factory`` configuration."""

    return DashScopeEmbeddingBackend(dimension=dimension)


__all__ = ["DashScopeEmbeddingBackend", "create_backend"]
