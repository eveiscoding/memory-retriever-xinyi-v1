"""Evaluation-only read-only public corpus vector cache.

Set AML_DENSE_BACKEND=scripts.public_vector_cache:create_backend and
AML_PUBLIC_VECTOR_CACHE to an explicitly prepared SQLite file. Schema:
vectors(content_hash TEXT, model_hash TEXT, dimension INTEGER, vector BLOB),
indexed on (content_hash, model_hash, dimension). The operator must verify public
source provenance; this adapter never discovers other databases or writes data.
Missing content uses the same real model. Cached Add timing is not a cold
provider/indexing performance measurement. Production defaults are unaffected.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import sqlite3
import struct


class PublicVectorCache:
    def __init__(self, backend, path):
        self.backend = backend
        self.path = Path(path).resolve(strict=True)
        self.identity = backend.identity
        self.model_hash = backend.model_hash
        self.dimension = backend.dimension
        self.hits = 0
        self.misses = 0

    def embed(self, texts):
        output = [None] * len(texts)
        missing = []
        con = sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True)
        try:
            for i, text in enumerate(texts):
                digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
                row = con.execute(
                    'SELECT vector FROM vectors WHERE content_hash=? '
                    'AND model_hash=? AND dimension=?',
                    (digest, self.model_hash, self.dimension),
                ).fetchone()
                if row is None:
                    missing.append(i)
                    continue
                blob = row[0]
                if len(blob) != self.dimension * 4:
                    raise ValueError('invalid cached vector length')
                vector = struct.unpack(f'<{self.dimension}f', blob)
                if not all(math.isfinite(value) for value in vector):
                    raise ValueError('non-finite cached vector')
                output[i] = vector
                self.hits += 1
        finally:
            con.close()
        if missing:
            vectors = self.backend.embed([texts[i] for i in missing])
            if len(vectors) != len(missing):
                raise ValueError('invalid provider vector count')
            for i, vector in zip(missing, vectors):
                output[i] = vector
            self.misses += len(missing)
        return output


def create_backend(*, dimension=1024):
    from aml_retriever.dashscope_embedding import create_backend as real_backend
    path = os.environ.get('AML_PUBLIC_VECTOR_CACHE', '')
    if not path:
        raise ValueError('AML_PUBLIC_VECTOR_CACHE is required')
    return PublicVectorCache(real_backend(dimension=dimension), path)
