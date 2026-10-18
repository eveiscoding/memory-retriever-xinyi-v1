"""Experimental pair relevance scoring. No query-type, entity or gold routing."""
from __future__ import annotations

import importlib
import math


def load_pair_backend(spec):
    if not spec:
        return None
    module, separator, factory = spec.partition(":")
    if not separator or not module or not factory:
        raise ValueError("pair backend must use module:factory")
    backend = getattr(importlib.import_module(module), factory)()
    if not callable(getattr(backend, "score", None)) or not getattr(backend, "identity", ""):
        raise ValueError("pair backend requires score(query, documents) and identity")
    return backend


def pair_order(records, scores, *, pool):
    """Reorder a fixed returned pool; malformed scores leave all records untouched.

    No Top5/20 membership or rank guarantee: these are empirical quality gates.
    Baseline scores remain diagnostic; consumers must respect returned order.
    """
    count = min(pool, len(records))
    scores = list(scores)
    if len(scores) != count or any(isinstance(s, bool) for s in scores):
        raise ValueError("invalid pair score count/type")
    scores = [float(score) for score in scores]
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("non-finite pair score")
    order = sorted(range(count), key=lambda i: (-scores[i], i))
    output = [records[i] for i in order] + records[count:]
    for rank, source in enumerate(order, 1):
        record = records[source]
        record["channel_ranks"]["pre_rerank"] = source + 1
        record["channel_ranks"]["post_rerank"] = rank
        record["semantic_score"] = scores[source]
        record["flags"].append("pair_rerank")
    return output
