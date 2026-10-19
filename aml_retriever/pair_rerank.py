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


def validate_pair_policy(mode, weight, k):
    if mode not in {"model", "rrf"}:
        raise ValueError("pair mode must be model or rrf")
    if (isinstance(weight, bool) or not math.isfinite(weight) or not 0 <= weight <= 1
            or isinstance(k, bool) or not math.isfinite(k) or k <= 0):
        raise ValueError("pair RRF requires finite weight in [0,1] and positive k")


def pair_order(records, scores, *, pool, mode="model", weight=0.25, k=60.0):
    """Reorder a fixed returned pool; malformed scores leave all records untouched.

    No Top5/20 membership or rank guarantee: these are empirical quality gates.
    Baseline scores remain diagnostic; consumers must respect returned order.
    """
    validate_pair_policy(mode, weight, k)
    count = min(pool, len(records))
    scores = list(scores)
    if len(scores) != count or any(isinstance(s, bool) for s in scores):
        raise ValueError("invalid pair score count/type")
    scores = [float(score) for score in scores]
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("non-finite pair score")
    order = sorted(range(count), key=lambda i: (-scores[i], i))
    model_rank = {source: rank for rank, source in enumerate(order, 1)}
    if mode == "rrf":
        # Rank fusion avoids treating RRF baseline scores and logits as a common
        # numeric scale. All queries use the same fixed policy; no intent routing.
        fused = [(1 - weight) / (k + i + 1) + weight / (k + model_rank[i])
                 for i in range(count)]
        order = sorted(range(count), key=lambda i: (-fused[i], i))
    output = [records[i] for i in order] + records[count:]
    for rank, source in enumerate(order, 1):
        record = records[source]
        record["channel_ranks"]["pre_rerank"] = source + 1
        record["channel_ranks"]["post_rerank"] = rank
        record["channel_ranks"]["pair_model"] = model_rank[source]
        record["semantic_score"] = scores[source]
        record["flags"].append("pair_rerank")
        if mode == "rrf":
            record["flags"].append("pair_rrf")
    return output
