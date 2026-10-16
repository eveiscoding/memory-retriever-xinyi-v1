"""Membership-preserving rerank; scores are bi-encoder cosine, not cross-encoder.

Freeze membership AFTER baseline dedup and message slots. Reorder separately on
each side of the configured boundary. This preserves source coverage at that
boundary and at the requested limit, not at every possible cutoff.
"""
from __future__ import annotations

import math

EXACT_FLAGS = {"exact_substring", "phrase", "entity", "date", "number"}


def protected_order(records, scores, *, boundary=20, pool=100, exact_quota=3):
    if len(scores) != len(records) or any(not math.isfinite(s) for s in scores):
        raise ValueError("semantic reranker returned invalid scores")
    stop = min(pool, len(records))
    output = list(records)
    for start, end in ((0, min(boundary, stop)), (min(boundary, stop), stop)):
        positions = list(range(start, end))
        # The first exact matches retain their original positions; no candidate
        # is injected or removed to implement this escape quota.
        anchors = [i for i in positions if EXACT_FLAGS.intersection(records[i]["flags"])][
            :exact_quota
        ]
        movable = [i for i in positions if i not in anchors]
        ranked = sorted(movable, key=lambda i: (-scores[i], i))
        for target, source in zip(movable, ranked):
            output[target] = records[source]
    before = {record["id"]: rank for rank, record in enumerate(records, 1)}
    score_by_id = {r["id"]: score for r, score in zip(records, scores)}
    for rank, record in enumerate(output, 1):
        record["channel_ranks"]["pre_rerank"] = before[record["id"]]
        record["channel_ranks"]["post_rerank"] = rank
        record["semantic_score"] = score_by_id[record["id"]]
        record["flags"].append("semantic_rerank")
    return output
