"""Membership-preserving rerank; scores are bi-encoder cosine, not cross-encoder.

Freeze membership AFTER baseline dedup and message slots. Reorder separately on
each side of the configured boundary. This preserves source coverage at that
boundary and at the requested limit, not at every possible cutoff.
"""
from __future__ import annotations

import math

from .features import has_event_relation_intent

EXACT_FLAGS = {"exact_substring", "phrase", "entity", "date", "number"}


def _diverse_anchors(records, positions, quota):
    """Spend escape slots on disjoint provenance, not repeated views of it.

    Provenance overlap and identical normalized content are conservative duplicate
    signals; different messages are not assumed to represent the same event.
    """
    anchors, sources, contents = [], set(), set()
    for i in positions:
        record = records[i]
        if len(anchors) >= quota:
            break
        if not EXACT_FLAGS.intersection(record["flags"]):
            continue
        source_ids = set(record.get("source_ids") or [record["id"]])
        content = " ".join((record.get("content") or "").casefold().split())
        if sources.intersection(source_ids) or (content and content in contents):
            continue
        anchors.append(i)
        sources.update(source_ids)
        if content:
            contents.add(content)
    return anchors


def protected_order(records, scores, *, boundary=20, pool=100, exact_quota=3,
                    query=""):
    if len(scores) != len(records) or any(not math.isfinite(s) for s in scores):
        raise ValueError("semantic reranker returned invalid scores")
    stop = min(pool, len(records))
    output = list(records)
    cuts = {0, min(boundary, stop), stop}
    # Temporal relations need multiple events. Preserve every baseline Top5
    # source, rather than guessing which event is gold or favoring recent ones.
    if has_event_relation_intent(query):
        cuts.add(min(5, stop))
    cuts = sorted(cuts)
    for start, end in zip(cuts, cuts[1:]):
        positions = list(range(start, end))
        # Diverse exact matches retain their original positions; no candidate
        # is injected or removed to implement this escape quota.
        anchors = _diverse_anchors(records, positions, exact_quota)
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
