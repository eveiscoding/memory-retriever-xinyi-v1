"""Membership-preserving rerank; scores are bi-encoder cosine, not cross-encoder.

Freeze membership AFTER baseline dedup and message slots. Keep baseline order
through the configured boundary; rerank only its tail. With default boundary=20,
all Top5/20 evidence and ranks are unchanged, while Top100 membership is retained.
"""
from __future__ import annotations

import math

from .features import extract_protection_entities

EXACT_FLAGS = {"exact_substring", "phrase", "entity", "date", "number"}


def _diverse_anchors(records, positions, quota, entities=None):
    """Spend escape slots on disjoint provenance, not repeated views of it.

    Provenance overlap and identical normalized content are conservative duplicate
    signals; different messages are not assumed to represent the same event.
    """
    anchors, sources, contents = [], set(), set()
    for i in positions:
        record = records[i]
        if len(anchors) >= quota:
            break
        flags = set(record['flags'])
        if entities is not None and 'entity' in flags and not any(
            entity in (record.get('content') or '').lower() for entity in entities
        ):
            flags.remove('entity')
        if not EXACT_FLAGS.intersection(flags):
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
    cuts = {0, min(5, stop), min(boundary, stop), stop}
    # Semantic ranking of the front bucket failed public quality gates. Keep
    # its complete order for every query; no dataset/intent routing is needed.
    entities = ([entity.lower() for entity in extract_protection_entities(query)]
                if query else None)
    cuts = sorted(cuts)
    for start, end in zip(cuts, cuts[1:]):
        positions = list(range(start, end))
        if end <= min(boundary, stop):
            continue
        # Diverse exact matches retain their original positions; no candidate
        # is injected or removed to implement this escape quota.
        anchors = _diverse_anchors(records, positions, exact_quota, entities)
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
