"""Experimental pair relevance scoring. No query-type, entity or gold routing."""
from __future__ import annotations

import importlib
import math

from .features import extract_protection_entities
from .protected_rerank import _diverse_anchors


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


def validate_pair_policy(mode, weight, k, *, protect_top20=False, protect_exact=False):
    if not isinstance(protect_top20, bool) or not isinstance(protect_exact, bool):
        raise ValueError("pair protection switches must be boolean")
    if mode != "model" and (protect_top20 or protect_exact):
        raise ValueError("pair protection ablations require pure model ordering")
    if mode not in {"model", "rrf", "cc", "cc3"}:
        raise ValueError("pair mode must be model, rrf, cc, or cc3")
    if (isinstance(weight, bool) or not math.isfinite(weight) or not 0 <= weight <= 1
            or isinstance(k, bool) or not math.isfinite(k) or k <= 0):
        raise ValueError("pair fusion requires finite weight in [0,1] and positive k")


def pair_order(records, scores, *, pool, mode="model", weight=0.25, k=60.0,
               protect_top20=False, protect_exact=False, query=""):
    """Reorder a fixed returned pool; malformed scores leave all records untouched.

    Protection switches are independent, off by default and only valid in model
    mode. Top20 protection freezes membership, not within-bucket ranks. Exact
    protection pins up to 3 diverse baseline slots across the whole scored pool.
    Neither guarantees correctness, Top5 coverage, or a retrieval improvement.
    Baseline scores remain diagnostic; consumers must respect returned order.
    """
    validate_pair_policy(mode, weight, k, protect_top20=protect_top20,
                         protect_exact=protect_exact)
    count = min(pool, len(records))
    scores = list(scores)
    if len(scores) != count or any(isinstance(s, bool) for s in scores):
        raise ValueError("invalid pair score count/type")
    scores = [float(score) for score in scores]
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("non-finite pair score")
    order = sorted(range(count), key=lambda i: (-scores[i], i))
    model_rank = {source: rank for rank, source in enumerate(order, 1)}
    diagnostics = None
    if mode == "rrf":
        # Rank fusion avoids treating RRF baseline scores and logits as a common
        # numeric scale. All queries use the same fixed policy; no intent routing.
        fused = [(1 - weight) / (k + i + 1) + weight / (k + model_rank[i])
                 for i in range(count)]
        order = sorted(range(count), key=lambda i: (-fused[i], i))
    elif mode == "cc":
        # Preserve within-query score gaps without treating either channel as a
        # calibrated probability. ``score`` is the baseline pipeline's final
        # RRF score, not an original feature score. A constant channel carries
        # no discrimination and therefore normalizes to zero for every item.
        baseline = []
        for record in records[:count]:
            value = record.get("score")
            if isinstance(value, bool):
                raise ValueError("invalid baseline score type")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("non-finite baseline score")
            baseline.append(value)

        def minmax(values):
            if not values:
                return []
            low, high = min(values), max(values)
            if high == low:
                return [0.0] * len(values)
            return [(value - low) / (high - low) for value in values]

        baseline_norm = minmax(baseline)
        model_norm = minmax(scores)
        fused = [(1 - weight) * baseline_norm[i] + weight * model_norm[i]
                 for i in range(count)]
        order = sorted(range(count), key=lambda i: (-fused[i], i))
    elif mode == "cc3":
        # These are pre-RRF component scores, not the already-fused baseline.
        channels = {"bm25": [], "rule": [], "model": scores}
        for record in records[:count]:
            for name in ("bm25", "rule"):
                value = record[f"cc3_{name}_score"]
                if isinstance(value, bool):
                    raise ValueError(f"invalid {name} score type")
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError(f"non-finite {name} score")
                channels[name].append(value)

        def normalize(values):
            if not values or max(values) == min(values):
                return [0.0] * len(values)
            # Scaling first also avoids overflow for extreme finite inputs.
            scale = max(abs(value) for value in values)
            scaled = [value / scale for value in values]
            low, high = min(scaled), max(scaled)
            return [(value - low) / (high - low) for value in scaled]

        normalized = {name: normalize(values) for name, values in channels.items()}
        fused = [sum(normalized[name][i] for name in channels) / 3.0
                 for i in range(count)]
        order = sorted(range(count), key=lambda i: (-fused[i], i))
        diagnostics = [dict(
            **{f"{name}_raw": values[i] for name, values in channels.items()},
            **{name: values[i] for name, values in normalized.items()},
            fused=fused[i],
        ) for i in range(count)]
    anchors = set()
    if protect_top20 or protect_exact:
        if protect_exact:
            entities = ([entity.lower() for entity in extract_protection_entities(query)]
                        if query else None)
            anchors = set(_diverse_anchors(records, range(count), 3, entities))
        cuts = [0, min(20, count), count] if protect_top20 else [0, count]
        order = list(range(count))
        for start, end in zip(cuts, cuts[1:]):
            movable = [i for i in range(start, end) if i not in anchors]
            ranked = sorted(movable, key=lambda i: (-scores[i], i))
            for target, source in zip(movable, ranked):
                order[target] = source
    output = [records[i] for i in order] + records[count:]
    for rank, source in enumerate(order, 1):
        record = records[source]
        record["channel_ranks"]["pre_rerank"] = source + 1
        record["channel_ranks"]["post_rerank"] = rank
        record["channel_ranks"]["pair_model"] = model_rank[source]
        record["semantic_score"] = scores[source]
        record["flags"].append("pair_rerank")
        if protect_top20:
            record["flags"].append("pair_top20_protect")
        if protect_exact:
            record["flags"].append("pair_exact_protect")
        if source in anchors:
            record["flags"].append("pair_exact_anchor")
        if mode == "rrf":
            record["flags"].append("pair_rrf")
        elif mode == "cc":
            record["flags"].append("pair_cc")
        elif mode == "cc3":
            record["flags"].append("pair_cc3")
            record["fusion_scores"] = diagnostics[source]
    return output
