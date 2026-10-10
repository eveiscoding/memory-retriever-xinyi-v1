"""Evidence-text retrieval metrics usable across black-box memory APIs."""
from __future__ import annotations

import math
import re


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def covered_gold(results: list[dict], gold: tuple[str, ...], k: int) -> set[int]:
    normalized_gold = [normalize(item) for item in gold]
    covered: set[int] = set()
    for result in results[:k]:
        content = normalize(str(result.get("content", "")))
        for index, marker in enumerate(normalized_gold):
            if marker and marker in content:
                covered.add(index)
    return covered


def recall_at_k(results: list[dict], gold: tuple[str, ...], k: int) -> float:
    return len(covered_gold(results, gold, k)) / len(gold) if gold else math.nan


def reciprocal_rank(results: list[dict], gold: tuple[str, ...]) -> float:
    if not gold:
        return math.nan
    for rank in range(1, len(results) + 1):
        if covered_gold(results[rank - 1:rank], gold, 1):
            return 1.0 / rank
    return 0.0


def forbidden_at_k(results: list[dict], forbidden: tuple[str, ...], k: int) -> float:
    return len(covered_gold(results, forbidden, k)) / len(forbidden) if forbidden else math.nan


def ndcg_at_k(results: list[dict], gold: tuple[str, ...], k: int) -> float:
    if not gold:
        return math.nan
    # One aggregate memory may contain several gold messages. Count only newly
    # covered markers at each rank and normalize against the best possible case:
    # one rank-1 aggregate containing every required evidence item.
    seen: set[int] = set()
    dcg = 0.0
    for rank, result in enumerate(results[:k], start=1):
        newly_covered = covered_gold([result], gold, 1) - seen
        dcg += len(newly_covered) / math.log2(rank + 1)
        seen.update(newly_covered)
    return dcg / len(gold)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def mean(values: list[float]) -> float | None:
    clean = [value for value in values if not math.isnan(value)]
    return sum(clean) / len(clean) if clean else None
