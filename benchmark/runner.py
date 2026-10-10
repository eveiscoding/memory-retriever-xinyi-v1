"""Execute dataset-neutral cases against real AML HTTP endpoints."""
from __future__ import annotations

import math
import time
import uuid

from .chunking import chunk_messages
from . import metrics as M
from .schema import BenchmarkCase


def _rounded(value: float | None) -> float | None:
    return round(value, 4) if value is not None and not math.isnan(value) else None


def run_cases(cases: list[BenchmarkCase], client, *, top_k: int = 100,
              max_probes: int | None = None, run_id: str | None = None) -> dict:
    if not 1 <= top_k <= 100:
        raise ValueError("top_k must be between 1 and 100")
    run_id = run_id or uuid.uuid4().hex[:12]
    rows: list[dict] = []
    add_latencies: list[float] = []
    search_latencies: list[float] = []
    add_calls = 0
    messages = 0
    ingested_users: list[tuple[str, str]] = []

    for case in cases:
        case.validate()
        user_id = f"local:{run_id}:{case.dataset}:{case.case_id}"
        probes_by_checkpoint: dict[int, list] = {}
        for probe in case.probes:
            checkpoint = probe.after_session if probe.after_session is not None else len(case.sessions) - 1
            probes_by_checkpoint.setdefault(checkpoint, []).append(probe)

        for session_index, session in enumerate(case.sessions):
            for chunk_index, chunk in enumerate(chunk_messages(session.messages)):
                payload = {
                    "request_id": f"{user_id}:{session.session_id}:chunk-{chunk_index}",
                    "user_id": user_id,
                    "session_id": f"{user_id}:{session.session_id}",
                    "messages": [message.as_api_message() for message in chunk],
                }
                started = time.perf_counter()
                client.add(payload)
                add_latencies.append((time.perf_counter() - started) * 1000)
                add_calls += 1
                messages += len(chunk)

            for probe in probes_by_checkpoint.get(session_index, []):
                if max_probes is not None and len(rows) >= max_probes:
                    break
                payload = {"query": probe.query, "user_id": user_id, "top_k": top_k}
                if probe.options:
                    payload["options"] = list(probe.options)
                started = time.perf_counter()
                results = client.search(payload)
                elapsed = (time.perf_counter() - started) * 1000
                search_latencies.append(elapsed)
                gold = probe.effective_gold
                rows.append({
                    "case_id": case.case_id,
                    "probe_id": probe.probe_id,
                    "dataset": case.dataset,
                    "capability": probe.capability,
                    "gold_count": len(gold),
                    "returned": len(results),
                    "recall@5": M.recall_at_k(results, gold, 5),
                    "recall@20": M.recall_at_k(results, gold, 20),
                    "recall@100": M.recall_at_k(results, gold, 100),
                    "mrr": M.reciprocal_rank(results, gold),
                    "ndcg@20": M.ndcg_at_k(results, gold, 20),
                    "complete@20": float(
                        len(M.covered_gold(results, gold, 20)) == len(gold)
                    ) if gold else math.nan,
                    "forbidden@10": M.forbidden_at_k(results, probe.forbidden_evidence, 10),
                    "latency_ms": elapsed,
                    "result_ids": [str(item["id"]) for item in results],
                })
            if max_probes is not None and len(rows) >= max_probes:
                break
        longest_message = max(
            (message.content for session in case.sessions for message in session.messages),
            key=len,
        )
        ingested_users.append((user_id, longest_message))
        if max_probes is not None and len(rows) >= max_probes:
            break

    isolation_checks = 0
    isolation_breaches = 0
    if len(ingested_users) > 1:
        for index, (user_id, _) in enumerate(ingested_users):
            foreign_marker = ingested_users[(index + 1) % len(ingested_users)][1]
            results = client.search({"query": foreign_marker, "user_id": user_id, "top_k": top_k})
            isolation_checks += 1
            if M.covered_gold(results, (foreign_marker,), top_k):
                isolation_breaches += 1

    scored = [row for row in rows if row["gold_count"]]

    def aggregate(items: list[dict]) -> dict:
        return {
            "queries": len(items),
            "recall@5": _rounded(M.mean([row["recall@5"] for row in items])),
            "recall@20": _rounded(M.mean([row["recall@20"] for row in items])),
            "recall@100": _rounded(M.mean([row["recall@100"] for row in items])),
            "mrr": _rounded(M.mean([row["mrr"] for row in items])),
            "ndcg@20": _rounded(M.mean([row["ndcg@20"] for row in items])),
            "complete@20": _rounded(M.mean([row["complete@20"] for row in items])),
            "forbidden@10": _rounded(M.mean([row["forbidden@10"] for row in items])),
        }

    capabilities = sorted({row["capability"] for row in scored})
    return {
        "run_id": run_id,
        "top_k": top_k,
        "index": {"add_calls": add_calls, "messages": messages},
        "isolation": {"checks": isolation_checks, "breaches": isolation_breaches},
        "overall": aggregate(scored),
        "by_capability": {
            capability: aggregate([row for row in scored if row["capability"] == capability])
            for capability in capabilities
        },
        "latency": {
            "add_p50_ms": _rounded(M.percentile(add_latencies, 50)),
            "add_p95_ms": _rounded(M.percentile(add_latencies, 95)),
            "search_p50_ms": _rounded(M.percentile(search_latencies, 50)),
            "search_p95_ms": _rounded(M.percentile(search_latencies, 95)),
        },
        "rows": rows,
    }
