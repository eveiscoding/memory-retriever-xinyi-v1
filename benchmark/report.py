"""Small human-readable companion to the complete JSON artifact."""
from __future__ import annotations


def _value(value) -> str:
    return "n/a" if value is None else str(value)


def render_markdown(report: dict) -> str:
    overall = report["overall"]
    latency = report["latency"]
    isolation = report["isolation"]
    lines = [
        "# Retrieval benchmark report",
        "",
        f"- Dataset: `{report.get('dataset', 'unknown')}`",
        f"- Run ID: `{report['run_id']}`",
        f"- Search Top K: `{report['top_k']}`",
        f"- Scored queries: `{overall['queries']}`",
        f"- Add calls / messages: `{report['index']['add_calls']}` / `{report['index']['messages']}`",
        f"- Isolation breaches: `{isolation['breaches']}` / `{isolation['checks']}`",
        "",
        "## Overall retrieval",
        "",
        "| Recall@5 | Recall@20 | Recall@100 | MRR | nDCG@20 | Complete@20 | Forbidden@10 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| " + " | ".join(_value(overall[key]) for key in (
            "recall@5", "recall@20", "recall@100", "mrr", "ndcg@20",
            "complete@20", "forbidden@10",
        )) + " |",
        "",
        "## By capability",
        "",
        "| Capability | Questions | Recall@20 | Recall@100 | MRR | Complete@20 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for capability, metrics in report["by_capability"].items():
        lines.append(
            f"| {capability} | {metrics['queries']} | {_value(metrics['recall@20'])} | "
            f"{_value(metrics['recall@100'])} | {_value(metrics['mrr'])} | "
            f"{_value(metrics['complete@20'])} |"
        )
    lines.extend([
        "",
        "## HTTP latency",
        "",
        f"- Add p50 / p95: `{latency['add_p50_ms']}` / `{latency['add_p95_ms']}` ms",
        f"- Search p50 / p95: `{latency['search_p50_ms']}` / `{latency['search_p95_ms']}` ms",
        "",
        "> Retrieval-only proxy. It does not reproduce AML's hidden suite, locked Answer model, judge, or final score.",
        "",
    ])
    return "\n".join(lines)
