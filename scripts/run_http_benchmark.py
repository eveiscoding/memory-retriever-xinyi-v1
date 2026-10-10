#!/usr/bin/env python3
"""Run competition-shaped retrieval evaluation against an AML endpoint."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.adapters import load_locomo, load_longmemeval, load_synthetic
from benchmark.http_client import AMLClient
from benchmark.report import render_markdown
from benchmark.runner import run_cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--api-key-env", default="AML_API_KEY")
    parser.add_argument("--dataset", choices=("synthetic", "locomo", "longmemeval-s"),
                        default="synthetic")
    parser.add_argument("--data-path", default="benchmark_data/locomo10.json")
    parser.add_argument("--synthetic-scale", choices=("smoke", "small", "medium", "large"),
                        default="small")
    parser.add_argument("--synthetic-difficulty", choices=("plain", "paraphrase", "mixed"),
                        default="mixed")
    parser.add_argument("--longmemeval-per-capability", type=int)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--max-probes", type=int, default=200)
    parser.add_argument("--run-id")
    parser.add_argument("--out", default="benchmark_reports/latest.json")
    args = parser.parse_args()

    if args.dataset == "synthetic":
        cases = load_synthetic(scale=args.synthetic_scale, difficulty=args.synthetic_difficulty)
    elif args.dataset == "locomo":
        cases = load_locomo(args.data_path)
    else:
        cases = load_longmemeval(
            args.data_path,
            limit=None if args.longmemeval_per_capability else args.max_probes,
            per_capability=args.longmemeval_per_capability,
        )
    client = AMLClient(args.base_url, api_key=os.environ.get(args.api_key_env, ""))
    report = run_cases(cases, client, top_k=args.top_k, max_probes=args.max_probes,
                       run_id=args.run_id)
    report["dataset"] = args.dataset
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = destination.with_suffix(".md")
    markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({
        "dataset": args.dataset,
        "run_id": report["run_id"],
        "overall": report["overall"],
        "latency": report["latency"],
        "report": str(destination),
        "markdown": str(markdown),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
