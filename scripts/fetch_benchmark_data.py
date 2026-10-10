#!/usr/bin/env python3
"""Fetch explicitly public benchmark artifacts; data remains git-ignored."""
from __future__ import annotations

import argparse
import pathlib
import shutil
import urllib.request

SOURCES = {
    "locomo": ((
        "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
        "locomo10.json",
    ),),
    "longmemeval-s": ((
        "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
        "longmemeval_s_cleaned.json",
    ),),
    "halumem-medium": ((
        "https://huggingface.co/datasets/IAAR-Shanghai/HaluMem/resolve/main/HaluMem-Medium.jsonl",
        "HaluMem-Medium.jsonl",
    ),),
    "halumem-long": ((
        "https://huggingface.co/datasets/IAAR-Shanghai/HaluMem/resolve/main/HaluMem-Long.jsonl",
        "HaluMem-Long.jsonl",
    ),),
    "locomo-refined": (
        (
            "https://raw.githubusercontent.com/mem-eval-suite/LoCoMo_refined/main/data/public/conversations.jsonl",
            "locomo_refined_conversations.jsonl",
        ),
        (
            "https://raw.githubusercontent.com/mem-eval-suite/LoCoMo_refined/main/data/public/questions.jsonl",
            "locomo_refined_questions.jsonl",
        ),
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(SOURCES))
    parser.add_argument("--out", default="benchmark_data")
    args = parser.parse_args()
    for url, filename in SOURCES[args.dataset]:
        destination = pathlib.Path(args.out) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            print(f"already exists: {destination}")
            continue
        request = urllib.request.Request(url, headers={"User-Agent": "aml-local-benchmark/1"})
        partial = destination.with_suffix(destination.suffix + ".part")
        with urllib.request.urlopen(request, timeout=300) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        partial.replace(destination)
        print(f"downloaded {url} -> {destination}")


if __name__ == "__main__":
    main()
