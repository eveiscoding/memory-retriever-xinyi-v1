#!/usr/bin/env python3
"""Fetch explicitly public benchmark artifacts; data remains git-ignored."""
from __future__ import annotations

import argparse
import pathlib
import urllib.request

SOURCES = {
    "locomo": (
        "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
        "locomo10.json",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(SOURCES))
    parser.add_argument("--out", default="benchmark_data")
    args = parser.parse_args()
    url, filename = SOURCES[args.dataset]
    destination = pathlib.Path(args.out) / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"already exists: {destination}")
        return
    request = urllib.request.Request(url, headers={"User-Agent": "aml-local-benchmark/1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        destination.write_bytes(response.read())
    print(f"downloaded {url} -> {destination}")


if __name__ == "__main__":
    main()
