#!/usr/bin/env python3
"""Search-only diagnostic replay of the existing Singapore smoke24 indexes.

Uses SQLite backups, never mutates source indexes, performs no Add, and cannot
serve as a fresh-index HTTP release gate. Requires already indexed matching
text-embedding-v4 vectors. No memory/query content is written to the report.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import statistics
import sys
import tempfile
import time

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', required=True, type=Path)
    parser.add_argument('--benchmark', required=True, type=Path)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--index-root', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.benchmark))
    from benchmark.adapters import load_longmemeval
    from benchmark import metrics as M
    from benchmark.provenance import file_hash, git_state
    core_sha, core_dirty = git_state(args.core)
    bench_sha, bench_dirty = git_state(args.benchmark)
    if core_dirty or bench_dirty:
        raise RuntimeError('replay requires clean committed checkouts')
    if args.out.exists():
        raise RuntimeError('refusing to overwrite report')
    sys.path.insert(0, str(args.core))
    from aml_retriever.config import RetrieverConfig
    from aml_retriever.retriever import RetrieverDB
    from aml_retriever.dashscope_embedding import create_backend

    cases = load_longmemeval(args.data, per_capability=4)
    assert len(cases) == 24
    capabilities = sorted({c.probes[0].capability for c in cases})
    backend = create_backend(dimension=1024)
    rows = []
    source_checks = []
    with tempfile.TemporaryDirectory(prefix='protected-replay-') as directory:
        for shard, capability in enumerate(capabilities):
            source = args.index_root / f'index-{shard}/dense.db'
            target = Path(directory) / f'{shard}.db'
            with sqlite3.connect(source.resolve().as_uri() + '?mode=ro', uri=True) as src:
                with sqlite3.connect(target) as dest:
                    src.backup(dest)
            source_checks.append({'shard': shard, 'source': str(source),
                                  'snapshot_sha256': file_hash(target)})
            baseline_db = RetrieverDB(RetrieverConfig(db_path=str(target)))
            # Disable eager index synchronization: this diagnostic may use only
            # vectors already present in the snapshot. Missing vectors fail the
            # replay rather than silently buying a new indexing run.
            candidate_db = RetrieverDB(RetrieverConfig(db_path=str(target)),
                                       embedding_backend=backend)
            candidate_db.semantic_enabled = True
            try:
                for case in (c for c in cases if c.probes[0].capability == capability):
                    probe = case.probes[0]
                    user = f'local:emb-sg120-{shard}:{case.dataset}:{case.case_id}'
                    results, times = {}, {}
                    for arm, db in (('baseline', baseline_db), ('protected', candidate_db)):
                        start = time.perf_counter()
                        evidence = db.search(user_id=user, query=probe.query,
                                             options=list(probe.options or ()), top_k=100)
                        times[arm] = (time.perf_counter() - start) * 1000
                        results[arm] = [asdict(r) for r in evidence.results]
                        if not evidence.results or any('semantic_rerank_fallback' in r.evidence_flags
                                                       for r in evidence.results):
                            raise RuntimeError(f'empty/fallback result for {probe.probe_id}')
                    for cutoff in (20, 100):
                        assert {r['id'] for r in results['baseline'][:cutoff]} == {
                            r['id'] for r in results['protected'][:cutoff]}
                    row = {'probe_id': probe.probe_id, 'capability': capability, 'arms': {}}
                    for arm, items in results.items():
                        gold = probe.effective_gold
                        row['arms'][arm] = {
                            'mrr': M.reciprocal_rank(items, gold),
                            'recall@5': M.recall_at_k(items, gold, 5),
                            'recall@20': M.recall_at_k(items, gold, 20),
                            'recall@100': M.recall_at_k(items, gold, 100),
                            'ndcg@20': M.ndcg_at_k(items, gold, 20),
                            'latency_ms': times[arm],
                            'result_ids': [r['id'] for r in items],
                        }
                    rows.append(row)
                    print(json.dumps({'completed': len(rows), 'probe_id': probe.probe_id}), flush=True)
            finally:
                baseline_db.close()
                candidate_db.close()
    def aggregate(subset, arm):
        return {key: statistics.mean(row['arms'][arm][key] for row in subset)
                for key in ('mrr', 'recall@5', 'recall@20', 'recall@100', 'ndcg@20', 'latency_ms')}
    report = {
        'kind': 'search-only in-process diagnostic; not HTTP/fresh-index release gate',
        'core_revision': core_sha, 'benchmark_revision': bench_sha,
        'input_sha256': file_hash(args.data), 'script_sha256': file_hash(Path(__file__)),
        'suite_sha256': file_hash(args.benchmark / 'suites/longmemeval-smoke.json'),
        'model_hash': backend.model_hash, 'model_identity': backend.identity,
        'source_snapshots': source_checks, 'queries': len(rows),
        'fixed_boundaries': [20, 100], 'exact_quota_per_bucket': 3,
        'overall': {arm: aggregate(rows, arm) for arm in ('baseline', 'protected')},
        'by_capability': {cap: {arm: aggregate([r for r in rows if r['capability'] == cap], arm)
                               for arm in ('baseline', 'protected')} for cap in capabilities},
        'rows': rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'overall': report['overall'], 'by_capability': report['by_capability']}), flush=True)


if __name__ == '__main__':
    main()
