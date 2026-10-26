"""Isolated synthetic ONNX concurrency proof. No database, HTTP or external API."""
import argparse
import concurrent.futures
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aml_retriever.pair_rerank import pair_order


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def orders(scores):
    records = [dict(id=i, score=1 / (60 + i + 1), channel_ranks={}, flags=[])
               for i in range(len(scores))]
    fused = pair_order(copy.deepcopy(records), scores, pool=len(scores),
                       mode='rrf', weight=.25, k=60)
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i)), [r['id'] for r in fused]


def compare(reference, candidate):
    if len(reference) != len(candidate) or not reference:
        raise ValueError('score count mismatch')
    if any(not math.isfinite(v) for v in reference + candidate):
        raise ValueError('nonfinite scores')
    a, b = orders(reference), orders(candidate)
    return dict(exact=reference == candidate,
                max_abs_diff=max(abs(x-y) for x, y in zip(reference, candidate)),
                allclose_1e6=all(abs(x-y) <= 1e-6 + 1e-6*abs(x)
                                for x, y in zip(reference, candidate)),
                model_order_equal=a[0] == b[0], rrf_order_equal=a[1] == b[1],
                model_top20_equal=a[0][:20] == b[0][:20],
                rrf_top20_equal=a[1][:20] == b[1][:20])


def corpus():
    topics = [('gardening', 'tomatoes balcony summer watering'),
              ('travel', 'train museum hotel autumn tickets'),
              ('work', 'project meeting deadline engineer office'),
              ('food', 'vegetarian dinner pasta recipe vegetables')]
    result = []
    for q, (topic, terms) in enumerate(topics):
        query = 'What did the person mention about ' + topic + '?'
        docs = []
        for i in range(100):
            other = topics[(q+i//25) % 4][1]
            repetitions = [8, 20, 45, 120][i % 4]
            docs.append(f'Synthetic event {i}: ' + (other + ' ') * repetitions
                        + f' Detail {terms.split()[i % 5]}.')
        result.append((query, docs))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--revision', required=True)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(exist_ok=False, parents=True)
    def save(name, value):
        (out/name).write_text(json.dumps(value, indent=2, allow_nan=False))
    os.environ.update(AML_PAIR_MODEL_PATH=args.model,
        AML_PAIR_ONNX_FILE='onnx/model_quint8_avx2.onnx', AML_PAIR_MAX_LENGTH='512',
        AML_PAIR_BATCH_SIZE='1', TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    from aml_retriever.onnx_pair_backend import OnnxPairBackend
    profiles = [('serial', 1, 1), ('serial_repeat', 1, 1),
                ('intra2', 1, 2), ('parallel2', 2, 1), ('parallel4', 4, 1)]
    cases = corpus()
    save('prelock.json', dict(revision=args.revision,
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256=digest(cases), synthetic_only=True, database=False,
        profiles=profiles, queries=4, documents_per_query=100, batch=1, context=512,
        strategy_weight=.75, model_weight=.25, rrf_k=60,
        image=os.environ.get('AML_TEST_IMAGE_ID'), python=sys.version,
        packages={n: importlib.metadata.version(n) for n in
                  ['onnxruntime', 'transformers', 'numpy', 'tokenizers']},
        scope='model backend only; excludes service lock, retrieval, HTTP and official quality'))
    reference = None
    summaries = []
    for name, workers, threads in profiles:
        os.environ['AML_PAIR_THREADS'] = str(threads)
        backends = [OnnxPairBackend() for _ in range(workers)]
        hashes = backends[0].artifact_hashes
        assert all(b.artifact_hashes == hashes for b in backends)
        if name == 'serial':
            save('model-manifest.json', hashes)
        else:
            assert hashes == json.loads((out/'model-manifest.json').read_text())
        # Every worker owns its tokenizer/session. No shared mutable tokenizer.
        for b in backends:
            b.score(cases[0][0], [cases[0][1][-1]])
        results = [None] * len(cases)
        tokens = []
        # Token lengths are measured outside timed regions.
        for q, docs in cases:
            tokenized = backends[0].tokenizer([q]*100, docs, padding=False,
                truncation='only_second', max_length=512)
            tokens.extend(map(len, tokenized['input_ids']))
        assert max(tokens) == 512
        def run_worker(w):
            for i in range(w, len(cases), workers):
                mem = Path('/proc/meminfo').read_text().split('MemAvailable:')[1].split()[0]
                if int(mem) < 1024*1024:
                    raise RuntimeError('resource guard')
                started = time.perf_counter()
                scores = backends[w].score(*cases[i])
                assert len(scores) == 100 and all(math.isfinite(x) for x in scores)
                results[i] = dict(seconds=time.perf_counter()-started, scores=scores)
        cpu_start = time.process_time()
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run_worker, range(workers)))
        elapsed = time.perf_counter()-started
        cpu_seconds = time.process_time()-cpu_start
        if reference is None:
            reference = [r['scores'] for r in results]
        comparisons = [compare(a, b['scores']) for a,b in zip(reference,results)]
        row = dict(profile=name, workers=workers, threads_per_session=threads,
            wall_seconds=elapsed, cpu_seconds=cpu_seconds,
            average_cpu_cores=cpu_seconds/elapsed, queries_per_second=4/elapsed,
            min_tokens=min(tokens), max_tokens=max(tokens),
            cases=[dict(case=i, seconds=r['seconds'], score_sha256=digest(r['scores']),
                        comparison=comparisons[i]) for i,r in enumerate(results)],
            errors=0, scored_pairs=400, fallback=0)
        for f in ['memory.peak', 'memory.events', 'cpu.stat']:
            path = Path('/sys/fs/cgroup')/f
            row[f] = path.read_text() if path.exists() else None
        save(name+'.json', row)
        summaries.append(row)
        save('summary.json', dict(complete=False, profiles=summaries))
        print(json.dumps(dict(profile=name, wall_seconds=elapsed,
            all_exact=all(x['exact'] for x in comparisons),
            all_rrf_equal=all(x['rrf_order_equal'] for x in comparisons))), flush=True)
        # Release sessions before the next profile; allocator peak is cumulative.
        del backends
        import gc
        gc.collect()
    save('summary.json', dict(complete=True, profiles=summaries,
        all_rrf_equal=all(c['comparison']['rrf_order_equal']
                          for r in summaries for c in r['cases'])))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps({'failed_type':type(exc).__name__}), flush=True)
        raise SystemExit(1)
