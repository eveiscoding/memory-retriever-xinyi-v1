"""Server-only six-probe real-model regression using previously locked public data.

No production database or credential is used. Requires an existing model and
Benchmark checkout; outputs hashes/metrics, never query/evidence text.
"""
import argparse
import dataclasses as D
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess as S
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def select(raw):
    # Before any result is read: two shortest distinct cases, one probe each.
    eligible = [c for c in raw if any(p.get('gold_evidence') or p.get('gold_evidence_groups')
                                     for p in c['probes'])]
    chosen = sorted(eligible, key=lambda c: (sum(len(s['messages']) for s in c['sessions']), c['case_id']))[:2]
    assert len(chosen) == 2
    result = []
    for c in chosen:
        probe = sorted([p for p in c['probes'] if p.get('gold_evidence') or p.get('gold_evidence_groups')],
                       key=lambda p: p['probe_id'])[0]
        result.append(dict(c, probes=[probe]))
    return result


def version(core):
    assert not S.check_output(['git', '-C', str(core), 'status', '--porcelain']).strip()
    return S.check_output(['git', '-C', str(core), 'rev-parse', 'HEAD'], text=True).strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ['baseline-core', 'benchmark', 'source-run', 'model', 'python', 'output']:
        p.add_argument('--' + arg, required=True)
    a = p.parse_args()
    candidate = Path(__file__).resolve().parents[1]
    baseline, bench, source, model, out = map(Path, [a.baseline_core, a.benchmark, a.source_run, a.model, a.output])
    versions = {'baseline': version(baseline), 'candidate': version(candidate), 'benchmark': version(bench)}
    sys.path.insert(0, str(bench))
    from benchmark.schema import BenchmarkCase, BenchmarkSession, BenchmarkMessage, BenchmarkProbe
    from benchmark.runner import run_cases
    from benchmark.http_client import AMLClient
    from benchmark.report import render_markdown

    out.mkdir(exist_ok=False, parents=True)

    def save(name, value):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2))

    old_lock = json.loads((source / 'prelock.json').read_text())
    assert versions['benchmark'] == old_lock['benchmark_revision']
    for rel, digest in old_lock['model'].items():
        assert sha(model / rel) == digest, 'model changed'
    raw, inputs = [], {}
    for dataset in ['longmemeval-s', 'convomem-lite', 'locomo-refined']:
        file = source / (dataset + '-cases.json')
        inputs[dataset] = sha(file)
        assert inputs[dataset] == old_lock['inputs'][dataset]
        raw.extend(select(json.loads(file.read_text())))
    cases = []
    for item in raw:
        c = dict(item)
        c['sessions'] = tuple(BenchmarkSession(s['session_id'], tuple(BenchmarkMessage(**m) for m in s['messages']))
                              for s in c['sessions'])
        c['probes'] = tuple(BenchmarkProbe(**probe) for probe in c['probes'])
        case = BenchmarkCase(**c)
        case.validate()
        cases.append(case)
    lock = {'versions': versions, 'source_inputs': inputs, 'selected_input_sha256': fingerprint(raw),
            'case_ids': [c.case_id for c in cases], 'probe_ids': [c.probes[0].probe_id for c in cases],
            'development_regression_shortest_cases_not_fresh_holdout': True,
            'driver_sha256': sha(__file__), 'model': old_lock['model'],
            'parameters': {'top_k': 100, 'rrf_model_weight': .25, 'rrf_k': 60,
                           'pool': 100, 'context': 512, 'batch': 1, 'threads': 1,
                           'fixed_clock': '2026-10-01T00:00:00+00:00'}}
    save('prelock.json', lock)
    print('PRELOCK', sha(out / 'prelock.json'), 'probes', len(cases),
          'messages', sum(len(s.messages) for c in cases for s in c.sessions), flush=True)
    arms = {}
    runtime_trees = {}
    for name, core in [('baseline', baseline), ('candidate', candidate)]:
        runtime_trees[name] = {str(f.relative_to(core)): sha(f) for f in (core / 'aml_retriever').glob('*.py')}
        db = out / (name + '.db')
        assert not db.exists()
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(('AML_', 'DASHSCOPE_', 'OPENAI_', 'PAIR_SCORE_'))}
        env.update(PYTHONPATH=str(core), PYTHONDONTWRITEBYTECODE='1', AML_DB_PATH=str(db),
            AML_HOST='127.0.0.1', AML_PORT=str(port), AML_SYSTEM_REVISION=versions[name],
            AML_AUTH_MODE='none', AML_RETENTION_DAYS='0', AML_ADMIN_ENABLED='0',
            AML_FLAG_VECTOR='0', AML_FLAG_SEMANTIC_RERANK='0', AML_FLAG_PAIR_RERANK='1',
            AML_FLAG_PAIR_TOP20_PROTECT='0', AML_FLAG_PAIR_EXACT_PROTECT='0',
            AML_PAIR_RERANK_MODE='rrf', AML_PAIR_RRF_WEIGHT='.25', AML_PAIR_RRF_K='60',
            AML_PAIR_RERANK_POOL='100', AML_PAIR_BACKEND='aml_retriever.onnx_pair_backend:create_backend',
            AML_PAIR_MODEL_PATH=str(model), AML_PAIR_ONNX_FILE='onnx/model_quint8_avx2.onnx',
            AML_PAIR_MAX_LENGTH='512', AML_PAIR_BATCH_SIZE='1', AML_PAIR_THREADS='1',
            AML_INCLUDE_PROVENANCE='1', TOKENIZERS_PARALLELISM='false',
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        # Freeze only the retriever's clock. Do not replace Python's process-
        # wide time function (timeouts and model runtimes use their own clock).
        entry = ("import time,types; import aml_retriever.retriever as r; "
                 "r.time=types.SimpleNamespace(time=lambda:1790812800.,sleep=time.sleep); "
                 "r._now_iso=lambda:'2026-10-01T00:00:00+00:00'; "
                 "from aml_retriever.server import serve; serve(quiet=True)")
        proc = S.Popen([a.python, '-c', entry], env=env, cwd=core,
                       stdout=(out / (name + '-service.log')).open('w'), stderr=S.STDOUT)
        obs = []

        class Client(AMLClient):
            def search(self, payload):
                rows = super().search(payload)
                obs.append({'request_sha256': fingerprint(payload), 'full_results_sha256': fingerprint(rows),
                            'ordered_ids': [r['id'] for r in rows],
                            'returned': len(rows), 'scored': sum('semantic_score' in r for r in rows),
                            'fallback': any('pair_rerank_fallback' in r.get('evidence_flags', []) for r in rows)})
                print('SEARCH_DONE', name, len(obs), flush=True)
                return rows

        try:
            client = Client('http://127.0.0.1:' + str(port), timeout=180)
            for _ in range(60):
                assert proc.poll() is None, 'service exited'
                try:
                    client.health(versions[name])
                    break
                except Exception:
                    time.sleep(1)
            else:
                raise RuntimeError('health timeout')
            report = run_cases(cases, client, top_k=100, run_id='index-write-six-20261026')
            assert report['coverage']['scored_queries'] == 6
            assert report['isolation']['breaches'] == 0
            assert all(o['scored'] == o['returned'] and not o['fallback'] for o in obs[:6])
            save(name + '.json', report)
            save(name + '-observations.json', obs)
            (out / (name + '.md')).write_text(render_markdown(report))
            arms[name] = {'report': report, 'observations': obs}
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except S.TimeoutExpired:
                proc.kill()
                proc.wait()
            with socket.socket() as sock:
                assert sock.connect_ex(('127.0.0.1', port)) != 0
            save(name + '-closed.json', {'port': port, 'pid': proc.pid, 'returncode': proc.returncode,
                                         'port_closed': True})
    # Compare everything that affects evidence quality, not latency/index bytes.
    old, new = arms['baseline'], arms['candidate']
    assert old['observations'] == new['observations'], 'evidence/order/score changed'
    for key in ['overall', 'by_capability', 'coverage', 'isolation']:
        assert old['report'][key] == new['report'][key], key
    for left, right in zip(old['report']['rows'], new['report']['rows']):
        assert {k: v for k, v in left.items() if k != 'latency_ms'} == {
            k: v for k, v in right.items() if k != 'latency_ms'}, 'case metrics changed'
    summary = {'pass': True, 'versions': versions, 'lock_sha256': sha(out / 'prelock.json'),
               'runtime_changed_files': [k for k in runtime_trees['baseline']
                                         if runtime_trees['baseline'][k] != runtime_trees['candidate'][k]],
               'quality_queries_per_arm': 6, 'top100_exact_evidence_order_scores_equal': True,
               'overall': new['report']['overall'], 'by_capability': new['report']['by_capability'],
               'isolation': new['report']['isolation'], 'cases': new['report']['rows'],
               'model_pairs_per_arm': sum(o['scored'] for o in new['observations'][:6]),
               'reports_sha256': {n: sha(out / (n + '.json')) for n in arms}}
    save('summary.json', summary)
    print('PASS', sha(out / 'summary.json'), json.dumps(summary['overall']), flush=True)


if __name__ == '__main__':
    main()
