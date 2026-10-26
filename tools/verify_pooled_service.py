"""Synthetic HTTP validation of indexed writes with real pooled ONNX scoring."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aml_retriever.api import MemoryService
from aml_retriever.config import RetrieverConfig
from aml_retriever.server import RetrieverServer


def main():
    output = Path(sys.argv[1])
    assert not output.exists()
    config = RetrieverConfig(db_path='/data/synthetic.db', host='127.0.0.1', port=0,
        retention_days=0, auth_mode='bearer', api_key='synthetic-only',
        pair_backend='aml_retriever.onnx_pair_backend:create_pooled_backend',
        pair_rerank_mode='rrf', pair_rrf_weight=.25, pair_rrf_k=60,
        pair_rerank_pool=100).with_flags(pair_rerank=True)
    service = MemoryService(config)
    server = RetrieverServer(config, service=service, quiet=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = 'http://127.0.0.1:' + str(server.server_address[1])
    def call(path, payload, auth=True):
        request = urllib.request.Request(base+path, json.dumps(payload).encode(),
            headers={'Content-Type':'application/json',
                     **({'Authorization':'Bearer synthetic-only'} if auth else {})})
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as exc:
            return exc.code, json.load(exc)
    try:
        payloads = [dict(user_id='pooled-test-'+str(u), session_id='session',
            request_id='seed-'+str(u), messages=[dict(role='user',
                timestamp=1704067200000+i*60000,
                content=f'Synthetic marker user{u} item{i} ' +
                    ('garden tomatoes balcony travel train museum ' * (5 if i%2 else 100)))
                for i in range(8)]) for u in range(2)]
        for payload in payloads:
            assert call('/add', payload)[0] == 200
            assert call('/add', payload)[0] == 200
        query = dict(user_id='pooled-test-0', query='garden tomatoes balcony', top_k=100)
        assert call('/search', query, auth=False)[0] == 401
        # Same candidate runtime and corpus; compare serial calls vs concurrency.
        reference = call('/search', query)
        assert reference[0] == 200 and reference[1]['data']
        signature = hashlib.sha256(json.dumps(reference[1]['data'],sort_keys=True).encode()).hexdigest()
        pool = service.db.pair_backend
        lock = threading.Lock()
        active = peak = measured_calls = 0
        def wrap(fn):
            def score(*args):
                nonlocal active, peak, measured_calls
                with lock:
                    active += 1
                    measured_calls += 1
                    peak = max(peak, active)
                try:
                    return fn(*args)
                finally:
                    with lock:
                        active -= 1
            return score
        for worker in list(pool._workers.queue):
            worker.score = wrap(worker.score)
        def search(i):
            started = time.perf_counter()
            status, body = call('/search', query)
            assert status == 200
            rows = body['data']
            assert rows and all('user1' not in r['content'] for r in rows)
            assert all('pair_rerank_fallback' not in r.get('evidence_flags',[]) for r in rows)
            assert hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest() == signature
            return time.perf_counter()-started
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(search,i) for i in range(8)]
            # Concurrent write to another user, while search workers run.
            assert call('/add',dict(user_id='write-test',session_id='s',request_id='concurrent',
                messages=[dict(role='user',content='Synthetic concurrent write')]))[0] == 200
            times = [f.result() for f in futures]
        assert peak == 2 and measured_calls == 8 and active == 0
        assert pool._workers.qsize() == 2
        assert call('/search', dict(user_id='absent',query='garden',top_k=100))[1]['data'] == []
        result = dict(passed=True, http_searches=8, peak_active_model_calls=peak,
            model_calls=measured_calls, total_seconds=time.perf_counter()-started,
            request_seconds=times, all_evidence_and_scores_equal=True,
            idempotent_add=True, unauthorized_401=True, isolation=True,
            concurrent_write=True, pool_returned=True, fallback=0,
            result_count=len(reference[1]['data']), reference_sha256=signature)
        output.write_text(json.dumps(result, indent=2))
        print(json.dumps(result),flush=True)
    finally:
        server.shutdown()
        server.server_close()
        service.db.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps({'failed_type':type(exc).__name__}),flush=True)
        raise SystemExit(1)
