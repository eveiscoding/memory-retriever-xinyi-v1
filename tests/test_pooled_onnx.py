import concurrent.futures
import threading
import unittest
from unittest.mock import patch

from aml_retriever.config import RetrieverConfig
from aml_retriever.onnx_pair_backend import PooledOnnxPairBackend
from aml_retriever.retriever import RetrieverDB


class FakeWorker:
    identity = 'synthetic'
    model_hash = 'fixture'
    artifact_hashes = {'fixture': 'same'}
    def score(self, query, documents):
        if query == 'fail':
            raise ValueError('fixture')
        return list(range(len(documents)))


class PooledTests(unittest.TestCase):
    def make_pool(self, factory=FakeWorker):
        with patch('aml_retriever.onnx_pair_backend.OnnxPairBackend', factory):
            return PooledOnnxPairBackend()

    def test_worker_returned_on_failure(self):
        pool = self.make_pool()
        for _ in range(5):
            with self.assertRaises(ValueError):
                pool.score('fail', ['x'])
        self.assertEqual(pool._workers.qsize(), 2)
        self.assertEqual(pool.score('ok', ['x', 'y']), [0, 1])

    def test_artifact_mismatch_fails_at_startup(self):
        other = FakeWorker()
        other.identity = 'different'
        with patch('aml_retriever.onnx_pair_backend.OnnxPairBackend',
                   side_effect=[FakeWorker(), other]):
            with self.assertRaises(ValueError):
                PooledOnnxPairBackend()

    def test_retriever_allows_two_and_never_more(self):
        lock = threading.Lock()
        entered_two = threading.Event()
        release = threading.Event()
        active = peak = calls = 0
        class Blocking(FakeWorker):
            def score(self, query, documents):
                nonlocal active, peak, calls
                with lock:
                    active += 1
                    calls += 1
                    peak = max(peak, active)
                    if active == 2:
                        entered_two.set()
                try:
                    if not release.wait(5):
                        raise RuntimeError('fixture wait expired')
                    return [1.] * len(documents)
                finally:
                    with lock:
                        active -= 1
        pool = self.make_pool(Blocking)
        db = RetrieverDB(RetrieverConfig().with_flags(pair_rerank=True), pair_backend=pool)
        def run(i):
            rows = [dict(id=str(i), content='fixture', flags=[], channel_ranks={})]
            return db._pair_rerank('query', rows)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                futures = [executor.submit(run, i) for i in range(6)]
                try:
                    self.assertTrue(entered_two.wait(3), 'outer lock still serializes pool')
                    with lock:
                        self.assertEqual(calls, 2)
                finally:
                    release.set()
                rows = [f.result(5) for f in futures]
            self.assertEqual(peak, 2)
            self.assertEqual(calls, 6)
            self.assertTrue(all('pair_rerank_fallback' not in r[0]['flags'] for r in rows))
            self.assertEqual(pool._workers.qsize(), 2)
        finally:
            release.set()
            db.close()

    def test_original_backend_remains_serial(self):
        db = RetrieverDB(RetrieverConfig().with_flags(pair_rerank=True), pair_backend=FakeWorker())
        try:
            with db._pair_lock:
                result = []
                def probe():
                    acquired = db._pair_lock.acquire(blocking=False)
                    result.append(acquired)
                    if acquired:
                        db._pair_lock.release()
                t = threading.Thread(target=probe)
                t.start()
                t.join(2)
                self.assertEqual(result, [False])
        finally:
            db.close()
