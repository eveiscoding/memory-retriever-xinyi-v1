import copy
import queue
import unittest
from unittest.mock import patch
from aml_retriever.source_dedup import source_aware_order,validate_visible_spans
from aml_retriever.onnx_pair_backend import OnnxPairBackend,PooledOnnxPairBackend
from tests import test_source_dedup as fixtures
rec=fixtures.rec


class VisibleOrderTests(unittest.TestCase):
    def test_fact_after_visible_prefix_is_not_deferred(self):
        rows=[rec('long',['a'],'intro then hidden fact'),rec('short',['a'],'hidden fact'),rec('new',['b'],'other')]
        out,a=source_aware_order(rows,user_id='u',visible_spans={'long':(0,5),'short':(0,11),'new':(0,5)})
        self.assertEqual(out,rows);self.assertEqual(a['short']['status'],'retained')

    def test_visible_full_body_can_be_deferred(self):
        rows=[rec('long',['a'],'intro fact tail'),rec('short',['a'],'fact'),rec('new',['b'],'other')]
        out,a=source_aware_order(rows,user_id='u',visible_spans={'long':(0,10),'short':(0,4),'new':(0,5)})
        self.assertEqual([r['id'] for r in out],['long','new','short'])
        self.assertEqual(a['short']['witness_visible_span'],[0,10])

    def test_missing_visibility_is_not_full_visibility(self):
        rows=[rec('long',['a'],'intro fact'),rec('short',['a'],'fact')]
        before=copy.deepcopy(rows)
        for spans in (None,{},[],{'short':(0,4)}):
            self.assertEqual(source_aware_order(rows,user_id='u',visible_spans=spans)[0],rows)
        self.assertEqual(rows,before)

    def test_bad_witness_spans_cannot_authorize_deferral(self):
        rows=[rec('long',['a'],'intro fact'),rec('short',['a'],'fact')]
        for span in (None,(0,),(-1,4),(0,999),(True,4),(0,4.0),'all',{'start':0}):
            self.assertEqual(source_aware_order(rows,user_id='u',visible_spans={'long':span})[0],rows)

    def test_count_and_bounds_validation_is_atomic(self):
        rows=[rec('a',['x'],'alpha')]
        for spans in (None,[],[(0,6)],[(False,2)],[(0,1.0)]):
            with self.assertRaises(ValueError):validate_visible_spans(rows,spans)
        self.assertEqual(validate_visible_spans(rows,[(0,5)]),{'a':(0,5)})

    def test_prefix_visible_markers_are_preserved(self):
        rows=[rec('a',['x'],'fact one; hidden two'),rec('b',['x'],'hidden two'),rec('c',['x'],'fact one'),rec('d',['y'],'new fact')]
        spans={'a':(0,8),'b':(0,10),'c':(0,8),'d':(0,8)}
        out,_=source_aware_order(rows,user_id='u',visible_spans=spans)
        for k in range(1,len(rows)+1):
            for marker in ('fact one','hidden two','new fact'):
                old=any(marker in r['content'][slice(*spans[r['id']])] for r in rows[:k])
                new=any(marker in r['content'][slice(*spans[r['id']])] for r in out[:k])
                self.assertFalse(old and not new)


class Encoding(dict):
    def sequence_ids(self,index):return [None,0,None,1,1,None]


class Tokenizer:
    def __init__(self):self.calls=[];self.bad=False
    def encode(self,q,**kwargs):return q.split()
    def num_special_tokens_to_add(self,**kwargs):return 4
    def __call__(self,queries,docs,**kwargs):
        self.calls.append((queries,docs,kwargs));offsets=[]
        for d in docs:
            end=len(d)+1 if self.bad else min(6,len(d))
            offsets.append([(0,0),(0,999),(0,0),(0,min(3,len(d))),(min(3,len(d)),end),(0,0)])
        return Encoding(offset_mapping=offsets)


class BackendSpanTests(unittest.TestCase):
    def test_document_offsets_exclude_query_and_match_score_policy(self):
        backend=object.__new__(OnnxPairBackend);backend.tokenizer=Tokenizer();backend.batch_size=2;backend.max_length=512
        self.assertEqual(backend.visible_spans('query',['abcdefghi','']),[(0,6),(0,0)])
        kwargs=backend.tokenizer.calls[0][2]
        self.assertEqual(kwargs,dict(padding=True,truncation='only_second',max_length=512,return_tensors='np',return_offsets_mapping=True))
        self.assertEqual(backend.visible_spans('query',[]),[])

    def test_invalid_offsets_are_rejected(self):
        backend=object.__new__(OnnxPairBackend);backend.tokenizer=Tokenizer();backend.tokenizer.bad=True;backend.batch_size=1;backend.max_length=512
        with self.assertRaises(ValueError):backend.visible_spans('q',['abc'])

    def test_pool_returns_visibility_owner_after_exception(self):
        class Worker:
            def __init__(self,fail=False):self.fail=fail
            def visible_spans(self,q,docs):
                if self.fail:raise ValueError('fixture')
                return [(0,len(d)) for d in docs]
        pool=object.__new__(PooledOnnxPairBackend);pool._workers=queue.Queue(maxsize=2)
        pool._workers.put(Worker(True));pool._workers.put(Worker())
        with self.assertRaises(ValueError):pool.visible_spans('q',['abc'])
        self.assertEqual(pool._workers.qsize(),2)
        self.assertEqual(pool.visible_spans('q',['abc']),[(0,3)])
        self.assertEqual(pool._workers.qsize(),2)


class VisibilityIntegrationTests(unittest.TestCase):
    setUp=fixtures.SourceIntegrationTests.setUp
    tearDown=fixtures.SourceIntegrationTests.tearDown

    def test_unavailable_spans_keep_successful_pair_order(self):
        before=self.db.search(user_id='alice',query='Museum',top_k=100);self.db.flags['source_aware_dedup']=True
        with patch.object(self.backend,'visible_spans',side_effect=RuntimeError('fixture')):
            after=self.db.search(user_id='alice',query='Museum',top_k=100)
        self.assertEqual([r.id for r in before.results],[r.id for r in after.results])
        self.assertTrue(all('source_visibility_unavailable' in r.evidence_flags for r in after.results))
        self.assertFalse(any('pair_rerank_fallback' in r.evidence_flags for r in after.results))

    def test_misaligned_spans_cannot_move_records(self):
        before=self.db.search(user_id='alice',query='Museum',top_k=100);self.db.flags['source_aware_dedup']=True
        with patch.object(self.backend,'visible_spans',return_value=[]):
            after=self.db.search(user_id='alice',query='Museum',top_k=100)
        self.assertEqual([r.id for r in before.results],[r.id for r in after.results])

    def test_default_off_does_not_tokenize_visibility(self):
        with patch.object(self.backend,'visible_spans',side_effect=AssertionError('must not be called')) as provider:
            self.db.search(user_id='alice',query='Museum',top_k=100)
        provider.assert_not_called()


class VisibilityPoolConcurrencyTests(unittest.TestCase):
    def test_score_and_offsets_share_the_same_two_owner_bound(self):
        import threading
        import time
        from concurrent.futures import ThreadPoolExecutor
        lock=threading.Lock();state={'active':0,'peak':0}
        class Worker:
            def __init__(self):self.busy=False
            def work(self,kind,docs):
                with lock:
                    if self.busy:raise AssertionError('tokenizer owner reused concurrently')
                    self.busy=True;state['active']+=1;state['peak']=max(state['peak'],state['active'])
                try:
                    time.sleep(.005)
                    return [(0,len(d)) for d in docs] if kind=='spans' else [1.]*len(docs)
                finally:
                    with lock:self.busy=False;state['active']-=1
            def score(self,q,docs):return self.work('score',docs)
            def visible_spans(self,q,docs):return self.work('spans',docs)
        pool=object.__new__(PooledOnnxPairBackend);pool._workers=queue.Queue(maxsize=2)
        for _ in range(2):pool._workers.put(Worker())
        def run(i):return pool.visible_spans('q',['abc']) if i%2 else pool.score('q',['abc'])
        with ThreadPoolExecutor(max_workers=8) as executor:list(executor.map(run,range(24)))
        self.assertLessEqual(state['peak'],2);self.assertEqual(state['active'],0);self.assertEqual(pool._workers.qsize(),2)
