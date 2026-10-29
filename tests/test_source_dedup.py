import copy
import json
import unittest
from unittest.mock import patch
from aml_retriever.source_dedup import source_aware_order
from aml_retriever.config import RetrieverConfig
from aml_retriever.retriever import RetrieverDB


def rec(i, sources, text, **extra):
    return dict(id=i,source_ids=sources,content=text,score=10.,flags=[],**extra)


class SourceOrderTests(unittest.TestCase):
    def test_message_deferred_and_new_fact_kept(self):
        rows=[rec('view',['a','b'],'user: alpha\nassistant: beta'),rec('msg',['a'],'alpha'),rec('new',['c'],'gamma')]
        before=copy.deepcopy(rows);ordered,a=source_aware_order(rows,user_id='u')
        self.assertEqual([r['id'] for r in ordered],['view','new','msg'])
        self.assertEqual(a['msg']['covered_by'],'view');self.assertEqual(rows,before)
        self.assertTrue(all(any(r is old for old in rows) for r in ordered))

    def test_distinct_events_with_same_text_kept(self):
        rows=[rec('old',['old-msg'],'bought one'),rec('new',['new-msg'],'bought one')]
        self.assertEqual(source_aware_order(rows,user_id='u')[0],rows)

    def test_partial_overlap_and_additional_context_kept(self):
        rows=[rec('a',['x','y'],'alpha beta'),rec('b',['y','z'],'beta gamma'),rec('c',['x'],'alpha new detail')]
        self.assertEqual(source_aware_order(rows,user_id='u')[0],rows)

    def test_union_of_sources_does_not_replace_joint_context(self):
        rows=[rec('a',['x'],'alpha'),rec('b',['y'],'beta'),rec('joint',['x','y'],'alpha beta')]
        self.assertEqual(source_aware_order(rows,user_id='u')[0],rows)

    def test_unknown_or_malformed_provenance_is_retained(self):
        for sources in (None,[],[''],[None],'x'):
            rows=[rec('a',['x'],'alpha'),rec('b',sources,'alpha')]
            self.assertEqual(source_aware_order(rows,user_id='u')[1]['b']['status'],'retained')

    def test_whitespace_only_normalization_and_case(self):
        rows=[rec('a',['x'],'user: alpha   beta'),rec('b',['x'],'alpha\n beta'),rec('c',['x'],'Alpha beta')]
        out,a=source_aware_order(rows,user_id='u')
        self.assertEqual([r['id'] for r in out],['a','c','b']);self.assertEqual(a['b']['status'],'deferred')

    def test_stable_tail_and_witness_is_retained(self):
        rows=[rec('a',['x','y'],'alpha beta'),rec('b',['x'],'alpha'),rec('c',['z'],'gamma'),rec('d',['y'],'beta')]
        out,a=source_aware_order(rows,user_id='u')
        self.assertEqual([r['id'] for r in out],['a','c','b','d'])
        self.assertEqual([a[i]['covered_by'] for i in ('b','d')],['a','a'])

    def test_duplicate_ids_and_cross_tenant_rejected_atomically(self):
        for rows in ([rec('a',['x'],'alpha'),rec('a',['y'],'beta')],
                     [rec('a',['x'],'alpha',user_id='u'),rec('b',['x'],'alpha',user_id='other')]):
            before=copy.deepcopy(rows)
            with self.assertRaises(ValueError):source_aware_order(rows,user_id='u')
            self.assertEqual(rows,before)

    def test_empty_content_and_empty_pool(self):
        self.assertEqual(source_aware_order([],user_id='u'),([],{}))
        rows=[rec('a',['x'],'alpha'),rec('b',['x'],'')]
        self.assertEqual(source_aware_order(rows,user_id='u')[1]['b']['status'],'retained')

    def test_prefix_complete_marker_coverage_is_not_lost(self):
        rows=[rec('ab',['a','b'],'alpha beta'),rec('a',['a'],'alpha'),rec('g',['g'],'gamma'),rec('b',['b'],'beta')]
        out,_=source_aware_order(rows,user_id='u')
        for k in range(1,len(rows)+1):
            for marker in ['alpha','beta','alpha beta','gamma']:
                if any(marker in r['content'] for r in rows[:k]):
                    self.assertTrue(any(marker in r['content'] for r in out[:k]))


class Pair:
    identity='fixture-not-quality'
    def __init__(self):self.calls=[]
    def score(self,q,docs):self.calls.append((q,list(docs)));return [len(d) for d in docs]


class SourceIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.backend=Pair();self.db=RetrieverDB(RetrieverConfig().with_flags(pair_rerank=True),pair_backend=self.backend)
        for u in ('alice','bob'):
            self.db.add(user_id=u,request_id="fixture",session_id='s',messages=[dict(role='user',content='Museum item '+str(i)+' '+u) for i in range(8)])
    def tearDown(self):self.db.close()

    def test_default_off_and_after_model_scope_content_score_contract(self):
        self.assertFalse(self.db.flags.get('source_aware_dedup',True));self.assertFalse(self.backend.calls)
        before=self.db.search(user_id='alice',query='Museum',top_k=100)
        model_input=copy.deepcopy(self.backend.calls[-1]);self.db.flags['source_aware_dedup']=True
        after=self.db.search(user_id='alice',query='Museum',top_k=100)
        self.assertEqual(self.backend.calls[-1],model_input)
        self.assertEqual({r.id for r in before.results},{r.id for r in after.results})
        self.assertEqual({r.id:(r.content,r.score,r.source_message_ids) for r in before.results},{r.id:(r.content,r.score,r.source_message_ids) for r in after.results})
        self.assertTrue(all(r.user_id=='alice' and 'bob' not in r.content for r in after.results))
        self.assertTrue(any('source_duplicate_deferred' in r.evidence_flags for r in after.results))
        for r in after.results:self.assertIn(';source_dedup=',r.provenance)

    def test_failure_returns_unchanged_order(self):
        before=self.db.search(user_id='alice',query='Museum',top_k=100);self.db.flags['source_aware_dedup']=True
        with patch('aml_retriever.retriever.source_aware_order',side_effect=ValueError('fixture')):
            after=self.db.search(user_id='alice',query='Museum',top_k=100)
        self.assertEqual([r.id for r in before.results],[r.id for r in after.results])
        self.assertTrue(all('source_dedup_fallback' in r.evidence_flags for r in after.results))
