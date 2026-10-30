import unittest
from unittest.mock import patch
from aml_retriever.model_input_budget import compact, model_query, request_context, _context
from aml_retriever.api import ApiError
from aml_retriever.parent_window_scoring import ParentWindowRetriever
from aml_retriever import source_evidence_output as output
from tests import test_fixed_window_service as fixture
from tests.test_source_evidence_output import CharTokenizer

class BudgetTests(unittest.TestCase):
    def test_short_inputs_exact_and_options_question_priority(self):
        tok=CharTokenizer()
        self.assertEqual(model_query('question',['answer one','answer two'],tok),'question answer one answer two')
        q='question'*25
        bounded=model_query(q,['x'*2000],tok)
        self.assertTrue(bounded.startswith(q));self.assertLessEqual(len(tok.encode(bounded)),256)
        self.assertEqual(model_query(q+'z'*900,['option'],tok),compact(q+'z'*900,tok))
    def test_unicode_head_tail_and_budget(self):
        tok=CharTokenizer();text='开始🙂'*300+'问题结尾'
        bounded=compact(text,tok)
        self.assertTrue(bounded.startswith('开始'));self.assertTrue(bounded.endswith('问题结尾'))
        self.assertLessEqual(len(tok.encode(bounded)),256)
    def test_context_reset(self):
        self.assertIsNone(_context.get())
        with request_context('one','a'):
            with request_context('two','b'):self.assertEqual(_context.get()['parent_query'],'b')
            self.assertEqual(_context.get()['request_id'],'one')
        self.assertIsNone(_context.get())

class RecoveryTests(fixture.ServiceTests):
    def test_long_query_and_options_both_stages_bounded(self):
        backend=self.db.pair_backend;observed=[];original=backend.score
        def score(q,texts):
            observed.append(q);self.assertLessEqual(len(backend.tokenizer.encode(q)),256)
            return original(q,texts)
        for query,options in [('TAIL '+('long '*200),[]),('TAIL',['option '*300])]:
            with patch.object(backend,'score',side_effect=score):
                reply=self.svc.official_search(dict(user_id='u',query=query,options=options,top_k=100))
            self.assertTrue(reply['data']);self.assertNotIn('degraded',reply)
        self.assertTrue(observed)
    def test_recall_keeps_original_question_options(self):
        original=self.db.search;q='TAIL '+('full '*200);opts=['option '*500]
        with patch.object(self.db,'search',wraps=original) as search:
            self.svc.official_search(dict(user_id='u',query=q,options=opts,top_k=100))
        self.assertEqual(search.call_args.kwargs['query'],q)
        self.assertEqual(search.call_args.kwargs['options'],opts)
    def test_short_query_options_parent_order_and_output_exact(self):
        legacy=ParentWindowRetriever(self.cfg,pair_backend=self.db.pair_backend)
        try:
            q='TAIL trip';options=['seven days']
            expected=legacy.search(user_id='u',query=q,options=options,top_k=100)
            with self.db.connection() as con:r=output.resolve_sources(con,expected.results,user_id='u')
            b=self.db.pair_backend;c=output.make_candidates(r,q,b.tokenizer)
            pack=output.selected_output(r,output.score_candidates(c,q,b),budget=8192,tokenizer=b.tokenizer)
            with patch.object(self.db,'search',wraps=self.db.search) as search:
                actual=self.svc.official_search(dict(user_id='u',query=q,options=options,top_k=100))
            got=self.db.search(user_id='u',query=q,options=options,top_k=100)
            self.assertEqual([(r.id,r.score) for r in got.results],[(r.id,r.score) for r in expected.results])
            self.assertEqual(actual['data'][0]['content'],pack['rendered'])
        finally:legacy.close()
    def test_model_failure_returns_explicit_validated_raw_prefix(self):
        with patch.object(self.db.pair_backend,'score',side_effect=RuntimeError('PRIVATE_SENTINEL')):
            with self.assertLogs('aml_retriever.fixed_window',level='WARNING') as logs:
                reply=self.svc.official_search(self.query())
        self.assertTrue(reply['degraded']);self.assertTrue(reply['data'])
        self.assertNotIn('PRIVATE_SENTINEL',' '.join(logs.output))
        self.assertNotIn('TAIL trip',' '.join(logs.output))
        self.assertLessEqual(output.tokens(self.db.pair_backend.tokenizer,reply['data'][0]['content']),8192)
        self.assertIn('stage=source_score',' '.join(logs.output))
    def test_window_limit_failure_has_safe_fallback(self):
        with patch('aml_retriever.fixed_window_service.output.make_candidates',side_effect=ValueError('window limit')):
            reply=self.svc.official_search(self.query())
        self.assertTrue(reply['degraded']);self.assertTrue(reply['data'])
    def test_source_integrity_fails_closed_and_logs_no_body(self):
        with patch('aml_retriever.fixed_window_service.output.resolve_sources',side_effect=ValueError('PRIVATE_SENTINEL')):
            with self.assertLogs('aml_retriever.fixed_window',level='ERROR') as logs:
                with self.assertRaises(ApiError) as error:self.svc.official_search(self.query())
        self.assertEqual(error.exception.status,503)
        self.assertIn('stage=source_validation',' '.join(logs.output))
        self.assertNotIn('PRIVATE_SENTINEL',' '.join(logs.output))
    def test_large_role_cannot_exhaust_pair_room(self):
        self.svc.official_add(dict(request_id='role',user_id='roles',session_id='s',
            messages=[dict(role='custom-role'*100,content='TAIL answer')]))
        reply=self.svc.official_search(dict(user_id='roles',query='TAIL',top_k=100))
        self.assertTrue(reply['data']);self.assertNotIn('degraded',reply)
        self.assertIn('custom-role'*100,reply['data'][0]['content'])
