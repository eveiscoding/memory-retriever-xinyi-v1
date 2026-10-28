import copy
import math
import unittest
from unittest.mock import patch

from aml_retriever.config import RetrieverConfig
from aml_retriever.fact_support import fact_support_order, query_needs
from aml_retriever.retriever import RetrieverDB


def rows(*texts):
    return [dict(id=str(i), content=text, flags=[], channel_ranks={}, score=10-i)
            for i, text in enumerate(texts)]


class Embedding:
    identity = 'fixture-not-a-quality-model'
    model_hash = 'fixture-v1'
    dimension = 8
    def __init__(self):
        self.calls = []
        self.bad = None
    def embed(self, texts):
        self.calls.append(list(texts))
        if self.bad == 'count':
            return []
        if self.bad == 'dimension':
            return [[1.]] * len(texts)
        if self.bad == 'nan':
            return [[math.nan] * 8 for _ in texts]
        if self.bad == 'bool':
            return [[True] * 8 for _ in texts]
        if self.bad == 'zero':
            return [[0.] * 8 for _ in texts]
        if self.bad == 'error':
            raise RuntimeError('private-provider-message-must-not-leak')
        return [[1.] + [0.] * 7 for _ in texts]


class FactSupportUnitTests(unittest.TestCase):
    def test_quantity_fact_beats_resource_advice(self):
        source = rows('You can check reports for the number of staff.',
                      'Our office has 47 staff.', 'I have 8 bicycles.')
        result = fact_support_order(source, 'How many staff does our office have?')
        self.assertEqual(result[0]['id'], '1')
        self.assertEqual({r['id'] for r in result}, {r['id'] for r in source})
        self.assertEqual(source, rows(*(r['content'] for r in source)))
        self.assertEqual(result[0]['score'], source[1]['score'])

    def test_duration_and_money_are_different_attributes(self):
        r = rows('I have 17 watches.', 'I have collected watches for six years.')
        out = fact_support_order(r, 'How long have I collected watches?')
        self.assertEqual(out[0]['id'], '1')
        r = rows('I spent 12 days in Lima.', 'I paid $83 for my hotel in Lima.')
        out = fact_support_order(r, 'What did I spend on my hotel in Lima?')
        self.assertEqual(out[0]['id'], '1')

    def test_same_place_wrong_money_attribute_is_rejected(self):
        r = rows('I paid $18 for dinner in Oslo.', 'My hotel in Oslo cost $91 per night.')
        out = fact_support_order(r, 'What did my hotel in Oslo cost?')
        self.assertEqual(out[0]['id'], '1')
        self.assertEqual(out[1]['fact_support_trace']['support'], 0)

    def test_question_plan_and_negation_do_not_become_facts(self):
        q = 'How long have I collected stamps?'
        for content in ['I will collect stamps for two years.',
                        'Have I collected stamps for two years?',
                        'I have not collected stamps for two years.']:
            r = rows(content, 'No evidence here.')
            self.assertIs(fact_support_order(r, q), r)

    def test_assistant_restatement_is_not_blanket_penalized(self):
        r = rows('Consider books about stamp collecting.',
                 'assistant: You have collected stamps for nine months.')
        out = fact_support_order(r, 'How long have I collected stamps?')
        self.assertEqual(out[0]['id'], '1')

    def test_same_query_conjuncts_do_not_require_both_in_one_sentence(self):
        q = 'What is the total number of days I spent in Peru and Chile?'
        self.assertEqual(len(query_needs(q)), 2)
        r = rows('I might visit Peru and Chile.', 'I stayed in Chile for five days.',
                 'I spent four days in Peru.')
        out = fact_support_order(r, q)
        support = {x['id']:x['fact_support_trace']['support'] for x in out}
        self.assertGreater(support['1'], 0)
        self.assertGreater(support['2'], 0)

    def test_embedding_cannot_override_explicit_wrong_entity(self):
        e = Embedding()
        r = rows('I stayed in Spain for four weeks.', 'I stayed in Portugal for six days.')
        out = fact_support_order(r, 'How long was I in Portugal?', mode='embedding', backend=e)
        self.assertEqual(out[0]['id'], '1')
        self.assertFalse(any('Spain' in t for t in e.calls[0]))

    def test_embedding_can_match_zero_lexical_overlap_after_fact_gate(self):
        e = Embedding()
        r = rows('You should read about watch collecting.', 'I have owned timepieces for seven years.')
        rule = fact_support_order(r, 'How long have I collected watches?')
        self.assertIs(rule, r)
        semantic = fact_support_order(r, 'How long have I collected watches?', mode='embedding', backend=e)
        self.assertEqual(semantic[0]['id'], '1')
        self.assertEqual(semantic[0]['fact_support_trace']['backend_hash'], e.model_hash)

    def test_unknown_zero_weight_and_constant_support_are_exact_noops(self):
        e = Embedding()
        for q, opts in [('Recommend a hobby', {}), ('How many coins do I have?', {'weight':0}),
                        ('How many coins do I have?', {})]:
            r = rows('I have 12 coins.', 'I have 12 coins.')
            before = copy.deepcopy(r)
            self.assertIs(fact_support_order(r, q, backend=e, **opts), r)
            self.assertEqual(r, before)
        self.assertEqual(e.calls, [])

    def test_long_view_tail_is_a_separate_signal_not_pair_input(self):
        text = 'Unrelated background. ' * 150 + 'I have collected coins for eight months.'
        r = rows('Read about collecting coins.', text)
        out = fact_support_order(r, 'How long have I collected coins?')
        self.assertEqual(out[0]['id'], '1')
        self.assertGreater(out[0]['fact_support_trace']['clause_index'], 100)

    def test_budgets_and_deduplication_bound_backend_work(self):
        e = Embedding()
        r = rows('I have 12 coins. I have 13 coins.', 'I have 14 coins.')
        out = fact_support_order(r, 'How many coins do I have?', mode='embedding', backend=e,
                                 max_total=2, per_doc=1)
        self.assertEqual(len(e.calls), 1)
        self.assertEqual(len(e.calls[0]), 3)  # query plus two unique clauses
        self.assertIs(out, r)  # equal support
        e = Embedding()
        r = rows('I have 12 coins.', 'I have 12 coins.', 'Read about coins.')
        fact_support_order(r, 'How many coins do I have?', mode='embedding', backend=e)
        self.assertEqual(len(e.calls[0]), 2)

    def test_cut_prefix_and_oversized_clauses_are_not_partially_embedded(self):
        r = rows('Background. ' * 10 + 'I have collected coins for eight months.', 'No evidence.')
        self.assertIs(fact_support_order(r, 'How long have I collected coins?', max_chars=60), r)
        r = rows('I have 12 coins ' + 'and coins ' * 100, 'No evidence.')
        self.assertIs(fact_support_order(r, 'How many coins do I have?'), r)

    def test_malformed_embeddings_fail_before_mutation(self):
        for bad in ['count', 'dimension', 'nan', 'bool', 'zero', 'error']:
            e = Embedding(); e.bad = bad
            r = rows('Read about coins.', 'I have 12 coins.'); before = copy.deepcopy(r)
            with self.assertRaises((ValueError, RuntimeError)):
                fact_support_order(r, 'How many coins do I have?', mode='embedding', backend=e)
            self.assertEqual(r, before)

    def test_invalid_policy_fails_closed(self):
        for opts in [dict(weight=math.nan), dict(weight=True), dict(k=0), dict(mode='other'),
                     dict(per_doc=0), dict(max_total=1601), dict(max_chars=True), dict(min_similarity=1)]:
            with self.assertRaises(ValueError):
                fact_support_order([], 'Unknown', **opts)
        with self.assertRaises(ValueError):
            RetrieverConfig(fact_support_dimension=True).validate()

    def test_duration_value_is_not_an_item_count(self):
        r = rows('I have collected coins for 12 years.', 'Read about coins.')
        self.assertIs(fact_support_order(r, 'How many coins do I have?'), r)

    def test_money_value_is_not_an_item_count(self):
        r = rows('I spent $135 on coins.', 'Read about coins.')
        self.assertIs(fact_support_order(r, 'How many coins do I have?'), r)

    def test_money_attribute_propagates_to_second_location(self):
        r = rows('I paid $21 for dinner in Quito.', 'My hotel in Quito cost $88.')
        out = fact_support_order(r, 'What did my hotel in Lima cost compared to Quito?')
        self.assertEqual(out[0]['id'], '1')
        self.assertEqual(out[1]['fact_support_trace']['support'], 0)

    def test_third_person_owner_is_not_self(self):
        e = Embedding()
        r = rows('My sister has 12 coins.', 'I have 9 coins.')
        out = fact_support_order(r, 'How many coins do I have?', mode='embedding', backend=e)
        self.assertEqual(out[0]['id'], '1')
        self.assertFalse(any('sister' in text for text in e.calls[0]))

    def test_date_range_and_month_may_are_supported(self):
        for month in ['May','December']:
            r = rows('Read about holidays in Peru.', 'I went to Peru from '+month+' 3 to 8.')
            out = fact_support_order(r, 'How long was I in Peru?')
            self.assertEqual(out[0]['id'], '1')

    def test_no_new_recency_rule_or_answer_value_selection(self):
        r = rows('I have 10 coins.', 'I have 20 coins.')
        self.assertIs(fact_support_order(r, 'How many coins do I currently have?'), r)


class FactSupportIntegrationTests(unittest.TestCase):
    def config(self, **kwargs):
        return RetrieverConfig(**kwargs).with_flags(fact_support=True, views=False)

    def seed(self, db):
        for user in ['alice','bob']:
            db.add(user_id=user, session_id='s', request_id='r', messages=[
                dict(role='assistant',content='You can read reports about '+user+' coins.'),
                dict(role='user',content='I have 12 '+user+' coins.')])

    def test_search_only_no_add_embeddings_or_index_changes(self):
        e = Embedding()
        db = RetrieverDB(self.config(fact_support_mode='embedding'), fact_embedding_backend=e)
        try:
            self.seed(db)
            self.assertEqual(e.calls, [])
            self.assertFalse(db.embedding_enabled)
            db.fact_support_enabled = False
            before = db.search(user_id='alice', query='How many alice coins do I have?', top_k=100)
            db.fact_support_enabled = True
            after = db.search(user_id='alice', query='How many alice coins do I have?', top_k=100)
            self.assertEqual({r.id for r in before.results}, {r.id for r in after.results})
            self.assertEqual({r.id:r.score for r in before.results}, {r.id:r.score for r in after.results})
            self.assertTrue(e.calls)
            self.assertFalse(any('bob' in t for call in e.calls for t in call))
            with db.connection() as con:
                self.assertEqual(con.execute('SELECT COUNT(*) FROM embeddings').fetchone()[0], 0)
            self.assertTrue(any('fact_support=' in r.provenance for r in after.results))
            self.assertFalse(any('private-provider' in r.provenance for r in after.results))
        finally:
            db.close()

    def test_disabled_and_zero_weight_do_not_load_fact_backend(self):
        for cfg in [RetrieverConfig(fact_support_mode='embedding', fact_support_backend='missing:backend'),
                    self.config(fact_support_mode='embedding', fact_support_backend='missing:backend', fact_support_weight=0)]:
            db = RetrieverDB(cfg)
            self.assertIsNone(db.fact_embedding_backend)
            db.close()

    def test_failed_backend_keeps_order_and_hides_provider_error(self):
        e = Embedding()
        db = RetrieverDB(self.config(fact_support_mode='embedding'), fact_embedding_backend=e)
        try:
            self.seed(db); db.fact_support_enabled=False
            before=db.search(user_id='alice', query='How many alice coins do I have?', top_k=100)
            db.fact_support_enabled=True; e.bad='error'
            after=db.search(user_id='alice', query='How many alice coins do I have?', top_k=100)
            self.assertEqual([r.id for r in before.results], [r.id for r in after.results])
            self.assertTrue(all('fact_support_fallback' in r.evidence_flags for r in after.results))
            self.assertNotIn('private-provider', repr(after))
        finally:
            db.close()

    def test_options_do_not_invent_fact_query_intent(self):
        e=Embedding(); db=RetrieverDB(self.config(fact_support_mode='embedding'), fact_embedding_backend=e)
        try:
            self.seed(db)
            db.search(user_id='alice', query='coins', options=['How many coins do I have?'], top_k=100)
            self.assertEqual(e.calls, [])
        finally:
            db.close()

    def test_pair_combination_keeps_pair_membership_and_flags(self):
        class Pair:
            identity='fixture-pair'
            def score(self, query, documents): return list(range(len(documents)))
        cfg=self.config(pair_rerank_mode='rrf').with_flags(pair_rerank=True)
        db=RetrieverDB(cfg, pair_backend=Pair())
        try:
            self.seed(db); db.fact_support_enabled=False
            before=db.search(user_id='alice', query='How many alice coins do I have?', top_k=100)
            db.fact_support_enabled=True
            after=db.search(user_id='alice', query='How many alice coins do I have?', top_k=100)
            self.assertEqual({r.id for r in before.results}, {r.id for r in after.results})
            self.assertTrue(all('pair_rrf' in r.evidence_flags for r in after.results))
            self.assertFalse(any('fallback' in flag for r in after.results for flag in r.evidence_flags))
        finally:
            db.close()

    def test_dashscope_adapter_through_search_with_mock_http(self):
        import io
        import json
        from aml_retriever.dashscope_embedding import DashScopeEmbeddingBackend
        class Response(io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *args): self.close()
        requests=[]
        def respond(request, timeout):
            body=json.loads(request.data); requests.append(body)
            return Response(json.dumps({'data':[
                {'index':i,'embedding':[1.]+[0.]*63} for i in range(len(body['input']))
            ]}).encode())
        backend=DashScopeEmbeddingBackend(dimension=64, api_key='fixture-secret',
                    base_url='https://fixture.invalid/v1', max_retries=0)
        db=RetrieverDB(self.config(fact_support_mode='embedding'), fact_embedding_backend=backend)
        try:
            self.seed(db)
            with patch('urllib.request.urlopen',side_effect=respond):
                result=db.search(user_id='alice',query='How many alice coins do I have?',top_k=100)
            self.assertTrue(requests)
            self.assertTrue(all(p['model']=='text-embedding-v4' for p in requests))
            self.assertTrue(all(p['dimensions']==64 for p in requests))
            self.assertTrue(any('fact_support_embedding' in r.evidence_flags for r in result.results))
            self.assertNotIn('fixture-secret',repr(result))
        finally:
            db.close()

    def test_malformed_environment_does_not_silently_enable_default_weight(self):
        with patch.dict('os.environ', {'AML_FACT_SUPPORT_WEIGHT':'not-a-number'}):
            with self.assertRaisesRegex(ValueError,'environment configuration'):
                RetrieverConfig.from_env()

    def test_env_configuration_and_missing_backend(self):
        with patch.dict('os.environ', {'AML_FLAG_FACT_SUPPORT':'1','AML_FACT_SUPPORT_MODE':'embedding',
                                     'AML_FACT_SUPPORT_WEIGHT':'0.12','AML_FACT_SUPPORT_MAX_TOTAL':'64'}):
            cfg=RetrieverConfig.from_env().validate()
        self.assertTrue(cfg.flags['fact_support']); self.assertEqual(cfg.fact_support_max_total,64)
        self.assertEqual(cfg.fact_support_weight,.12)
        with self.assertRaisesRegex(ValueError,'requires a backend'):
            RetrieverDB(cfg)
