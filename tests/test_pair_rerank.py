import copy
import math
import unittest
from aml_retriever.config import RetrieverConfig
from aml_retriever.pair_rerank import pair_order, load_pair_backend
from aml_retriever.retriever import RetrieverDB


def records():
    return [dict(id=str(i), content='sample '+str(i), flags=[], channel_ranks={}) for i in range(8)]


class PairOrderTests(unittest.TestCase):
    def test_prefix_moves_tail_stable_ties(self):
        result = pair_order(records(), [0, 1, 8, 2, 2], pool=5)
        self.assertEqual([r['id'] for r in result], ['2','3','4','1','0','5','6','7'])
        self.assertEqual(result[0]['channel_ranks'], {'pre_rerank':3,'post_rerank':1,'pair_model':1})
        self.assertNotIn('semantic_score', result[-1])

    def test_invalid_scores_atomic(self):
        for values in ([0]*7, [0]*7+[math.nan], [0]*7+[True], ['bad']*8):
            original = records()
            expected = copy.deepcopy(original)
            with self.assertRaises((ValueError, TypeError)):
                pair_order(original, values, pool=8)
            self.assertEqual(original, expected)

    def test_configuration(self):
        with self.assertRaises(ValueError):
            load_pair_backend('missing-colon')
        with self.assertRaises(ValueError):
            RetrieverConfig().with_flags(pair_rerank=True, semantic_rerank=True).validate()
        with self.assertRaises(ValueError):
            RetrieverDB(RetrieverConfig().with_flags(pair_rerank=True))


class Backend:
    identity = 'fixture-not-quality'
    def __init__(self):
        self.calls = []
        self.fail = False
    def score(self, query, documents):
        self.calls.append((query, list(documents)))
        return [math.nan]*len(documents) if self.fail else list(range(len(documents)))


class PairIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.backend = Backend()
        self.db = RetrieverDB(RetrieverConfig().with_flags(pair_rerank=True), pair_backend=self.backend)
        for user in ('alice','bob'):
            self.db.add(user_id=user, session_id='s', request_id='r', messages=[
                dict(role='user',content='Museum visit '+str(i)+' '+user) for i in range(8)])
    def tearDown(self):
        self.db.close()

    def test_membership_isolation_options_and_no_add_scoring(self):
        self.assertEqual(self.backend.calls, [])
        self.db.pair_enabled = False
        baseline = self.db.search(user_id='alice',query='Museum',options=['visit'],top_k=100)
        self.db.pair_enabled = True
        result = self.db.search(user_id='alice',query='Museum',options=['visit'],top_k=100)
        self.assertEqual([r.id for r in result.results], list(reversed([r.id for r in baseline.results])))
        self.assertEqual(self.backend.calls[-1][0], 'Museum visit')
        self.assertTrue(all('bob' not in d for d in self.backend.calls[-1][1]))
        self.assertTrue(all('pair_rerank' in r.evidence_flags for r in result.results))
        self.assertFalse(self.db.embedding_enabled)

    def test_failure_returns_baseline(self):
        self.db.pair_enabled = False
        before = self.db.search(user_id='alice',query='Museum',top_k=100)
        self.db.pair_enabled = True
        self.backend.fail = True
        after = self.db.search(user_id='alice',query='Museum',top_k=100)
        self.assertEqual([r.id for r in before.results], [r.id for r in after.results])
        self.assertTrue(all('pair_rerank_fallback' in r.evidence_flags for r in after.results))
        self.assertTrue(all(r.semantic_score is None for r in after.results))

    def test_cc_runs_through_retriever_with_baseline_scores(self):
        self.db.config.pair_rerank_mode = 'cc'
        self.db.config.pair_rrf_weight = .5
        result = self.db.search(user_id='alice', query='Museum', top_k=100)
        self.assertTrue(all('pair_cc' in row.evidence_flags for row in result.results))
        self.assertFalse(any('pair_rerank_fallback' in row.evidence_flags
                             for row in result.results))
        self.assertTrue(all(row.semantic_score is not None for row in result.results))


class PairFusionTests(unittest.TestCase):
    def test_zero_weight_retains_baseline_one_matches_pure(self):
        values = [0, 7, 3, 9, 1, 6, 4, 2]
        ids = lambda rows: [r['id'] for r in rows]
        self.assertEqual(ids(pair_order(records(), values, pool=8, mode='rrf', weight=0)),
                         ids(records()))
        self.assertEqual(ids(pair_order(records(), values, pool=8, mode='rrf', weight=1)),
                         ids(pair_order(records(), values, pool=8)))

    def test_fusion_is_invariant_to_logit_scale_and_keeps_pool(self):
        values = [7, 6, 5, 4, 3, 2, 1, 0]
        values[0], values[7] = values[7], values[0]
        a = pair_order(records(), values, pool=8, mode='rrf', weight=.25)
        b = pair_order(records(), [100*v-70 for v in values], pool=8, mode='rrf', weight=.25)
        self.assertEqual([r['id'] for r in a], [r['id'] for r in b])
        self.assertEqual({r['id'] for r in a}, {r['id'] for r in records()})
        self.assertEqual(a[0]['id'], '1')
        self.assertIn('pair_rrf', a[0]['flags'])

    def test_invalid_policy_is_rejected_before_mutation(self):
        for params in ({'mode':'other'}, {'weight':-1}, {'weight':float('nan')},
                       {'weight':True}, {'k':0}, {'k':float('inf')}):
            original=records()
            with self.assertRaises(ValueError):
                pair_order(original, list(range(8)), pool=8, **params)
            self.assertEqual(original, records())
        with self.assertRaises(ValueError):
            RetrieverConfig(pair_rerank_mode='bad').validate()


class PairScoreFusionTests(unittest.TestCase):
    def rows(self, baseline):
        rows = records()
        for row, score in zip(rows, baseline):
            row['score'] = score
        return rows

    def test_weight_endpoints_match_baseline_and_model_order(self):
        baseline = [8, 7, 6, 5, 4, 3, 2, 1]
        model = [0, 1, 2, 3, 4, 5, 6, 7]
        ids = lambda rows: [row['id'] for row in rows]
        self.assertEqual(ids(pair_order(self.rows(baseline), model, pool=8,
                                        mode='cc', weight=0)),
                         list(map(str, range(8))))
        self.assertEqual(ids(pair_order(self.rows(baseline), model, pool=8,
                                        mode='cc', weight=1)),
                         list(map(str, range(7, -1, -1))))

    def test_minmax_preserves_score_gap_that_rank_fusion_discards(self):
        baseline = [100, 99, 1, 0, -1, -2, -3, -4]
        model = [0, 1, 8, 7, 6, 5, 4, 3]
        cc = pair_order(self.rows(baseline), model, pool=8, mode='cc', weight=.25)
        rrf = pair_order(self.rows(baseline), model, pool=8, mode='rrf', weight=.25)
        self.assertEqual(cc[0]['id'], '1')
        self.assertEqual(rrf[0]['id'], '2')
        self.assertIn('pair_cc', cc[0]['flags'])
        self.assertNotIn('pair_rrf', cc[0]['flags'])

    def test_constant_channels_have_no_artificial_discrimination(self):
        model = [0, 1, 2, 3, 4, 5, 6, 7]
        only_model = pair_order(self.rows([5] * 8), model, pool=8,
                                mode='cc', weight=.5)
        self.assertEqual([row['id'] for row in only_model],
                         list(map(str, range(7, -1, -1))))
        both_constant = pair_order(self.rows([5] * 8), [9] * 8, pool=8,
                                   mode='cc', weight=.5)
        self.assertEqual([row['id'] for row in both_constant],
                         list(map(str, range(8))))

    def test_bad_baseline_scores_fail_atomically(self):
        for bad in (None, True, math.nan, math.inf, 'bad'):
            rows = self.rows([8, 7, 6, 5, 4, 3, 2, 1])
            rows[3]['score'] = bad
            before = copy.deepcopy(rows)
            with self.assertRaises((ValueError, TypeError)):
                pair_order(rows, list(range(8)), pool=8, mode='cc')
            self.assertEqual(rows, before)

    def test_cc_configuration_accepts_shared_alpha(self):
        RetrieverConfig(pair_rerank_mode='cc', pair_rrf_weight=.75).validate()


class PairProtectionTests(unittest.TestCase):
    def rows(self):
        return [dict(id=str(i),content='record '+str(i),flags=[],source_ids=['s'+str(i)],channel_ranks={}) for i in range(25)]

    def test_top20_membership_is_fixed_but_internal_order_is_free(self):
        rows=self.rows();rows[0]['flags']=['date']
        result=pair_order(rows,range(25),pool=25,protect_top20=True)
        self.assertEqual([r['id'] for r in result],list(map(str,range(19,-1,-1)))+list(map(str,range(24,19,-1))))
        self.assertFalse(any('pair_exact_anchor' in r['flags'] for r in result))

    def test_exact_only_deduplicates_anchors_but_allows_top20_crossing(self):
        rows=self.rows()
        for i in range(5):rows[i]['flags']=['date']
        rows[1]['source_ids']=rows[0]['source_ids'];rows[2]['content']=rows[0]['content']
        result=pair_order(rows,range(25),pool=25,protect_exact=True,query='2026')
        self.assertEqual([r['id'] for r in result if 'pair_exact_anchor' in r['flags']],['0','3','4'])
        for i in (0,3,4):self.assertEqual(result[i]['id'],str(i))
        self.assertIn('24',[r['id'] for r in result[:20]])
        self.assertFalse(any('pair_top20_protect' in r['flags'] for r in result))

    def test_unrelated_entity_flag_cannot_reserve_a_slot(self):
        rows=self.rows();rows[0]['flags']=['entity'];rows[0]['content']='How unrelated'
        result=pair_order(rows,range(25),pool=25,protect_exact=True,query='Which museum?')
        self.assertFalse(any('pair_exact_anchor' in r['flags'] for r in result))
        self.assertEqual(result[0]['id'],'24')

    def test_invalid_policy_or_scores_fail_before_mutation(self):
        for opts,values in [({'protect_top20':True},[0]*24+[math.nan]),({'protect_exact':True,'mode':'rrf'},list(range(25)))]:
            rows=self.rows();before=copy.deepcopy(rows)
            with self.assertRaises(ValueError):pair_order(rows,values,pool=25,**opts)
            self.assertEqual(rows,before)
        with self.assertRaises(ValueError):RetrieverConfig(pair_rerank_mode='rrf').with_flags(pair_top20_protect=True).validate()

    def test_env_switches_and_actual_retriever_propagation(self):
        from unittest.mock import patch
        with patch.dict('os.environ',{'AML_FLAG_PAIR_TOP20_PROTECT':'1','AML_FLAG_PAIR_EXACT_PROTECT':'0'}):cfg=RetrieverConfig.from_env().validate()
        self.assertTrue(cfg.flags['pair_top20_protect']);self.assertFalse(cfg.flags['pair_exact_protect'])
        db=RetrieverDB(cfg.with_flags(pair_rerank=True),pair_backend=Backend())
        try:
            for i in range(25):db.add(user_id='u',session_id=str(i),request_id=str(i),messages=[dict(role='user',content='museum '+str(i))])
            db.pair_enabled=False;before=db.search(user_id='u',query='museum',top_k=100)
            db.pair_enabled=True;after=db.search(user_id='u',query='museum',top_k=100)
            self.assertEqual({r.id for r in before.results[:20]},{r.id for r in after.results[:20]})
            self.assertTrue(all('pair_top20_protect' in r.evidence_flags for r in after.results))
            self.assertFalse(any('pair_rerank_fallback' in r.evidence_flags for r in after.results))
        finally:db.close()
