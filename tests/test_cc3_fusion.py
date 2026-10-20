import copy
import math
import unittest
from unittest.mock import patch

from aml_retriever.config import RetrieverConfig
from aml_retriever.api import MemoryService
from aml_retriever.pair_rerank import pair_order
from aml_retriever.retriever import RetrieverDB
from tests.test_pair_rerank import Backend


def rows(bm25, rules):
    return [dict(id=str(i), content=str(i), score=100-i, flags=[], channel_ranks={},
                 cc3_bm25_score=b, cc3_rule_score=r)
            for i, (b, r) in enumerate(zip(bm25, rules))]


class ThreeChannelFusionTests(unittest.TestCase):
    def test_equal_weight_after_independent_normalization_and_stable_ties(self):
        result = pair_order(rows([1e-6, 0, 0], [0, 100, 0]), [-10, -10, 10],
                            pool=3, mode='cc3', weight=.99)
        self.assertEqual([r['id'] for r in result], ['0', '1', '2'])
        for row in result:
            self.assertAlmostEqual(row['fusion_scores']['fused'], 1/3)
            self.assertIn('pair_cc3', row['flags'])
            self.assertNotIn('pair_cc', row['flags'])
        self.assertEqual([r['score'] for r in result], [100, 99, 98])

    def test_hand_calculated_scores_pool_tail_and_unused_alpha(self):
        source = rows([10, 0, 5, 1e10], [0, 10, 10, 1e10])
        tail = copy.deepcopy(source[-1])
        result = pair_order(source, [0, 0, 10], pool=3, mode='cc3', weight=0)
        self.assertEqual([r['id'] for r in result], ['2', '0', '1', '3'])
        self.assertAlmostEqual(result[0]['fusion_scores']['fused'], 5/6)
        self.assertEqual(result[-1], tail)
        second = pair_order(rows([10, 0, 5, 1e10], [0, 10, 10, 1e10]),
                            [0, 0, 10], pool=3, mode='cc3', weight=1)
        self.assertEqual(result, second)

    def test_constant_channels_and_negative_rule_scores(self):
        result = pair_order(rows([0, 0, 0], [-5, -3, -1]), [7]*3,
                            pool=3, mode='cc3')
        self.assertEqual([r['id'] for r in result], ['2', '1', '0'])
        self.assertAlmostEqual(result[0]['fusion_scores']['fused'], 1/3)
        constant = pair_order(rows([0]*3, [-2]*3), [7]*3, pool=3, mode='cc3')
        self.assertEqual([r['id'] for r in constant], ['0', '1', '2'])

    def test_extreme_finite_scores_do_not_overflow(self):
        result = pair_order(rows([-1e308, 0, 1e308], [0]*3), [0]*3,
                            pool=3, mode='cc3')
        self.assertTrue(all(math.isfinite(r['fusion_scores']['fused']) for r in result))
        self.assertEqual([r['id'] for r in result], ['2', '1', '0'])

    def test_bad_components_reject_before_mutating_any_result(self):
        for key in ('cc3_bm25_score', 'cc3_rule_score'):
            for bad in (None, True, math.nan, math.inf, 'bad', 'missing'):
                source = rows([3, 2, 1], [1, 2, 3])
                if bad == 'missing':
                    del source[1][key]
                else:
                    source[1][key] = bad
                before = copy.deepcopy(source)
                with self.assertRaises((ValueError, TypeError, KeyError)):
                    pair_order(source, [1, 2, 3], pool=3, mode='cc3')
                self.assertEqual(source, before)

    def test_config_and_protection_exclusion(self):
        with patch.dict('os.environ', {'AML_PAIR_RERANK_MODE': 'cc3'}):
            self.assertEqual(RetrieverConfig.from_env().validate().pair_rerank_mode, 'cc3')
        for flag in ('pair_top20_protect', 'pair_exact_protect'):
            with self.assertRaises(ValueError):
                RetrieverConfig(pair_rerank_mode='cc3').with_flags(**{flag: True}).validate()


class ThreeChannelIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.backend = Backend()
        self.db = RetrieverDB(
            RetrieverConfig(pair_rerank_mode='cc3', supersession_max_pairs=2)
            .with_flags(pair_rerank=True), pair_backend=self.backend)
        for user in ('alice', 'bob'):
            for i in range(8):
                self.db.add(user_id=user, session_id=str(i), request_id=str(i), messages=[
                    dict(role='user', content=f'Museum budget updated to {i+10} for {user}',
                         timestamp=1704067200000+i*86400000)])

    def tearDown(self):
        self.db.close()

    def test_actual_search_keeps_baseline_pool_and_exposes_auditable_formula(self):
        self.db.pair_enabled = False
        baseline = self.db.search(user_id='alice', query='current museum budget', top_k=100)
        self.db.pair_enabled = True
        result = self.db.search(user_id='alice', query='current museum budget', top_k=100)
        self.assertEqual({r.id for r in result.results}, {r.id for r in baseline.results})
        self.assertEqual({r.id: r.score for r in result.results},
                         {r.id: r.score for r in baseline.results})
        fused = []
        for row in result.results:
            self.assertIn('pair_cc3', row.evidence_flags)
            self.assertNotIn('pair_rerank_fallback', row.evidence_flags)
            self.assertNotIn('bob', row.content)
            d = row.fusion_scores
            self.assertEqual(d['model_raw'], row.semantic_score)
            self.assertAlmostEqual(d['fused'], (d['bm25']+d['rule']+d['model'])/3)
            fused.append(d['fused'])
        self.assertEqual(fused, sorted(fused, reverse=True))

    def test_rule_channel_is_independent_of_fts_order_and_governance_truncation(self):
        # Keep membership fixed while changing the lexical signal. Aggregated
        # views can otherwise be removed differently by baseline source dedup.
        self.db.flags['views'] = False
        query = 'current museum budget'
        tokens = ['museum', 'budget']
        with self.db.connection() as con:
            candidates = self.db._fts_candidates(con, 'alice', tokens)
        n = len(candidates)
        self.assertGreater(n, 2)
        def search(reverse):
            changed = [(rid, kind, -float(n-i if reverse else i+1))
                       for i, (rid, kind, _) in enumerate(candidates)]
            changed.sort(key=lambda item: item[2])
            with patch.object(self.db, '_fts_candidates', return_value=changed):
                return self.db.search(user_id='alice', query=query, top_k=100)
        left, right = search(False), search(True)
        rules = lambda result: {r.id: r.fusion_scores['rule_raw'] for r in result.results}
        self.assertEqual(rules(left), rules(right))
        lexical = lambda result: {r.id: r.fusion_scores['bm25_raw'] for r in result.results}
        self.assertNotEqual(lexical(left), lexical(right))

    def test_failure_retains_baseline_and_no_partial_fusion_diagnostics(self):
        self.db.pair_enabled = False
        baseline = self.db.search(user_id='alice', query='Museum', top_k=100)
        self.db.pair_enabled = True
        self.backend.fail = True
        result = self.db.search(user_id='alice', query='Museum', top_k=100)
        self.assertEqual([r.id for r in result.results], [r.id for r in baseline.results])
        for row in result.results:
            self.assertIn('pair_rerank_fallback', row.evidence_flags)
            self.assertEqual(row.fusion_scores, {})
            self.assertIsNone(row.semantic_score)

    def test_non_fts_candidate_has_zero_bm25_evidence(self):
        with patch.object(self.db, '_fts_candidates', return_value=[]):
            result = self.db.search(user_id='alice', query='Museum', top_k=100)
        self.assertTrue(result.results)
        self.assertTrue(all(r.fusion_scores['bm25_raw'] == 0 for r in result.results))
        self.assertTrue(all(r.fusion_scores['bm25'] == 0 for r in result.results))

    def test_official_api_diagnostics_follow_provenance_switch(self):
        service = MemoryService(self.db.config, db=self.db)
        service.include_provenance = True
        payload = dict(user_id='alice', query='Museum', top_k=100)
        detailed = service.official_search(payload)['data']
        self.assertTrue(detailed)
        self.assertTrue(all('fused' in row['fusion_scores'] for row in detailed))
        service.include_provenance = False
        minimal = service.official_search(payload)['data']
        self.assertEqual([r['id'] for r in detailed], [r['id'] for r in minimal])
        self.assertTrue(all('fusion_scores' not in r for r in minimal))
