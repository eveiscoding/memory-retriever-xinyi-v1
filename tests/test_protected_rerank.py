import math
import unittest

from aml_retriever.config import RetrieverConfig
from aml_retriever import features
from aml_retriever.protected_rerank import protected_order
from aml_retriever.retriever import RetrieverDB


class Backend:
    identity = "fixture"
    model_hash = "fixture-v1"
    dimension = 8

    def __init__(self):
        self.calls = 0
        self.fail = False

    def embed(self, texts):
        self.calls += 1
        if self.fail:
            raise RuntimeError("unavailable")
        return [[float("target" in t), float("target" not in t)] + [0.] * 6
                for t in texts]


def records(count):
    return [dict(id=str(i), flags=[], channel_ranks={}, source_ids=[str(i)])
            for i in range(count)]


class ProtectedOrderTests(unittest.TestCase):
    def test_question_words_are_not_entities_but_names_survive(self):
        self.assertEqual(features.extract_protection_entities(
            'How Which When What Who Where Did Does Is The Alice NASA MoMA'),
            ['Alice', 'NASA', 'MoMA'])
        self.assertIn('北京', features.extract_entities('北京'))
        self.assertEqual(features.extract_entities('How Which Alice'), ['How', 'Which', 'Alice'])

    def test_entity_flags_from_question_words_do_not_anchor(self):
        original = records(10)
        original[1].update(flags=['entity'], content='How is your day?')
        original[2].update(flags=['entity'], content='Alice confirmed the booking.')
        ranked = protected_order(original, list(range(10)), query='How did Alice book?')
        self.assertEqual(ranked[0]['id'], '0')
        self.assertNotEqual(ranked[1]['id'], '1')
        self.assertEqual(ranked[2]['id'], '2')

    def test_temporal_and_current_state_keep_front_bucket_order(self):
        for query in ('How long between two visits?', 'Where did Rachel move recently?',
                      'When did I graduate?', 'Which trip happened first?'):
            original = records(100)
            ranked = protected_order(original, list(range(100)), query=query)
            self.assertEqual([r['id'] for r in ranked[:20]], [str(i) for i in range(20)])
            self.assertEqual(ranked[20]['id'], '99')

    def test_event_relations_are_separate_from_current_state_intent(self):
        for query in ('How long between my two visits?', 'Which happened first before dinner?',
                      'How many months after graduation?', 'In what order did I move?',
                      'Which of my visits happened first?',
                      '两次参观相隔多久？', '搬家和入职的先后顺序'):
            with self.subTest(query=query):
                self.assertTrue(features.has_event_relation_intent(query))
        for query in ('What is my current address?', 'Recommend a museum',
                      'How many tickets?', 'the afternoon walk', ''):
            with self.subTest(query=query):
                self.assertFalse(features.has_event_relation_intent(query))
        self.assertFalse(features.has_temporal_intent('How long between my two visits?'))

    def test_duplicate_views_do_not_exhaust_escape_quota(self):
        original = records(10)
        for i in range(5):
            original[i]['flags'] = ['entity']
        original[0]['source_ids'] = ['event-a', 'context']
        original[1]['source_ids'] = ['event-a']
        original[2]['content'] = '  SAME evidence '
        original[3]['content'] = 'same   EVIDENCE'
        ranked = protected_order(original, list(range(10)), exact_quota=3)
        # Distinct provenance/content at 0, 2, 4 is anchored. The two duplicate
        # records can move; no evidence is discarded, including shared sources.
        self.assertEqual([ranked[i]['id'] for i in (0, 2, 4)], ['0', '2', '4'])
        self.assertNotEqual(ranked[1]['id'], '1')
        self.assertNotEqual(ranked[3]['id'], '3')
        self.assertEqual({r['id'] for r in ranked}, {str(i) for i in range(10)})

    def test_temporal_top5_preserves_both_events_and_boundary_membership(self):
        for boundary, pool in ((20, 100), (3, 100), (20, 3), (2, 4)):
            for count in (0, 1, 5, 12, 120):
                original = records(count)
                ranked = protected_order(original, list(range(count)), query=
                    'How long between the first visit and the second visit?',
                    exact_quota=0, boundary=boundary, pool=pool)
                for cutoff in (min(5, pool), boundary, pool):
                    self.assertEqual({r['id'] for r in ranked[:cutoff]},
                                     {str(i) for i in range(min(cutoff, count))})
                self.assertEqual([r['id'] for r in ranked[pool:]],
                                 [str(i) for i in range(pool, count)])

    def test_membership_boundaries_and_diagnostics(self):
        original = records(120)
        ranked = protected_order(original, list(range(120)))
        self.assertEqual({r['id'] for r in ranked[:20]}, {str(i) for i in range(20)})
        self.assertEqual({r['id'] for r in ranked[:100]}, {str(i) for i in range(100)})
        self.assertEqual([r['id'] for r in ranked[100:]], [str(i) for i in range(100, 120)])
        self.assertEqual([r['id'] for r in ranked[:5]], ['0', '4', '3', '2', '1'])
        self.assertEqual(ranked[1]['channel_ranks'], {'pre_rerank': 5, 'post_rerank': 2})

    def test_exact_escape_quota(self):
        original = records(10)
        for i in range(4):
            original[i]['flags'] = ['number']
        ranked = protected_order(original, list(range(10)), exact_quota=3)
        self.assertEqual([r['id'] for r in ranked[:4]], ['0', '1', '2', '4'])

    def test_reject_invalid_scores_before_mutating(self):
        for values in ([1.], [math.nan, 1.], [math.inf, 1.]):
            original = records(2)
            with self.assertRaises(ValueError):
                protected_order(original, values)
            self.assertEqual(original[0]['channel_ranks'], {})


class SemanticIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.backend = Backend()
        self.cfg = RetrieverConfig().with_flags(semantic_rerank=True, vector=True, views=False)
        self.cfg.semantic_rerank_exact_quota = 0
        self.db = RetrieverDB(self.cfg, embedding_backend=self.backend)
        self.db.add(request_id='r', user_id='u', session_id='s', messages=[
            {'role': 'user', 'content': 'topic ' + ('target ' if i % 3 == 0 else '')
             + f'entry {i}'} for i in range(110)
        ])

    def tearDown(self):
        self.db.close()

    def search(self, user='u', limit=100):
        return self.db.search(user_id=user, query='topic target', top_k=limit)

    def test_actual_baseline_sets_are_preserved_with_dense_flag_on(self):
        self.db.semantic_enabled = False
        self.db.dense_enabled = False
        baseline = self.search()
        self.db.semantic_enabled = True
        self.db.dense_enabled = True
        candidate = self.search()
        for cutoff in (5, 20, 100):
            self.assertEqual({r.id for r in baseline.results[:cutoff]},
                             {r.id for r in candidate.results[:cutoff]})
        self.assertTrue(all(r.semantic_score is not None for r in candidate.results))
        self.assertEqual(baseline.total, candidate.total)

    def test_cached_scores_and_failure_fallback(self):
        first = self.search()
        calls = self.backend.calls
        self.assertEqual([r.id for r in first.results], [r.id for r in self.search().results])
        self.assertEqual(self.backend.calls, calls)
        self.db._semantic_cache.clear()
        self.backend.fail = True
        result = self.search()
        self.assertTrue(all('semantic_rerank_fallback' in r.evidence_flags for r in result.results))
        self.assertTrue(all(r.semantic_score is None for r in result.results))

    def test_event_query_guard_is_used_by_search(self):
        query = 'How long between topic target entry visits?'
        self.db.semantic_enabled = False
        self.db.dense_enabled = False
        baseline = self.db.search(user_id='u', query=query, top_k=100)
        self.db.semantic_enabled = True
        candidate = self.db.search(user_id='u', query=query, top_k=100)
        self.assertGreater(len(baseline.results), 5)
        for cutoff in (5, 20, 100):
            self.assertEqual({r.id for r in baseline.results[:cutoff]},
                             {r.id for r in candidate.results[:cutoff]})
        self.assertTrue(all(r.semantic_score is not None for r in candidate.results))

    def test_small_limit_and_isolation(self):
        self.assertEqual(len(self.search(limit=5).results), 5)
        self.assertEqual(self.search(user='other').results, [])

    def test_backend_required(self):
        with self.assertRaises(ValueError):
            RetrieverDB(RetrieverConfig().with_flags(semantic_rerank=True))


if __name__ == '__main__':
    unittest.main()
