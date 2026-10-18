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
        self.assertEqual(result[0]['channel_ranks'], {'pre_rerank':3,'post_rerank':1})
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
