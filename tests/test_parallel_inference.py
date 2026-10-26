import unittest
from tools.parallel_inference import compare, corpus


class ParallelProofTests(unittest.TestCase):
    def test_detects_score_and_order_changes(self):
        result = compare([1., 2., 3.], [3., 2., 1.])
        self.assertFalse(result['exact'])
        self.assertFalse(result['model_order_equal'])
        self.assertEqual(result['max_abs_diff'], 2.)

    def test_tie_order_and_small_float_difference(self):
        result = compare([1., 1., 0.], [1.+1e-8, 1., 0.])
        self.assertTrue(result['allclose_1e6'])
        self.assertFalse(result['exact'])
        self.assertTrue(result['rrf_order_equal'])

    def test_invalid_results_fail(self):
        for values in [[], [1.], [float('nan'), 2.]]:
            with self.assertRaises(ValueError):
                compare([1., 2.], values)

    def test_corpus_is_fixed_diverse_and_complete(self):
        self.assertEqual(corpus(), corpus())
        self.assertEqual(len(corpus()), 4)
        self.assertTrue(all(len(set(docs)) == 100 for _,docs in corpus()))
