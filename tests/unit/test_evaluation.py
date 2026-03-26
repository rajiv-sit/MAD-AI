from __future__ import annotations

import unittest

import numpy as np

from mad_ai.inference import combine_weighted_scores, evaluate_threshold


class EvaluationTestCase(unittest.TestCase):
    def test_combine_weighted_scores_supports_scalar_broadcast(self) -> None:
        combined = combine_weighted_scores(10.0, np.array([1.0, 3.0]), spatial_weight=0.25, temporal_weight=0.75)
        self.assertEqual(combined.shape, (2,))
        self.assertTrue(np.all(combined > 0.0))

    def test_evaluate_threshold_returns_confusion_metrics(self) -> None:
        metrics = evaluate_threshold(scores=[0.1, 0.9, 0.2, 1.2], labels=[0, 1, 0, 1], threshold=0.5)
        self.assertEqual(metrics.true_positives, 2)
        self.assertEqual(metrics.true_negatives, 2)
        self.assertEqual(metrics.false_positives, 0)
        self.assertEqual(metrics.false_negatives, 0)
        self.assertEqual(metrics.accuracy, 1.0)

    def test_combine_weighted_scores_rejects_unaligned_arrays(self) -> None:
        with self.assertRaises(ValueError):
            combine_weighted_scores([1.0, 2.0], [3.0, 4.0, 5.0])

    def test_evaluate_threshold_rejects_mismatched_inputs(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_threshold(scores=[0.1, 0.2], labels=[1], threshold=0.5)


if __name__ == "__main__":
    unittest.main()
