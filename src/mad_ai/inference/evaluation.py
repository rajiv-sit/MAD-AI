from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ClassificationMetrics:
    threshold: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    accuracy: float
    f1_score: float


def evaluate_threshold(scores, labels, threshold: float) -> ClassificationMetrics:
    score_array = np.asarray(scores, dtype=float).reshape(-1)
    label_array = np.asarray(labels, dtype=int).reshape(-1)
    if score_array.size == 0 or label_array.size == 0 or score_array.size != label_array.size:
        raise ValueError("scores and labels must be non-empty and have the same length.")

    predictions = score_array >= threshold
    positives = label_array == 1
    negatives = ~positives

    tp = int(np.sum(predictions & positives))
    tn = int(np.sum((~predictions) & negatives))
    fp = int(np.sum(predictions & negatives))
    fn = int(np.sum((~predictions) & positives))

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    accuracy = _safe_divide(tp + tn, score_array.size)
    f1_score = _safe_divide(2.0 * precision * recall, precision + recall)

    return ClassificationMetrics(
        threshold=float(threshold),
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        accuracy=accuracy,
        f1_score=f1_score,
    )


def combine_weighted_scores(spatial_scores, temporal_scores, spatial_weight: float = 0.5, temporal_weight: float = 0.5) -> np.ndarray:
    spatial = np.asarray(spatial_scores, dtype=float).reshape(-1)
    temporal = np.asarray(temporal_scores, dtype=float).reshape(-1)
    if spatial.size == 1 and temporal.size > 1:
        spatial = np.repeat(spatial[0], temporal.size)
    if temporal.size == 1 and spatial.size > 1:
        temporal = np.repeat(temporal[0], spatial.size)
    if spatial.size != temporal.size:
        raise ValueError("spatial_scores and temporal_scores must align or be broadcastable from a scalar.")

    total = spatial_weight + temporal_weight
    return ((spatial_weight * spatial) + (temporal_weight * temporal)) / total


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)
