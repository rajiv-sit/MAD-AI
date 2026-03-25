"""Inference and score fusion."""

from .calibration import CalibrationSummary, ThresholdCalibrator, summarize_scores
from .engine import AnomalyFusionEngine, SpatialScorer, TemporalScorer
from .evaluation import ClassificationMetrics, combine_weighted_scores, evaluate_threshold
from .global_scoring import score_global_dataset
from .observed_scoring import prepare_observed_features, score_observed_features, train_observed_models

__all__ = [
    "AnomalyFusionEngine",
    "CalibrationSummary",
    "ClassificationMetrics",
    "SpatialScorer",
    "TemporalScorer",
    "ThresholdCalibrator",
    "combine_weighted_scores",
    "evaluate_threshold",
    "prepare_observed_features",
    "score_observed_features",
    "train_observed_models",
    "score_global_dataset",
    "summarize_scores",
]
