"""Inference and score fusion."""

from .bahamas_realtime import score_bahamas_realtime
from .calibration import CalibrationSummary, ThresholdCalibrator, summarize_scores
from .engine import AnomalyFusionEngine, SpatialScorer, TemporalScorer
from .evaluation import ClassificationMetrics, combine_weighted_scores, evaluate_threshold
from .global_scoring import score_global_dataset
from .observed_scoring import (
    build_spatial_training_samples,
    build_temporal_training_samples,
    prepare_observed_features,
    score_observed_features,
    score_spatial_rows,
    score_temporal_rows,
    train_observed_models,
)

__all__ = [
    "AnomalyFusionEngine",
    "CalibrationSummary",
    "ClassificationMetrics",
    "SpatialScorer",
    "TemporalScorer",
    "ThresholdCalibrator",
    "score_bahamas_realtime",
    "build_spatial_training_samples",
    "build_temporal_training_samples",
    "combine_weighted_scores",
    "evaluate_threshold",
    "prepare_observed_features",
    "score_observed_features",
    "score_spatial_rows",
    "score_temporal_rows",
    "train_observed_models",
    "score_global_dataset",
    "summarize_scores",
]
