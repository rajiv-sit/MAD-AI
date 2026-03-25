from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from mad_ai.datasets import SpatialGridBuilder
from mad_ai.features import ResidualFeatureBuilder, TemporalFeatureBuilder
from mad_ai.inference.calibration import ThresholdCalibrator
from mad_ai.inference.engine import AnomalyFusionEngine
from mad_ai.inference.evaluation import evaluate_threshold
from mad_ai.models.spatial import CNNAnomalyModel
from mad_ai.models.temporal import LSTMAnomalyModel
from mad_ai.wmm import WMMMagneticModel

SPATIAL_FEATURE_COLUMNS = [
    "residual_total_nt",
    "residual_declination_deg",
    "residual_inclination_deg",
    "rolling_residual_total_nt",
]

TEMPORAL_FEATURE_COLUMNS = [
    "residual_total_nt",
    "residual_declination_deg",
    "residual_inclination_deg",
    "delta_residual_total_nt",
    "rolling_residual_total_nt",
]


def prepare_observed_features(
    raw: pd.DataFrame,
    magnetic_model: WMMMagneticModel | None = None,
) -> pd.DataFrame:
    model = magnetic_model or WMMMagneticModel(cache_path="data/cache/wmm_cache.json")
    enriched = ResidualFeatureBuilder(model).transform(raw)
    temporal = TemporalFeatureBuilder().transform(enriched)
    return temporal.reset_index(drop=True)


def train_observed_models(
    nominal_features: pd.DataFrame,
    spatial_model: CNNAnomalyModel | None = None,
    temporal_model: LSTMAnomalyModel | None = None,
    spatial_window_size: int = 24,
    temporal_sequence_length: int = 12,
    stride: int = 6,
) -> tuple[CNNAnomalyModel, LSTMAnomalyModel, dict[str, int]]:
    spatial_samples = _build_spatial_samples(
        nominal_features,
        window_size=spatial_window_size,
        stride=stride,
    )
    temporal_samples = _build_temporal_samples(
        nominal_features,
        sequence_length=temporal_sequence_length,
        stride=stride,
    )
    if len(spatial_samples) == 0 or len(temporal_samples) == 0:
        raise ValueError("Observed training data did not produce enough spatial and temporal samples.")

    spatial_model = spatial_model or CNNAnomalyModel(epochs=8, batch_size=16, latent_channels=24)
    temporal_model = temporal_model or LSTMAnomalyModel(epochs=10, batch_size=32, hidden_size=48)
    spatial_model.train(spatial_samples)
    temporal_model.train(temporal_samples)
    return spatial_model, temporal_model, {
        "spatial_sample_count": int(len(spatial_samples)),
        "temporal_sample_count": int(len(temporal_samples)),
    }


def score_observed_features(
    features: pd.DataFrame,
    spatial_model: CNNAnomalyModel,
    temporal_model: LSTMAnomalyModel,
    calibration_features: pd.DataFrame | None = None,
    threshold: float | None = None,
    calibration_percentile: float = 97.5,
    spatial_window_size: int = 24,
    temporal_sequence_length: int = 12,
    stride: int = 6,
    spatial_weight: float = 0.5,
    temporal_weight: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, object]]:
    scored = features.copy().reset_index(drop=True)
    spatial_scores = _score_rows_with_spatial_windows(
        scored,
        spatial_model,
        window_size=spatial_window_size,
        stride=stride,
    )
    temporal_scores = _score_rows_with_temporal_windows(
        scored,
        temporal_model,
        sequence_length=temporal_sequence_length,
        stride=stride,
    )
    scored["spatial_anomaly_score"] = spatial_scores
    scored["temporal_anomaly_score"] = temporal_scores

    total_weight = spatial_weight + temporal_weight
    scored["final_anomaly_score"] = (
        (spatial_weight * scored["spatial_anomaly_score"]) + (temporal_weight * scored["temporal_anomaly_score"])
    ) / total_weight

    calibration_summary = None
    effective_threshold = threshold
    if effective_threshold is None:
        calibration_source = calibration_features if calibration_features is not None else features
        calibration_scores = _compute_final_scores_for_calibration(
            calibration_source,
            spatial_model,
            temporal_model,
            spatial_window_size=spatial_window_size,
            temporal_sequence_length=temporal_sequence_length,
            stride=stride,
            spatial_weight=spatial_weight,
            temporal_weight=temporal_weight,
        )
        calibration = ThresholdCalibrator(percentile=calibration_percentile).calibrate(calibration_scores)
        effective_threshold = calibration.threshold
        calibration_summary = asdict(calibration)

    engine = AnomalyFusionEngine(
        spatial_weight=spatial_weight,
        temporal_weight=temporal_weight,
        threshold=float(effective_threshold),
    )
    scored["is_anomaly"] = scored["final_anomaly_score"] >= engine.threshold

    summary: dict[str, object] = {
        "rows": int(len(scored)),
        "threshold": float(engine.threshold),
        "spatial_mean": float(scored["spatial_anomaly_score"].mean()),
        "temporal_mean": float(scored["temporal_anomaly_score"].mean()),
        "final_mean": float(scored["final_anomaly_score"].mean()),
        "anomaly_count": int(scored["is_anomaly"].sum()),
    }
    if calibration_summary is not None:
        summary["calibration"] = calibration_summary
    if "is_injected_anomaly" in scored.columns:
        metrics = evaluate_threshold(
            scored["final_anomaly_score"].to_numpy(dtype=float),
            scored["is_injected_anomaly"].astype(int).to_numpy(dtype=int),
            engine.threshold,
        )
        summary["metrics"] = {
            "threshold": metrics.threshold,
            "true_positives": metrics.true_positives,
            "true_negatives": metrics.true_negatives,
            "false_positives": metrics.false_positives,
            "false_negatives": metrics.false_negatives,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "accuracy": metrics.accuracy,
            "f1_score": metrics.f1_score,
        }
    return scored, summary


def _compute_final_scores_for_calibration(
    features: pd.DataFrame,
    spatial_model: CNNAnomalyModel,
    temporal_model: LSTMAnomalyModel,
    spatial_window_size: int,
    temporal_sequence_length: int,
    stride: int,
    spatial_weight: float,
    temporal_weight: float,
) -> np.ndarray:
    spatial_scores = _score_rows_with_spatial_windows(
        features,
        spatial_model,
        window_size=spatial_window_size,
        stride=stride,
    )
    temporal_scores = _score_rows_with_temporal_windows(
        features,
        temporal_model,
        sequence_length=temporal_sequence_length,
        stride=stride,
    )
    total_weight = spatial_weight + temporal_weight
    return ((spatial_weight * spatial_scores) + (temporal_weight * temporal_scores)) / total_weight


def _build_spatial_samples(
    features: pd.DataFrame,
    window_size: int,
    stride: int,
) -> np.ndarray:
    grids: list[np.ndarray] = []
    builder = SpatialGridBuilder(rows=8, cols=8, feature_columns=SPATIAL_FEATURE_COLUMNS)
    for group in _iter_feature_groups(features):
        for start, end in _window_bounds(len(group), window_size, stride):
            window = group.iloc[start:end].copy()
            grids.append(builder.build(window))
    return np.stack(grids, axis=0) if grids else np.zeros((0, 8, 8, len(SPATIAL_FEATURE_COLUMNS)), dtype=np.float32)


def _build_temporal_samples(
    features: pd.DataFrame,
    sequence_length: int,
    stride: int,
) -> np.ndarray:
    sequences: list[np.ndarray] = []
    for group in _iter_feature_groups(features):
        values = group[TEMPORAL_FEATURE_COLUMNS].fillna(0.0).to_numpy(dtype=np.float32)
        for start, end in _window_bounds(len(group), sequence_length, stride):
            sequences.append(_pad_sequence(values[start:end], sequence_length))
    return (
        np.stack(sequences, axis=0)
        if sequences
        else np.zeros((0, sequence_length, len(TEMPORAL_FEATURE_COLUMNS)), dtype=np.float32)
    )


def _score_rows_with_spatial_windows(
    features: pd.DataFrame,
    model: CNNAnomalyModel,
    window_size: int,
    stride: int,
) -> np.ndarray:
    row_scores = np.zeros(len(features), dtype=np.float32)
    row_counts = np.zeros(len(features), dtype=np.float32)
    builder = SpatialGridBuilder(rows=8, cols=8, feature_columns=SPATIAL_FEATURE_COLUMNS)
    for group in _iter_feature_groups(features):
        group_indices = group.index.to_numpy(dtype=int)
        for start, end in _window_bounds(len(group), window_size, stride):
            window = group.iloc[start:end].copy()
            score = float(model.score(builder.build(window)))
            indices = group_indices[start:end]
            row_scores[indices] += score
            row_counts[indices] += 1.0
    row_counts = np.where(row_counts <= 0.0, 1.0, row_counts)
    return row_scores / row_counts


def _score_rows_with_temporal_windows(
    features: pd.DataFrame,
    model: LSTMAnomalyModel,
    sequence_length: int,
    stride: int,
) -> np.ndarray:
    row_scores = np.zeros(len(features), dtype=np.float32)
    row_counts = np.zeros(len(features), dtype=np.float32)
    for group in _iter_feature_groups(features):
        values = group[TEMPORAL_FEATURE_COLUMNS].fillna(0.0).to_numpy(dtype=np.float32)
        group_indices = group.index.to_numpy(dtype=int)
        for start, end in _window_bounds(len(group), sequence_length, stride):
            sequence = _pad_sequence(values[start:end], sequence_length)
            score = float(model.score(sequence))
            indices = group_indices[start:end]
            row_scores[indices] += score
            row_counts[indices] += 1.0
    row_counts = np.where(row_counts <= 0.0, 1.0, row_counts)
    return row_scores / row_counts


def _iter_feature_groups(features: pd.DataFrame):
    if "run_id" in features.columns:
        for _, group in features.sort_values(["run_id", "timestamp"]).groupby("run_id", sort=False):
            yield group.reset_index(drop=False).set_index("index")
        return
    if "track_id" in features.columns:
        for _, group in features.sort_values(["track_id", "timestamp"]).groupby("track_id", sort=False):
            yield group.reset_index(drop=False).set_index("index")
        return
    yield features.sort_values("timestamp").reset_index(drop=False).set_index("index")


def _window_bounds(length: int, window_size: int, stride: int) -> list[tuple[int, int]]:
    if length <= 0:
        return []
    size = max(1, min(window_size, length))
    step = max(1, stride)
    bounds: list[tuple[int, int]] = []
    starts = list(range(0, max(1, length - size + 1), step))
    if not starts:
        starts = [0]
    last_start = max(0, length - size)
    if starts[-1] != last_start:
        starts.append(last_start)
    for start in starts:
        end = min(length, start + size)
        bounds.append((start, end))
    return bounds


def _pad_sequence(values: np.ndarray, sequence_length: int) -> np.ndarray:
    padded = np.zeros((sequence_length, values.shape[1]), dtype=np.float32)
    padded[: len(values)] = values[:sequence_length]
    return padded
