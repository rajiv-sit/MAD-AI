from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from mad_ai.inference.calibration import CalibrationSummary, ThresholdCalibrator
from mad_ai.inference.engine import AnomalyFusionEngine
from mad_ai.models.spatial import CNNAnomalyModel
from mad_ai.models.temporal import LSTMAnomalyModel


def score_global_dataset(
    data: pd.DataFrame,
    spatial_model: CNNAnomalyModel | None = None,
    temporal_model: LSTMAnomalyModel | None = None,
    patch_radius: int = 1,
    calibrator_percentile: float = 97.5,
) -> tuple[pd.DataFrame, dict[str, object]]:
    required = {
        "latitude_deg",
        "longitude_deg",
        "altitude_m",
        "baseline_total_nt",
        "baseline_declination_deg",
        "baseline_inclination_deg",
        "horizontal_intensity_nt",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns for global scoring: {sorted(missing)}")

    scored = data.copy()
    scored["spatial_anomaly_score"] = 0.0
    scored["temporal_anomaly_score"] = 0.0
    scored["final_anomaly_score"] = 0.0
    scored["is_anomaly"] = False

    spatial_model = spatial_model or CNNAnomalyModel(epochs=3, batch_size=64, latent_channels=24)
    temporal_model = temporal_model or LSTMAnomalyModel(epochs=4, batch_size=128, hidden_size=32)

    spatial_feature_columns = [
        "baseline_total_nt",
        "baseline_declination_deg",
        "baseline_inclination_deg",
    ]
    temporal_feature_columns = [
        "baseline_total_nt",
        "baseline_declination_deg",
        "baseline_inclination_deg",
        "horizontal_intensity_nt",
    ]

    patch_batches: list[np.ndarray] = []
    patch_indices: list[np.ndarray] = []
    for _, altitude_frame in scored.groupby("altitude_m", sort=True):
        ordered_frame = altitude_frame.sort_values(["latitude_deg", "longitude_deg"]).reset_index()
        feature_grid = _frame_to_feature_grid(ordered_frame, spatial_feature_columns)
        patches = _extract_patches(feature_grid, patch_radius=patch_radius)
        patch_batches.append(patches)
        patch_indices.append(ordered_frame["index"].to_numpy(dtype=int))

    if patch_batches:
        spatial_train = np.concatenate(patch_batches, axis=0)
        spatial_model.train(spatial_train)
        start = 0
        for patches, indices in zip(patch_batches, patch_indices):
            count = len(patches)
            scores = np.asarray(spatial_model.score(patches), dtype=float).reshape(-1)
            scored.loc[indices, "spatial_anomaly_score"] = scores
            start += count

    temporal_sequences, temporal_group_indices = _build_altitude_sequences(scored, temporal_feature_columns)
    if len(temporal_sequences) > 0:
        temporal_model.train(temporal_sequences)
        temporal_scores = np.asarray(temporal_model.score(temporal_sequences), dtype=float).reshape(-1)
        for indices, score in zip(temporal_group_indices, temporal_scores):
            scored.loc[indices, "temporal_anomaly_score"] = float(score)

    preliminary_scores = 0.5 * scored["spatial_anomaly_score"].to_numpy(dtype=float) + 0.5 * scored["temporal_anomaly_score"].to_numpy(dtype=float)
    calibration = ThresholdCalibrator(percentile=calibrator_percentile).calibrate(preliminary_scores)
    fusion_engine = AnomalyFusionEngine(threshold=calibration.threshold)

    final_scores: list[float] = []
    final_flags: list[bool] = []
    for spatial_score, temporal_score in zip(
        scored["spatial_anomaly_score"].to_numpy(dtype=float),
        scored["temporal_anomaly_score"].to_numpy(dtype=float),
    ):
        result = fusion_engine.fuse(float(spatial_score), float(temporal_score))
        final_scores.append(result.final_score)
        final_flags.append(result.is_anomaly)

    scored["final_anomaly_score"] = final_scores
    scored["is_anomaly"] = final_flags

    summary = {
        "calibration": asdict(calibration),
        "spatial_mean": float(scored["spatial_anomaly_score"].mean()),
        "temporal_mean": float(scored["temporal_anomaly_score"].mean()),
        "final_mean": float(scored["final_anomaly_score"].mean()),
        "anomaly_count": int(scored["is_anomaly"].sum()),
        "patch_radius": int(patch_radius),
    }
    return scored, summary


def _frame_to_feature_grid(frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    latitudes = np.sort(frame["latitude_deg"].unique())
    longitudes = np.sort(frame["longitude_deg"].unique())
    lat_index = {value: idx for idx, value in enumerate(latitudes)}
    lon_index = {value: idx for idx, value in enumerate(longitudes)}
    grid = np.zeros((len(latitudes), len(longitudes), len(feature_columns)), dtype=np.float32)

    for row in frame.itertuples(index=False):
        grid_r = lat_index[float(row.latitude_deg)]
        grid_c = lon_index[float(row.longitude_deg)]
        for idx, column in enumerate(feature_columns):
            grid[grid_r, grid_c, idx] = float(getattr(row, column))
    return grid


def _extract_patches(feature_grid: np.ndarray, patch_radius: int) -> np.ndarray:
    if patch_radius <= 0:
        return feature_grid[np.newaxis, ...]
    patch_size = (2 * patch_radius) + 1
    padded = np.pad(feature_grid, ((patch_radius, patch_radius), (patch_radius, patch_radius), (0, 0)), mode="edge")
    patches: list[np.ndarray] = []
    for row in range(feature_grid.shape[0]):
        for col in range(feature_grid.shape[1]):
            patches.append(padded[row : row + patch_size, col : col + patch_size, :])
    return np.stack(patches, axis=0)


def _build_altitude_sequences(
    data: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[np.ndarray, list[np.ndarray]]:
    sequences: list[np.ndarray] = []
    group_indices: list[np.ndarray] = []
    ordered = data.sort_values(["latitude_deg", "longitude_deg", "altitude_m"]).reset_index()
    for _, group in ordered.groupby(["latitude_deg", "longitude_deg"], sort=False):
        sequence = group[feature_columns].to_numpy(dtype=np.float32)
        if len(sequence) < 2:
            continue
        sequences.append(sequence)
        group_indices.append(group["index"].to_numpy(dtype=int))
    if not sequences:
        return np.zeros((0, 0, len(feature_columns)), dtype=np.float32), []
    return np.stack(sequences, axis=0), group_indices
