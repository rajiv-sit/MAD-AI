from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from mad_ai.inference.calibration import ThresholdCalibrator
from mad_ai.inference.observed_scoring import prepare_observed_features, score_observed_features, train_observed_models
from mad_ai.models.spatial import CNNAnomalyModel
from mad_ai.models.temporal import LSTMAnomalyModel
from mad_ai.wmm import WMMMagneticModel


def score_bahamas_realtime(
    raw: pd.DataFrame,
    magnetic_model: WMMMagneticModel | None = None,
    warmup_fraction: float = 0.3,
    calibration_fraction_of_warmup: float = 0.25,
    reference_window_mode: str = "stable_window",
    spatial_window_size: int = 24,
    temporal_sequence_length: int = 12,
    stride: int = 6,
    spatial_weight: float = 0.5,
    temporal_weight: float = 0.5,
    calibration_percentile: float = 97.5,
    spatial_model: CNNAnomalyModel | None = None,
    temporal_model: LSTMAnomalyModel | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if raw.empty:
        raise ValueError("Bahamas realtime scoring requires at least one row.")

    warmup_fraction = float(min(max(warmup_fraction, 0.05), 0.9))
    calibration_fraction_of_warmup = float(min(max(calibration_fraction_of_warmup, 0.1), 0.8))

    ordered = raw.sort_values("timestamp").reset_index(drop=True)
    features = prepare_observed_features(ordered, magnetic_model)
    reference_rows = max(
        spatial_window_size,
        temporal_sequence_length,
        int(len(features) * warmup_fraction),
    )
    reference_rows = min(max(reference_rows, 2), len(features) - 1) if len(features) > 1 else 1
    reference_start, reference_end, reference_notes = _select_reference_window(
        features,
        reference_rows=reference_rows,
        reference_window_mode=reference_window_mode,
    )
    reference_frame = features.iloc[reference_start:reference_end].reset_index(drop=True).copy()
    calibration_rows = max(
        temporal_sequence_length,
        int(len(reference_frame) * calibration_fraction_of_warmup),
    )
    calibration_rows = min(max(calibration_rows, 1), len(reference_frame) - 1) if len(reference_frame) > 1 else 1
    training = reference_frame.iloc[:-calibration_rows].reset_index(drop=True).copy()
    calibration = reference_frame.iloc[-calibration_rows:].reset_index(drop=True).copy()
    if training.empty:
        training = reference_frame.iloc[: max(1, len(reference_frame) // 2)].reset_index(drop=True).copy()
        calibration = reference_frame.iloc[len(training) :].reset_index(drop=True).copy()
    if calibration.empty:
        calibration = training.tail(min(len(training), temporal_sequence_length)).reset_index(drop=True).copy()

    spatial_model, temporal_model, training_summary = train_observed_models(
        training,
        spatial_model=spatial_model,
        temporal_model=temporal_model,
        spatial_window_size=spatial_window_size,
        temporal_sequence_length=temporal_sequence_length,
        stride=stride,
    )
    scored, summary = score_observed_features(
        features,
        spatial_model=spatial_model,
        temporal_model=temporal_model,
        calibration_features=calibration,
        calibration_percentile=calibration_percentile,
        spatial_window_size=spatial_window_size,
        temporal_sequence_length=temporal_sequence_length,
        stride=stride,
        spatial_weight=spatial_weight,
        temporal_weight=temporal_weight,
    )

    baseline_calibration = ThresholdCalibrator(percentile=calibration_percentile).calibrate(
        calibration["residual_total_nt"].abs().to_numpy(dtype=float)
    )
    baseline_threshold = float(baseline_calibration.threshold)
    scored["baseline_residual_score"] = scored["residual_total_nt"].abs()
    scored["baseline_is_anomaly"] = scored["baseline_residual_score"] >= baseline_threshold
    scored["realtime_phase"] = [
        "reference" if reference_start <= index < reference_end else "online"
        for index in range(len(scored))
    ]

    summary["reference_strategy"] = {
        "mode": reference_window_mode,
        "warmup_fraction": warmup_fraction,
        "reference_start_row": int(reference_start),
        "reference_end_row": int(reference_end),
        "reference_rows": int(reference_end - reference_start),
        "training_rows": int(len(training)),
        "calibration_rows": int(len(calibration)),
        "calibration_fraction_of_warmup": calibration_fraction_of_warmup,
        "notes": reference_notes,
    }
    summary["baseline_reference"] = asdict(baseline_calibration)
    summary["baseline_threshold"] = baseline_threshold
    summary["baseline_anomaly_count"] = int(scored["baseline_is_anomaly"].sum())
    summary["fused_anomaly_count"] = int(scored["is_anomaly"].sum())
    summary["training_summary"] = training_summary
    summary["top_anomalies"] = _top_anomalies(scored)
    return scored, summary


def _top_anomalies(scored: pd.DataFrame, limit: int = 10) -> list[dict[str, object]]:
    hotspots = (
        scored.sort_values("final_anomaly_score", ascending=False)
        .head(limit)
        [["timestamp", "latitude_deg", "longitude_deg", "altitude_m", "final_anomaly_score", "residual_total_nt"]]
        .copy()
    )
    hotspots["timestamp"] = hotspots["timestamp"].astype(str)
    return hotspots.to_dict(orient="records")


def _select_reference_window(
    features: pd.DataFrame,
    reference_rows: int,
    reference_window_mode: str,
) -> tuple[int, int, str]:
    if len(features) <= reference_rows:
        return 0, len(features), "The full dataset was used as the reference window because it is shorter than the requested window."

    mode = reference_window_mode.strip().lower()
    if mode == "chronological_warmup":
        return 0, reference_rows, "The earliest segment of the flight is treated as nominal reference for training and threshold calibration."

    if mode != "stable_window":
        raise ValueError(f"Unsupported reference window mode: {reference_window_mode}")

    residual = features["residual_total_nt"].to_numpy(dtype=float)
    delta = features["delta_residual_total_nt"].to_numpy(dtype=float) if "delta_residual_total_nt" in features.columns else np.zeros(len(features), dtype=float)
    best_score = float("inf")
    best_start = 0
    step = max(1, reference_rows // 8)
    last_start = len(features) - reference_rows
    starts = list(range(0, last_start + 1, step))
    if starts[-1] != last_start:
        starts.append(last_start)
    for start in starts:
        end = start + reference_rows
        window_residual = np.abs(residual[start:end])
        window_delta = np.abs(delta[start:end])
        score = float(np.median(window_residual) + 0.5 * np.std(window_residual) + 0.25 * np.median(window_delta))
        if score < best_score:
            best_score = score
            best_start = start
    best_end = best_start + reference_rows
    return (
        best_start,
        best_end,
        "A stability-based reference window was selected using the lowest combined residual magnitude and residual-change score.",
    )
