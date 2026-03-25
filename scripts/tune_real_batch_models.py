from __future__ import annotations

import json
from itertools import product
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mad_ai.core.config import load_config
from mad_ai.ingest import SplitBatchSensorIngestor
from mad_ai.inference import prepare_observed_features, score_observed_features, train_observed_models
from mad_ai.models.spatial import CNNAnomalyModel
from mad_ai.models.temporal import LSTMAnomalyModel
from mad_ai.wmm import AnalyticMagneticModel, WMMMagneticModel


def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config/real_batch_large.yaml")
    config = load_config(config_path)
    schema_mapping = config.get("schema_mapping", {})
    split_sources = config.get("split_sources", {})
    tuning = config.get("tuning", {})
    training_config = config.get("training", {})
    if not split_sources:
        raise ValueError("real batch tuning requires split_sources in config.")

    ingestor = SplitBatchSensorIngestor(
        schema_mapping=schema_mapping,
        sqlite_table_name=str(config.get("source_options", {}).get("sqlite_table_name", "sensor_readings")),
    )
    raw_splits = ingestor.load_splits(split_sources)
    magnetic_model = _make_magnetic_model(config)
    prepared = {name: prepare_observed_features(frame, magnetic_model) for name, frame in raw_splits.items()}

    spatial_window_sizes = [int(value) for value in tuning.get("spatial_window_sizes", [12, 18])]
    temporal_sequence_lengths = [int(value) for value in tuning.get("temporal_sequence_lengths", [6, 12])]
    strides = [int(value) for value in tuning.get("strides", [3, 6])]
    spatial_weights = [float(value) for value in tuning.get("spatial_weights", [0.5])]
    spatial_epochs = int(training_config.get("spatial_epochs", 3))
    temporal_epochs = int(training_config.get("temporal_epochs", 4))
    spatial_batch_size = int(training_config.get("spatial_batch_size", 16))
    temporal_batch_size = int(training_config.get("temporal_batch_size", 24))

    rows: list[dict[str, float | int]] = []
    for spatial_window_size, temporal_sequence_length, stride, spatial_weight in product(
        spatial_window_sizes,
        temporal_sequence_lengths,
        strides,
        spatial_weights,
    ):
        temporal_weight = 1.0 - spatial_weight
        spatial_model, temporal_model, _ = train_observed_models(
            prepared["train"],
            spatial_model=CNNAnomalyModel(epochs=spatial_epochs, batch_size=spatial_batch_size, latent_channels=24),
            temporal_model=LSTMAnomalyModel(epochs=temporal_epochs, batch_size=temporal_batch_size, hidden_size=48),
            spatial_window_size=spatial_window_size,
            temporal_sequence_length=temporal_sequence_length,
            stride=stride,
        )
        nominal_scored, nominal_summary = score_observed_features(
            prepared["nominal_eval"],
            spatial_model=spatial_model,
            temporal_model=temporal_model,
            calibration_features=prepared["calibration"],
            spatial_window_size=spatial_window_size,
            temporal_sequence_length=temporal_sequence_length,
            stride=stride,
            spatial_weight=spatial_weight,
            temporal_weight=temporal_weight,
        )
        anomalous_scored, anomalous_summary = score_observed_features(
            prepared["anomalous_eval"],
            spatial_model=spatial_model,
            temporal_model=temporal_model,
            threshold=float(nominal_summary["threshold"]),
            spatial_window_size=spatial_window_size,
            temporal_sequence_length=temporal_sequence_length,
            stride=stride,
            spatial_weight=spatial_weight,
            temporal_weight=temporal_weight,
        )
        anomalous_metrics = anomalous_summary.get("metrics", {})
        nominal_metrics = nominal_summary.get("metrics", {})
        rows.append(
            {
                "spatial_window_size": spatial_window_size,
                "temporal_sequence_length": temporal_sequence_length,
                "stride": stride,
                "spatial_weight": spatial_weight,
                "temporal_weight": temporal_weight,
                "threshold": float(nominal_summary["threshold"]),
                "nominal_false_positive_count": int(nominal_summary["anomaly_count"]),
                "anomalous_recall": float(anomalous_metrics.get("recall", 0.0)),
                "anomalous_precision": float(anomalous_metrics.get("precision", 0.0)),
                "anomalous_f1": float(anomalous_metrics.get("f1_score", 0.0)),
                "anomalous_accuracy": float(anomalous_metrics.get("accuracy", 0.0)),
                "nominal_accuracy": float(nominal_metrics.get("accuracy", 0.0)),
            }
        )

    leaderboard = pd.DataFrame(rows).sort_values(
        ["anomalous_f1", "anomalous_recall", "nominal_accuracy"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    output_dir = Path("outputs/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_path = output_dir / "real_batch_tuning_leaderboard.csv"
    summary_path = output_dir / "real_batch_tuning_summary.json"
    leaderboard.to_csv(leaderboard_path, index=False)

    best = leaderboard.iloc[0].to_dict() if not leaderboard.empty else {}
    summary_path.write_text(
        json.dumps(
            {
                "config_path": str(config_path),
                "candidate_count": int(len(leaderboard)),
                "best_candidate": best,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved tuning leaderboard to {leaderboard_path}")
    print(f"Saved tuning summary to {summary_path}")


def _make_magnetic_model(config: dict) -> AnalyticMagneticModel | WMMMagneticModel:
    backend = str(config.get("magnetic_backend", "wmm")).strip().lower()
    if backend == "analytic":
        return AnalyticMagneticModel()
    return WMMMagneticModel(cache_path="data/cache/wmm_cache.json")


if __name__ == "__main__":
    main()
