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

    search_space = _build_search_space(tuning, training_config)

    rows: list[dict[str, float | int]] = []
    for candidate in search_space:
        spatial_model, temporal_model, _ = train_observed_models(
            prepared["train"],
            spatial_model=CNNAnomalyModel(
                epochs=int(candidate["spatial_epochs"]),
                learning_rate=float(candidate["spatial_learning_rate"]),
                batch_size=int(candidate["spatial_batch_size"]),
                latent_channels=int(candidate["spatial_latent_channels"]),
                dropout=float(candidate["spatial_dropout"]),
            ),
            temporal_model=LSTMAnomalyModel(
                epochs=int(candidate["temporal_epochs"]),
                learning_rate=float(candidate["temporal_learning_rate"]),
                hidden_size=int(candidate["temporal_hidden_size"]),
                batch_size=int(candidate["temporal_batch_size"]),
                dropout=float(candidate["temporal_dropout"]),
            ),
            spatial_window_size=int(candidate["spatial_window_size"]),
            temporal_sequence_length=int(candidate["temporal_sequence_length"]),
            stride=int(candidate["stride"]),
        )
        nominal_scored, nominal_summary = score_observed_features(
            prepared["nominal_eval"],
            spatial_model=spatial_model,
            temporal_model=temporal_model,
            calibration_features=prepared["calibration"],
            spatial_window_size=int(candidate["spatial_window_size"]),
            temporal_sequence_length=int(candidate["temporal_sequence_length"]),
            stride=int(candidate["stride"]),
            spatial_weight=float(candidate["spatial_weight"]),
            temporal_weight=float(candidate["temporal_weight"]),
        )
        anomalous_scored, anomalous_summary = score_observed_features(
            prepared["anomalous_eval"],
            spatial_model=spatial_model,
            temporal_model=temporal_model,
            threshold=float(nominal_summary["threshold"]),
            spatial_window_size=int(candidate["spatial_window_size"]),
            temporal_sequence_length=int(candidate["temporal_sequence_length"]),
            stride=int(candidate["stride"]),
            spatial_weight=float(candidate["spatial_weight"]),
            temporal_weight=float(candidate["temporal_weight"]),
        )
        anomalous_metrics = anomalous_summary.get("metrics", {})
        nominal_metrics = nominal_summary.get("metrics", {})
        rows.append(
            {
                **candidate,
                "threshold": float(nominal_summary["threshold"]),
                "nominal_false_positive_count": int(nominal_summary["anomaly_count"]),
                "nominal_anomaly_rate": float(nominal_summary["anomaly_count"]) / max(len(nominal_scored), 1),
                "anomalous_anomaly_rate": float(anomalous_summary["anomaly_count"]) / max(len(anomalous_scored), 1),
                "anomalous_recall": float(anomalous_metrics.get("recall", 0.0)),
                "anomalous_precision": float(anomalous_metrics.get("precision", 0.0)),
                "anomalous_f1": float(anomalous_metrics.get("f1_score", 0.0)),
                "anomalous_accuracy": float(anomalous_metrics.get("accuracy", 0.0)),
                "nominal_accuracy": float(nominal_metrics.get("accuracy", 0.0)),
                "spatial_training_final_loss": float(spatial_model.training_history[-1]) if spatial_model.training_history else None,
                "temporal_training_final_loss": float(temporal_model.training_history[-1]) if temporal_model.training_history else None,
            }
        )

    leaderboard = pd.DataFrame(rows).sort_values(
        ["anomalous_f1", "anomalous_recall", "nominal_false_positive_count", "nominal_accuracy"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)
    output_dir = Path("outputs/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_path = output_dir / "real_batch_tuning_leaderboard.csv"
    summary_path = output_dir / "real_batch_tuning_summary.json"
    best_config_path = output_dir / "real_batch_best_config.json"
    leaderboard.to_csv(leaderboard_path, index=False)

    best = leaderboard.iloc[0].to_dict() if not leaderboard.empty else {}
    best_config = _best_candidate_config(config, best) if best else {}
    summary_path.write_text(
        json.dumps(
            {
                "config_path": str(config_path),
                "candidate_count": int(len(leaderboard)),
                "best_candidate": best,
                "best_config_path": str(best_config_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    best_config_path.write_text(json.dumps(best_config, indent=2), encoding="utf-8")

    print(f"Saved tuning leaderboard to {leaderboard_path}")
    print(f"Saved tuning summary to {summary_path}")
    print(f"Saved best config to {best_config_path}")


def _build_search_space(tuning: dict, training_config: dict) -> list[dict[str, float | int]]:
    spatial_window_sizes = [int(value) for value in tuning.get("spatial_window_sizes", [12, 18])]
    temporal_sequence_lengths = [int(value) for value in tuning.get("temporal_sequence_lengths", [6, 12])]
    strides = [int(value) for value in tuning.get("strides", [3, 6])]
    spatial_weights = [float(value) for value in tuning.get("spatial_weights", [0.5])]
    spatial_latent_channels = [int(value) for value in tuning.get("spatial_latent_channels", [24, 32])]
    temporal_hidden_sizes = [int(value) for value in tuning.get("temporal_hidden_sizes", [48, 64])]
    spatial_dropouts = [float(value) for value in tuning.get("spatial_dropouts", [0.1])]
    temporal_dropouts = [float(value) for value in tuning.get("temporal_dropouts", [0.1])]
    spatial_learning_rates = [float(value) for value in tuning.get("spatial_learning_rates", [float(training_config.get("spatial_learning_rate", 1e-3))])]
    temporal_learning_rates = [float(value) for value in tuning.get("temporal_learning_rates", [float(training_config.get("temporal_learning_rate", 1e-3))])]
    spatial_batch_sizes = [int(value) for value in tuning.get("spatial_batch_sizes", [int(training_config.get("spatial_batch_size", 16))])]
    temporal_batch_sizes = [int(value) for value in tuning.get("temporal_batch_sizes", [int(training_config.get("temporal_batch_size", 24))])]
    spatial_epochs = [int(value) for value in tuning.get("spatial_epochs", [int(training_config.get("spatial_epochs", 3))])]
    temporal_epochs = [int(value) for value in tuning.get("temporal_epochs", [int(training_config.get("temporal_epochs", 4))])]

    candidates: list[dict[str, float | int]] = []
    for values in product(
        spatial_window_sizes,
        temporal_sequence_lengths,
        strides,
        spatial_weights,
        spatial_latent_channels,
        temporal_hidden_sizes,
        spatial_dropouts,
        temporal_dropouts,
        spatial_learning_rates,
        temporal_learning_rates,
        spatial_batch_sizes,
        temporal_batch_sizes,
        spatial_epochs,
        temporal_epochs,
    ):
        (
            spatial_window_size,
            temporal_sequence_length,
            stride,
            spatial_weight,
            spatial_latent,
            temporal_hidden,
            spatial_dropout,
            temporal_dropout,
            spatial_lr,
            temporal_lr,
            spatial_batch_size,
            temporal_batch_size,
            candidate_spatial_epochs,
            candidate_temporal_epochs,
        ) = values
        candidates.append(
            {
                "spatial_window_size": int(spatial_window_size),
                "temporal_sequence_length": int(temporal_sequence_length),
                "stride": int(stride),
                "spatial_weight": float(spatial_weight),
                "temporal_weight": float(1.0 - float(spatial_weight)),
                "spatial_latent_channels": int(spatial_latent),
                "temporal_hidden_size": int(temporal_hidden),
                "spatial_dropout": float(spatial_dropout),
                "temporal_dropout": float(temporal_dropout),
                "spatial_learning_rate": float(spatial_lr),
                "temporal_learning_rate": float(temporal_lr),
                "spatial_batch_size": int(spatial_batch_size),
                "temporal_batch_size": int(temporal_batch_size),
                "spatial_epochs": int(candidate_spatial_epochs),
                "temporal_epochs": int(candidate_temporal_epochs),
            }
        )
    return candidates


def _best_candidate_config(config: dict, best: dict[str, object]) -> dict[str, object]:
    base = dict(config)
    training = dict(base.get("training", {}))
    inference = dict(base.get("inference", {}))
    training.update(
        {
            "spatial_window_size": int(best["spatial_window_size"]),
            "temporal_sequence_length": int(best["temporal_sequence_length"]),
            "stride": int(best["stride"]),
            "spatial_epochs": int(best["spatial_epochs"]),
            "temporal_epochs": int(best["temporal_epochs"]),
            "spatial_batch_size": int(best["spatial_batch_size"]),
            "temporal_batch_size": int(best["temporal_batch_size"]),
            "spatial_learning_rate": float(best["spatial_learning_rate"]),
            "temporal_learning_rate": float(best["temporal_learning_rate"]),
            "spatial_latent_channels": int(best["spatial_latent_channels"]),
            "temporal_hidden_size": int(best["temporal_hidden_size"]),
            "spatial_dropout": float(best["spatial_dropout"]),
            "temporal_dropout": float(best["temporal_dropout"]),
        }
    )
    inference.update(
        {
            "spatial_weight": float(best["spatial_weight"]),
            "temporal_weight": float(best["temporal_weight"]),
            "calibration_percentile": float(base.get("inference", {}).get("calibration_percentile", 97.5)),
        }
    )
    base["training"] = training
    base["inference"] = inference
    base["tuning_result"] = {
        "threshold": float(best["threshold"]),
        "anomalous_f1": float(best["anomalous_f1"]),
        "anomalous_recall": float(best["anomalous_recall"]),
        "nominal_false_positive_count": int(best["nominal_false_positive_count"]),
    }
    return base


def _make_magnetic_model(config: dict) -> AnalyticMagneticModel | WMMMagneticModel:
    backend = str(config.get("magnetic_backend", "wmm")).strip().lower()
    if backend == "analytic":
        return AnalyticMagneticModel()
    return WMMMagneticModel(cache_path="data/cache/wmm_cache.json")


if __name__ == "__main__":
    main()
