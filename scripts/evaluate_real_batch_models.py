from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mad_ai.core.config import load_config
from mad_ai.core.manifests import load_dataset_manifest
from mad_ai.ingest import SplitBatchSensorIngestor
from mad_ai.inference import ThresholdCalibrator
from mad_ai.inference import prepare_observed_features, score_observed_features, train_observed_models
from mad_ai.models.spatial import CNNAnomalyModel
from mad_ai.models.temporal import LSTMAnomalyModel
from mad_ai.wmm import AnalyticMagneticModel, WMMMagneticModel


def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config/real_batch.yaml")
    config = load_config(config_path)
    dataset_manifest_path = config.get("dataset_manifest")
    schema_mapping = config.get("schema_mapping", {})
    split_sources = config.get("split_sources", {})
    training_config = config.get("training", {})
    inference_config = config.get("inference", {})
    source_options = config.get("source_options", {})
    if not split_sources:
        raise ValueError("real batch config must define split_sources.")
    spatial_window_size = int(training_config.get("spatial_window_size", 12))
    temporal_sequence_length = int(training_config.get("temporal_sequence_length", 6))
    stride = int(training_config.get("stride", 3))
    spatial_epochs = int(training_config.get("spatial_epochs", 8))
    temporal_epochs = int(training_config.get("temporal_epochs", 10))
    spatial_batch_size = int(training_config.get("spatial_batch_size", 16))
    temporal_batch_size = int(training_config.get("temporal_batch_size", 32))
    spatial_weight = float(inference_config.get("spatial_weight", 0.5))
    temporal_weight = float(inference_config.get("temporal_weight", 0.5))
    calibration_percentile = float(inference_config.get("calibration_percentile", 97.5))
    robustness_percentiles = [float(value) for value in inference_config.get("robustness_percentiles", [90.0, 95.0, 97.5, 99.0])]
    dataset_manifest = load_dataset_manifest(dataset_manifest_path) if dataset_manifest_path else None

    ingestor = SplitBatchSensorIngestor(
        schema_mapping=schema_mapping,
        sqlite_table_name=str(source_options.get("sqlite_table_name", "sensor_readings")),
    )
    raw_splits = ingestor.load_splits(split_sources)
    magnetic_model = _make_magnetic_model(config)
    prepared = {name: prepare_observed_features(frame, magnetic_model) for name, frame in raw_splits.items()}

    spatial_model, temporal_model, training_summary = train_observed_models(
        prepared["train"],
        spatial_model=CNNAnomalyModel(epochs=spatial_epochs, batch_size=spatial_batch_size, latent_channels=24),
        temporal_model=LSTMAnomalyModel(epochs=temporal_epochs, batch_size=temporal_batch_size, hidden_size=48),
        spatial_window_size=spatial_window_size,
        temporal_sequence_length=temporal_sequence_length,
        stride=stride,
    )

    models_dir = Path("outputs/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    spatial_model_path = models_dir / "real_batch_spatial.pt"
    temporal_model_path = models_dir / "real_batch_temporal.pt"
    spatial_model.save(spatial_model_path)
    temporal_model.save(temporal_model_path)

    nominal_scored, nominal_summary = score_observed_features(
        prepared["nominal_eval"],
        spatial_model=spatial_model,
        temporal_model=temporal_model,
        calibration_features=prepared["calibration"],
        calibration_percentile=calibration_percentile,
        spatial_window_size=spatial_window_size,
        temporal_sequence_length=temporal_sequence_length,
        stride=stride,
        spatial_weight=spatial_weight,
        temporal_weight=temporal_weight,
    )
    threshold = float(nominal_summary["threshold"])
    anomalous_scored, anomalous_summary = score_observed_features(
        prepared["anomalous_eval"],
        spatial_model=spatial_model,
        temporal_model=temporal_model,
        threshold=threshold,
        spatial_window_size=spatial_window_size,
        temporal_sequence_length=temporal_sequence_length,
        stride=stride,
        spatial_weight=spatial_weight,
        temporal_weight=temporal_weight,
    )

    baseline_threshold = ThresholdCalibrator(percentile=calibration_percentile).calibrate(
        prepared["calibration"]["residual_total_nt"].abs().to_numpy(dtype=float)
    ).threshold
    baseline_robustness = {
        f"{percentile:.1f}": ThresholdCalibrator(percentile=percentile).calibrate(
            prepared["calibration"]["residual_total_nt"].abs().to_numpy(dtype=float)
        ).threshold
        for percentile in robustness_percentiles
    }
    fusion_robustness = {
        f"{percentile:.1f}": ThresholdCalibrator(percentile=percentile).calibrate(
            nominal_scored["final_anomaly_score"].to_numpy(dtype=float)
        ).threshold
        for percentile in robustness_percentiles
    }

    calibration_path = Path("outputs/calibration/real_batch_thresholds.json")
    calibration_payload = {
        "threshold": threshold,
        "spatial_weight": spatial_weight,
        "temporal_weight": temporal_weight,
        "baseline_threshold": baseline_threshold,
        "training_summary": training_summary,
        "calibration": nominal_summary.get("calibration", {}),
        "robustness": {
            "fusion_thresholds": fusion_robustness,
            "baseline_thresholds": baseline_robustness,
        },
        "training": {
            "spatial_window_size": spatial_window_size,
            "temporal_sequence_length": temporal_sequence_length,
            "stride": stride,
            "spatial_epochs": spatial_epochs,
            "temporal_epochs": temporal_epochs,
            "spatial_batch_size": spatial_batch_size,
            "temporal_batch_size": temporal_batch_size,
        },
        "magnetic_backend": str(config.get("magnetic_backend", "wmm")).lower(),
        "config_path": str(config_path),
        "dataset_manifest_path": str(dataset_manifest_path) if dataset_manifest_path else None,
        "dataset_manifest": dataset_manifest,
    }
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path.write_text(json.dumps(calibration_payload, indent=2), encoding="utf-8")

    output_dir = Path("outputs/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "real_batch_score_comparison.csv"
    metrics_path = output_dir / "real_batch_metrics.json"
    histogram_path = output_dir / "real_batch_score_histogram.png"
    robustness_path = output_dir / "real_batch_calibration_robustness.json"

    comparison = pd.concat(
        [
            _tag_frame(nominal_scored, "nominal_eval"),
            _tag_frame(anomalous_scored, "anomalous_eval"),
        ],
        ignore_index=True,
    )
    comparison.to_csv(comparison_path, index=False)
    metrics_path.write_text(
        json.dumps(
            {
                "threshold": threshold,
                "baseline_threshold": baseline_threshold,
                "training_summary": training_summary,
                "nominal_eval": nominal_summary,
                "anomalous_eval": anomalous_summary,
                "dataset_manifest_path": str(dataset_manifest_path) if dataset_manifest_path else None,
                "dataset_manifest": dataset_manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    robustness_path.write_text(
        json.dumps(
            {
                "calibration_percentile": calibration_percentile,
                "fusion_thresholds": fusion_robustness,
                "baseline_thresholds": baseline_robustness,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _save_histogram(nominal_scored["final_anomaly_score"], anomalous_scored["final_anomaly_score"], threshold, histogram_path)

    print(f"Saved real batch spatial model to {spatial_model_path}")
    print(f"Saved real batch temporal model to {temporal_model_path}")
    print(f"Saved real batch calibration to {calibration_path}")
    print(f"Saved real batch comparison to {comparison_path}")
    print(f"Saved real batch metrics to {metrics_path}")
    print(f"Saved real batch robustness report to {robustness_path}")
    print(f"Saved real batch histogram to {histogram_path}")


def _make_magnetic_model(config: dict) -> AnalyticMagneticModel | WMMMagneticModel:
    backend = str(config.get("magnetic_backend", "wmm")).strip().lower()
    if backend == "analytic":
        return AnalyticMagneticModel()
    return WMMMagneticModel(cache_path="data/cache/wmm_cache.json")


def _tag_frame(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    tagged = frame.copy()
    tagged["split"] = split_name
    return tagged


def _save_histogram(nominal_scores, anomalous_scores, threshold: float, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(nominal_scores, bins=16, alpha=0.65, label="Nominal", color="#336699")
    ax.hist(anomalous_scores, bins=16, alpha=0.65, label="Anomalous", color="#cc5533")
    ax.axvline(threshold, color="#111111", linestyle="--", linewidth=1.5, label="Threshold")
    ax.set_title("Real Batch Final Anomaly Score Distribution")
    ax.set_xlabel("Final anomaly score")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
