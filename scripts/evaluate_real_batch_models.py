from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mad_ai.core.config import load_config
from mad_ai.ingest import SplitBatchSensorIngestor
from mad_ai.inference import prepare_observed_features, score_observed_features, train_observed_models
from mad_ai.wmm import WMMMagneticModel


def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config/real_batch.yaml")
    config = load_config(config_path)
    schema_mapping = config.get("schema_mapping", {})
    split_sources = config.get("split_sources", {})
    if not split_sources:
        raise ValueError("real batch config must define split_sources.")

    ingestor = SplitBatchSensorIngestor(schema_mapping=schema_mapping)
    raw_splits = ingestor.load_splits(split_sources)
    magnetic_model = WMMMagneticModel(cache_path="data/cache/wmm_cache.json")
    prepared = {name: prepare_observed_features(frame, magnetic_model) for name, frame in raw_splits.items()}

    spatial_model, temporal_model, training_summary = train_observed_models(
        prepared["train"],
        spatial_window_size=12,
        temporal_sequence_length=6,
        stride=3,
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
        spatial_window_size=12,
        temporal_sequence_length=6,
        stride=3,
    )
    threshold = float(nominal_summary["threshold"])
    anomalous_scored, anomalous_summary = score_observed_features(
        prepared["anomalous_eval"],
        spatial_model=spatial_model,
        temporal_model=temporal_model,
        threshold=threshold,
        spatial_window_size=12,
        temporal_sequence_length=6,
        stride=3,
    )

    calibration_path = Path("outputs/calibration/real_batch_thresholds.json")
    calibration_payload = {
        "threshold": threshold,
        "spatial_weight": 0.5,
        "temporal_weight": 0.5,
        "training_summary": training_summary,
        "calibration": nominal_summary.get("calibration", {}),
        "config_path": str(config_path),
    }
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path.write_text(json.dumps(calibration_payload, indent=2), encoding="utf-8")

    output_dir = Path("outputs/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "real_batch_score_comparison.csv"
    metrics_path = output_dir / "real_batch_metrics.json"
    histogram_path = output_dir / "real_batch_score_histogram.png"

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
                "training_summary": training_summary,
                "nominal_eval": nominal_summary,
                "anomalous_eval": anomalous_summary,
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
    print(f"Saved real batch histogram to {histogram_path}")


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
