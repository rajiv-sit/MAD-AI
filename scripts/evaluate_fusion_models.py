from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mad_ai.inference import prepare_observed_features, score_observed_features
from mad_ai.models.spatial import CNNAnomalyModel
from mad_ai.models.temporal import LSTMAnomalyModel
from mad_ai.utils.sample_data import make_observed_residual_datasets
from mad_ai.wmm import WMMMagneticModel


def main() -> None:
    datasets = make_observed_residual_datasets()
    magnetic_model = WMMMagneticModel(cache_path="data/cache/wmm_cache.json")
    prepared = {name: prepare_observed_features(frame, magnetic_model) for name, frame in datasets.items()}

    spatial_model = CNNAnomalyModel()
    spatial_model.load(Path("outputs/models/spatial_autoencoder.pt"))
    temporal_model = LSTMAnomalyModel()
    temporal_model.load(Path("outputs/models/temporal_autoencoder.pt"))

    spatial_threshold = float(_load_threshold(Path("outputs/calibration/spatial_threshold.json")))
    temporal_threshold = float(_load_threshold(Path("outputs/calibration/temporal_threshold.json")))

    nominal_scored, nominal_summary = score_observed_features(
        prepared["nominal_eval"],
        spatial_model=spatial_model,
        temporal_model=temporal_model,
        calibration_features=prepared["calibration"],
        spatial_scale=spatial_threshold,
        temporal_scale=temporal_threshold,
    )
    fusion_threshold = float(nominal_summary["threshold"])
    anomalous_scored, anomalous_summary = score_observed_features(
        prepared["anomalous_eval"],
        spatial_model=spatial_model,
        temporal_model=temporal_model,
        threshold=fusion_threshold,
        spatial_scale=spatial_threshold,
        temporal_scale=temporal_threshold,
    )

    calibration_path = Path("outputs/calibration/fusion_threshold.json")
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_payload = {
        "threshold": fusion_threshold,
        "spatial_weight": 0.5,
        "temporal_weight": 0.5,
        "spatial_scale": spatial_threshold,
        "temporal_scale": temporal_threshold,
        "calibration": nominal_summary.get("calibration", {}),
    }
    calibration_path.write_text(json.dumps(calibration_payload, indent=2), encoding="utf-8")

    output_dir = Path("outputs/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "fusion_model_score_comparison.csv"
    metrics_path = output_dir / "fusion_model_metrics.json"
    histogram_path = output_dir / "fusion_model_score_histogram.png"

    comparison = pd.concat(
        [
            _score_frame(nominal_scored, "nominal_eval"),
            _score_frame(anomalous_scored, "anomalous_eval"),
        ],
        ignore_index=True,
    )
    comparison.to_csv(comparison_path, index=False)

    metrics_payload = {
        "threshold": fusion_threshold,
        "spatial_scale": spatial_threshold,
        "temporal_scale": temporal_threshold,
        "nominal_eval": {
            "rows": int(len(nominal_scored)),
            "anomaly_count": int(nominal_scored["is_anomaly"].sum()),
            "mean_final_score": float(nominal_scored["final_anomaly_score"].mean()),
            "metrics": nominal_summary.get("metrics", {}),
        },
        "anomalous_eval": {
            "rows": int(len(anomalous_scored)),
            "anomaly_count": int(anomalous_scored["is_anomaly"].sum()),
            "mean_final_score": float(anomalous_scored["final_anomaly_score"].mean()),
            "metrics": anomalous_summary.get("metrics", {}),
        },
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    _save_histogram(
        nominal_scored["final_anomaly_score"],
        anomalous_scored["final_anomaly_score"],
        fusion_threshold,
        histogram_path,
    )

    print(f"Saved fusion calibration to {calibration_path}")
    print(f"Saved fusion metrics to {metrics_path}")
    print(f"Saved fusion score comparison to {comparison_path}")
    print(f"Saved fusion score histogram to {histogram_path}")


def _load_threshold(path: Path) -> float:
    return float(json.loads(path.read_text(encoding="utf-8"))["threshold"])


def _score_frame(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": split_name,
            "run_id": frame.get("run_id", ""),
            "timestamp": frame.get("timestamp", ""),
            "spatial_anomaly_score": frame["spatial_anomaly_score"],
            "temporal_anomaly_score": frame["temporal_anomaly_score"],
            "final_anomaly_score": frame["final_anomaly_score"],
            "is_injected_anomaly": frame.get("is_injected_anomaly", False),
            "is_anomaly": frame["is_anomaly"],
        }
    )


def _save_histogram(nominal_scores, anomalous_scores, threshold: float, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(nominal_scores, bins=24, alpha=0.65, label="Nominal", color="#336699")
    ax.hist(anomalous_scores, bins=24, alpha=0.65, label="Injected anomaly", color="#cc5533")
    ax.axvline(threshold, color="#111111", linestyle="--", linewidth=1.5, label="Fusion threshold")
    ax.set_title("Fused Anomaly Score Distribution")
    ax.set_xlabel("Final anomaly score")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
