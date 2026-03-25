from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mad_ai.inference import (
    ThresholdCalibrator,
    build_spatial_training_samples,
    evaluate_threshold,
    prepare_observed_features,
    score_spatial_rows,
)
from mad_ai.models.spatial import CNNAnomalyModel
from mad_ai.utils.sample_data import make_observed_residual_datasets
from mad_ai.wmm import WMMMagneticModel


def main() -> None:
    datasets = make_observed_residual_datasets()
    magnetic_model = WMMMagneticModel(cache_path="data/cache/wmm_cache.json")
    prepared = {name: prepare_observed_features(frame, magnetic_model) for name, frame in datasets.items()}

    spatial_samples = build_spatial_training_samples(prepared["train"], window_size=24, stride=6)
    model = CNNAnomalyModel(epochs=8, batch_size=16, latent_channels=24)
    model.train(spatial_samples)

    model_path = Path("outputs/models/spatial_autoencoder.pt")
    model.save(model_path)

    calibration_scores = score_spatial_rows(prepared["calibration"], model, window_size=24, stride=6)
    calibration = ThresholdCalibrator(percentile=97.5).calibrate(calibration_scores)

    nominal_scores = score_spatial_rows(prepared["nominal_eval"], model, window_size=24, stride=6)
    anomalous_scores = score_spatial_rows(prepared["anomalous_eval"], model, window_size=24, stride=6)

    nominal_metrics = evaluate_threshold(
        nominal_scores,
        prepared["nominal_eval"]["is_injected_anomaly"].astype(int).to_numpy(dtype=int),
        calibration.threshold,
    )
    anomalous_metrics = evaluate_threshold(
        anomalous_scores,
        prepared["anomalous_eval"]["is_injected_anomaly"].astype(int).to_numpy(dtype=int),
        calibration.threshold,
    )

    evaluation_dir = Path("outputs/evaluation")
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = evaluation_dir / "spatial_model_metrics.json"
    comparison_path = evaluation_dir / "spatial_model_score_comparison.csv"
    histogram_path = evaluation_dir / "spatial_model_score_histogram.png"
    threshold_path = Path("outputs/calibration/spatial_threshold.json")
    threshold_path.parent.mkdir(parents=True, exist_ok=True)

    comparison = pd.concat(
        [
            _frame_with_scores(prepared["nominal_eval"], nominal_scores, "nominal_eval"),
            _frame_with_scores(prepared["anomalous_eval"], anomalous_scores, "anomalous_eval"),
        ],
        ignore_index=True,
    )
    comparison.to_csv(comparison_path, index=False)

    threshold_path.write_text(
        json.dumps(
            {
                "threshold": calibration.threshold,
                "percentile": calibration.percentile,
                "mean_score": calibration.mean_score,
                "std_score": calibration.std_score,
                "sample_count": calibration.sample_count,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps(
            {
                "training_samples": int(len(spatial_samples)),
                "threshold": calibration.threshold,
                "calibration": {
                    "percentile": calibration.percentile,
                    "mean_score": calibration.mean_score,
                    "std_score": calibration.std_score,
                    "min_score": calibration.min_score,
                    "max_score": calibration.max_score,
                    "sample_count": calibration.sample_count,
                },
                "nominal_eval": _metrics_to_dict(nominal_metrics, len(prepared["nominal_eval"]), nominal_scores),
                "anomalous_eval": _metrics_to_dict(anomalous_metrics, len(prepared["anomalous_eval"]), anomalous_scores),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _save_histogram(
        nominal_scores=nominal_scores,
        anomalous_scores=anomalous_scores,
        threshold=calibration.threshold,
        title="Spatial Model Score Distribution",
        output_path=histogram_path,
    )

    print(f"Saved spatial model to {model_path}")
    print(f"Saved spatial threshold to {threshold_path}")
    print(f"Saved spatial metrics to {metrics_path}")
    print(f"Saved spatial score comparison to {comparison_path}")
    print(f"Saved spatial score histogram to {histogram_path}")


def _frame_with_scores(features: pd.DataFrame, scores, split_name: str) -> pd.DataFrame:
    frame = features[["run_id", "timestamp", "is_injected_anomaly"]].copy()
    frame["split"] = split_name
    frame["score"] = scores
    return frame


def _metrics_to_dict(metrics, row_count: int, scores) -> dict[str, float | int]:
    return {
        "rows": int(row_count),
        "threshold": metrics.threshold,
        "true_positives": metrics.true_positives,
        "true_negatives": metrics.true_negatives,
        "false_positives": metrics.false_positives,
        "false_negatives": metrics.false_negatives,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "accuracy": metrics.accuracy,
        "f1_score": metrics.f1_score,
        "mean_score": float(pd.Series(scores).mean()),
    }


def _save_histogram(nominal_scores, anomalous_scores, threshold: float, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(nominal_scores, bins=24, alpha=0.65, label="Nominal", color="#336699")
    ax.hist(anomalous_scores, bins=24, alpha=0.65, label="Injected anomaly", color="#cc5533")
    ax.axvline(threshold, color="#111111", linestyle="--", linewidth=1.5, label="Threshold")
    ax.set_title(title)
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
