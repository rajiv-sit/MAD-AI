from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mad_ai.inference import prepare_observed_features, score_observed_features, train_observed_models
from mad_ai.utils.sample_data import make_observed_residual_datasets
from mad_ai.wmm import WMMMagneticModel


def main() -> None:
    datasets = make_observed_residual_datasets()
    magnetic_model = WMMMagneticModel(cache_path="data/cache/wmm_cache.json")

    prepared = {name: prepare_observed_features(frame, magnetic_model) for name, frame in datasets.items()}

    spatial_model, temporal_model, training_summary = train_observed_models(prepared["train"])

    models_dir = Path("outputs/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    spatial_model_path = models_dir / "observed_residual_spatial.pt"
    temporal_model_path = models_dir / "observed_residual_temporal.pt"
    spatial_model.save(spatial_model_path)
    temporal_model.save(temporal_model_path)

    nominal_scored, nominal_summary = score_observed_features(
        prepared["nominal_eval"],
        spatial_model=spatial_model,
        temporal_model=temporal_model,
        calibration_features=prepared["calibration"],
    )

    threshold = float(nominal_summary["threshold"])
    anomalous_scored, anomalous_summary = score_observed_features(
        prepared["anomalous_eval"],
        spatial_model=spatial_model,
        temporal_model=temporal_model,
        threshold=threshold,
    )

    calibration_payload = {
        "threshold": threshold,
        "spatial_weight": 0.5,
        "temporal_weight": 0.5,
        "calibration": nominal_summary.get("calibration", {}),
        "training_summary": training_summary,
    }
    calibration_path = Path("outputs/calibration/observed_thresholds.json")
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path.write_text(json.dumps(calibration_payload, indent=2), encoding="utf-8")

    comparison_rows: list[dict[str, object]] = []
    for split_name, frame in (("nominal_eval", nominal_scored), ("anomalous_eval", anomalous_scored)):
        for row in frame.itertuples(index=False):
            comparison_rows.append(
                {
                    "split": split_name,
                    "run_id": getattr(row, "run_id", ""),
                    "timestamp": str(getattr(row, "timestamp", "")),
                    "final_anomaly_score": float(row.final_anomaly_score),
                    "spatial_anomaly_score": float(row.spatial_anomaly_score),
                    "temporal_anomaly_score": float(row.temporal_anomaly_score),
                    "is_injected_anomaly": bool(getattr(row, "is_injected_anomaly", False)),
                    "is_anomaly": bool(row.is_anomaly),
                }
            )
    comparison = pd.DataFrame(comparison_rows)

    output_dir = Path("outputs/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "observed_residual_score_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    metrics_payload = {
        "threshold": threshold,
        "training_summary": training_summary,
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
    metrics_path = output_dir / "observed_residual_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    plot_path = output_dir / "observed_residual_score_histogram.png"
    _save_score_plot(nominal_scored, anomalous_scored, threshold, plot_path)

    print(f"Saved observed residual spatial model to {spatial_model_path}")
    print(f"Saved observed residual temporal model to {temporal_model_path}")
    print(f"Saved observed calibration to {calibration_path}")
    print(f"Saved observed evaluation comparison to {comparison_path}")
    print(f"Saved observed evaluation metrics to {metrics_path}")
    print(f"Saved observed score histogram to {plot_path}")


def _save_score_plot(
    nominal_scored: pd.DataFrame,
    anomalous_scored: pd.DataFrame,
    threshold: float,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(
        nominal_scored["final_anomaly_score"],
        bins=24,
        alpha=0.65,
        label="Nominal",
        color="#336699",
    )
    ax.hist(
        anomalous_scored["final_anomaly_score"],
        bins=24,
        alpha=0.65,
        label="Injected anomaly",
        color="#cc5533",
    )
    ax.axvline(threshold, color="#111111", linestyle="--", linewidth=1.5, label="Threshold")
    ax.set_title("Observed Residual Anomaly Score Distribution")
    ax.set_xlabel("Final anomaly score")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
