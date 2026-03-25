from __future__ import annotations

from pathlib import Path
import json
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mad_ai.inference import prepare_observed_features, score_observed_features
from mad_ai.models.spatial import CNNAnomalyModel
from mad_ai.models.temporal import LSTMAnomalyModel
from mad_ai.utils.sample_data import make_observed_residual_datasets
from mad_ai.wmm import WMMMagneticModel


def main() -> None:
    calibration_path = Path("outputs/calibration/fusion_threshold.json")
    if not calibration_path.exists():
        raise SystemExit("Missing fusion calibration artifact. Run python scripts\\evaluate_fusion_models.py first.")

    calibration_payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    spatial_model = CNNAnomalyModel()
    spatial_model.load(Path("outputs/models/spatial_autoencoder.pt"))
    temporal_model = LSTMAnomalyModel()
    temporal_model.load(Path("outputs/models/temporal_autoencoder.pt"))

    datasets = make_observed_residual_datasets()
    magnetic_model = WMMMagneticModel(cache_path="data/cache/wmm_cache.json")
    prepared = {name: prepare_observed_features(frame, magnetic_model) for name, frame in datasets.items()}

    nominal_scored, nominal_summary = score_observed_features(
        prepared["nominal_eval"],
        spatial_model=spatial_model,
        temporal_model=temporal_model,
        threshold=float(calibration_payload["threshold"]),
        spatial_weight=float(calibration_payload.get("spatial_weight", 0.5)),
        temporal_weight=float(calibration_payload.get("temporal_weight", 0.5)),
        spatial_scale=float(calibration_payload.get("spatial_scale", 1.0)),
        temporal_scale=float(calibration_payload.get("temporal_scale", 1.0)),
    )
    anomalous_scored, anomalous_summary = score_observed_features(
        prepared["anomalous_eval"],
        spatial_model=spatial_model,
        temporal_model=temporal_model,
        threshold=float(calibration_payload["threshold"]),
        spatial_weight=float(calibration_payload.get("spatial_weight", 0.5)),
        temporal_weight=float(calibration_payload.get("temporal_weight", 0.5)),
        spatial_scale=float(calibration_payload.get("spatial_scale", 1.0)),
        temporal_scale=float(calibration_payload.get("temporal_scale", 1.0)),
    )

    output_dir = Path("outputs/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / "score_comparison.csv"
    metrics_path = output_dir / "metrics.json"

    comparison = pd.concat(
        [
            _frame_with_scores(nominal_scored, "nominal_eval"),
            _frame_with_scores(anomalous_scored, "anomalous_eval"),
        ],
        ignore_index=True,
    )
    comparison.to_csv(score_path, index=False)

    metrics_path.write_text(
        json.dumps(
            {
                "threshold": calibration_payload["threshold"],
                "spatial_scale": calibration_payload.get("spatial_scale", 1.0),
                "temporal_scale": calibration_payload.get("temporal_scale", 1.0),
                "nominal_eval": nominal_summary.get("metrics", {}),
                "anomalous_eval": anomalous_summary.get("metrics", {}),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved evaluation scores to {score_path}")
    print(f"Saved evaluation metrics to {metrics_path}")


def _frame_with_scores(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
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


if __name__ == "__main__":
    main()
