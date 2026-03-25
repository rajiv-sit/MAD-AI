from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mad_ai.inference import prepare_observed_features, score_observed_features
from mad_ai.models.spatial import CNNAnomalyModel
from mad_ai.models.temporal import LSTMAnomalyModel
from mad_ai.utils.sample_data import make_sample_sensor_data
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

    raw = make_sample_sensor_data()
    features = prepare_observed_features(raw, WMMMagneticModel(cache_path="data/cache/wmm_cache.json"))
    scored, summary = score_observed_features(
        features,
        spatial_model=spatial_model,
        temporal_model=temporal_model,
        threshold=float(calibration_payload["threshold"]),
        spatial_weight=float(calibration_payload.get("spatial_weight", 0.5)),
        temporal_weight=float(calibration_payload.get("temporal_weight", 0.5)),
        spatial_scale=float(calibration_payload.get("spatial_scale", 1.0)),
        temporal_scale=float(calibration_payload.get("temporal_scale", 1.0)),
    )

    result = {
        "rows": int(len(scored)),
        "threshold": summary["threshold"],
        "spatial_mean": summary["spatial_mean"],
        "temporal_mean": summary["temporal_mean"],
        "final_mean": summary["final_mean"],
        "anomaly_count": summary["anomaly_count"],
    }
    print(result)


if __name__ == "__main__":
    main()
