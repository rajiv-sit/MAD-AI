from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mad_ai.core.config import load_config
from mad_ai.ingest import BatchSensorIngestor
from mad_ai.inference import prepare_observed_features, score_observed_features
from mad_ai.models.spatial import CNNAnomalyModel
from mad_ai.models.temporal import LSTMAnomalyModel
from mad_ai.wmm import AnalyticMagneticModel, WMMMagneticModel


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python scripts\\score_real_batch_folder.py <input_dir> [config_path] [output_csv] [summary_json]"
        )

    input_dir = Path(sys.argv[1])
    config_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("config/real_batch.yaml")
    output_csv = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("data/processed/real_batch_scored/real_batch_scored.csv")
    summary_json = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("outputs/evaluation/real_batch_scored_summary.json")

    config = load_config(config_path)
    schema_mapping = config.get("schema_mapping", {})
    source_options = config.get("source_options", {})
    raw = BatchSensorIngestor(
        schema_mapping=schema_mapping,
        sqlite_table_name=str(source_options.get("sqlite_table_name", "sensor_readings")),
    ).load(input_dir)
    features = prepare_observed_features(raw, _make_magnetic_model(config))

    spatial_model = CNNAnomalyModel()
    spatial_model.load(Path("outputs/models/real_batch_spatial.pt"))
    temporal_model = LSTMAnomalyModel()
    temporal_model.load(Path("outputs/models/real_batch_temporal.pt"))
    calibration_payload = json.loads(Path("outputs/calibration/real_batch_thresholds.json").read_text(encoding="utf-8"))

    scored, summary = score_observed_features(
        features,
        spatial_model=spatial_model,
        temporal_model=temporal_model,
        threshold=float(calibration_payload["threshold"]),
        spatial_weight=float(calibration_payload.get("spatial_weight", 0.5)),
        temporal_weight=float(calibration_payload.get("temporal_weight", 0.5)),
    )
    baseline_threshold = float(calibration_payload.get("baseline_threshold", 1.0))
    scored["baseline_residual_score"] = scored["residual_total_nt"].abs() / max(baseline_threshold, 1e-6)
    scored["baseline_is_anomaly"] = scored["baseline_residual_score"] >= 1.0
    summary["input_dir"] = str(input_dir)
    summary["output_csv"] = str(output_csv)
    summary["config_path"] = str(config_path)
    summary["baseline_threshold"] = baseline_threshold
    summary["baseline_anomaly_count"] = int(scored["baseline_is_anomaly"].sum())

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_csv, index=False)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved real batch scored CSV to {output_csv}")
    print(f"Saved real batch scoring summary to {summary_json}")


def _make_magnetic_model(config: dict) -> AnalyticMagneticModel | WMMMagneticModel:
    backend = str(config.get("magnetic_backend", "wmm")).strip().lower()
    if backend == "analytic":
        return AnalyticMagneticModel()
    return WMMMagneticModel(cache_path="data/cache/wmm_cache.json")


if __name__ == "__main__":
    main()
