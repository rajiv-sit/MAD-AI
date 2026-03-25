from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mad_ai.inference import prepare_observed_features, score_observed_features
from mad_ai.models.spatial import CNNAnomalyModel
from mad_ai.models.temporal import LSTMAnomalyModel
from mad_ai.ingest import CsvSensorIngestor
from mad_ai.wmm import WMMMagneticModel


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python scripts\\score_observed_csv_anomalies.py <input_csv> [output_csv] [summary_json]"
        )

    input_csv = Path(sys.argv[1])
    output_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/processed/observed_scored/observed_anomaly_scored.csv")
    summary_json = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("outputs/evaluation/observed_anomaly_summary.json")

    raw = CsvSensorIngestor().load(input_csv)
    features = prepare_observed_features(raw, WMMMagneticModel(cache_path="data/cache/wmm_cache.json"))

    spatial_model = _load_spatial_model(Path("outputs/models/observed_residual_spatial.pt"))
    temporal_model = _load_temporal_model(Path("outputs/models/observed_residual_temporal.pt"))
    calibration_payload = _load_calibration_payload(Path("outputs/calibration/observed_thresholds.json"))

    scored, summary = score_observed_features(
        features,
        spatial_model=spatial_model,
        temporal_model=temporal_model,
        threshold=float(calibration_payload["threshold"]),
        spatial_weight=float(calibration_payload.get("spatial_weight", 0.5)),
        temporal_weight=float(calibration_payload.get("temporal_weight", 0.5)),
    )
    summary["input_csv"] = str(input_csv)
    summary["output_csv"] = str(output_csv)
    summary["model_source"] = "pretrained-observed-residual-models"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_csv, index=False)

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved scored observed anomaly CSV to {output_csv}")
    print(f"Saved observed anomaly summary to {summary_json}")


def _load_spatial_model(path: Path) -> CNNAnomalyModel:
    model = CNNAnomalyModel()
    model.load(path)
    return model


def _load_temporal_model(path: Path) -> LSTMAnomalyModel:
    model = LSTMAnomalyModel()
    model.load(path)
    return model


def _load_calibration_payload(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(
            f"Observed calibration file not found: {path}. Run scripts\\evaluate_observed_residual_models.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
