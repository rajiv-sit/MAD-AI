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
from mad_ai.wmm import WMMMagneticModel


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
    raw = BatchSensorIngestor(schema_mapping=schema_mapping).load(input_dir)
    features = prepare_observed_features(raw, WMMMagneticModel(cache_path="data/cache/wmm_cache.json"))

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
    summary["input_dir"] = str(input_dir)
    summary["output_csv"] = str(output_csv)
    summary["config_path"] = str(config_path)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_csv, index=False)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved real batch scored CSV to {output_csv}")
    print(f"Saved real batch scoring summary to {summary_json}")


if __name__ == "__main__":
    main()
