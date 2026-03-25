from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from mad_ai.datasets import SpatialGridBuilder, TemporalSequenceBuilder
from mad_ai.features import ResidualFeatureBuilder, TemporalFeatureBuilder
from mad_ai.inference import AnomalyFusionEngine, SpatialScorer, TemporalScorer
from mad_ai.ingest import CsvSensorIngestor
from mad_ai.models.spatial import CNNAnomalyModel
from mad_ai.models.temporal import LSTMAnomalyModel
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
    enriched = ResidualFeatureBuilder(WMMMagneticModel(cache_path="data/cache/wmm_cache.json")).transform(raw)
    temporal = TemporalFeatureBuilder().transform(enriched)

    grid = SpatialGridBuilder().build(temporal)
    sequences = TemporalSequenceBuilder().build(temporal)

    spatial_model = CNNAnomalyModel()
    spatial_model.train(grid)
    temporal_model = LSTMAnomalyModel()
    temporal_model.train(sequences)

    spatial_score = SpatialScorer(spatial_model).score(grid)
    temporal_scores = temporal_model.score(sequences)
    temporal["spatial_anomaly_score"] = float(spatial_score)
    temporal["temporal_anomaly_score"] = 0.0
    temporal["final_anomaly_score"] = 0.0
    temporal["is_anomaly"] = False

    engine = AnomalyFusionEngine()
    start_index = max(0, len(temporal) - len(temporal_scores))
    for idx, temporal_score in enumerate(temporal_scores):
        row_index = start_index + idx
        result = engine.fuse(float(spatial_score), float(temporal_score))
        temporal.at[row_index, "temporal_anomaly_score"] = result.temporal_score
        temporal.at[row_index, "final_anomaly_score"] = result.final_score
        temporal.at[row_index, "is_anomaly"] = result.is_anomaly

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporal.to_csv(output_csv, index=False)

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(
            {
                "input_csv": str(input_csv),
                "output_csv": str(output_csv),
                "rows": int(len(temporal)),
                "spatial_score": float(spatial_score),
                "temporal_score_mean": float(pd.Series(temporal["temporal_anomaly_score"]).mean()),
                "final_score_mean": float(pd.Series(temporal["final_anomaly_score"]).mean()),
                "anomaly_count": int(pd.Series(temporal["is_anomaly"]).sum()),
                "threshold": float(engine.threshold),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved scored observed anomaly CSV to {output_csv}")
    print(f"Saved observed anomaly summary to {summary_json}")


if __name__ == "__main__":
    main()
