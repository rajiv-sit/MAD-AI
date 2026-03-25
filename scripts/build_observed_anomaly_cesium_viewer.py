from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mad_ai.datasets import SpatialGridBuilder, TemporalSequenceBuilder
from mad_ai.features import ResidualFeatureBuilder, TemporalFeatureBuilder
from mad_ai.inference import AnomalyFusionEngine, SpatialScorer, TemporalScorer
from mad_ai.ingest import CsvSensorIngestor
from mad_ai.models.spatial import CNNAnomalyModel
from mad_ai.models.temporal import LSTMAnomalyModel
from mad_ai.visualizer import CesiumGlobeViewerBuilder
from mad_ai.wmm import WMMMagneticModel


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts\\build_observed_anomaly_cesium_viewer.py <input_csv> [output_html]")

    input_csv = Path(sys.argv[1])
    output_html = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs/viewer/cesium_observed_anomaly_globe.html")

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

    viewer = CesiumGlobeViewerBuilder(
        title="MAD-AI Observed Anomaly Globe",
        component_columns={
            "Observed Total": "observed_total_nt",
            "Baseline Total": "baseline_total_nt",
            "Residual Total": "residual_total_nt",
            "Temporal Anomaly Score": "temporal_anomaly_score",
            "Final Anomaly Score": "final_anomaly_score",
        },
        default_component="final_anomaly_score",
        anomaly_flag_column="is_anomaly",
        anomaly_score_column="final_anomaly_score",
    )
    output = viewer.build(temporal, output_html)
    print(f"Saved observed anomaly Cesium globe viewer to {output}")


if __name__ == "__main__":
    main()
