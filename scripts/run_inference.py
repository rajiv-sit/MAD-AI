from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mad_ai.datasets import SpatialGridBuilder, TemporalSequenceBuilder
from mad_ai.features import ResidualFeatureBuilder, TemporalFeatureBuilder
from mad_ai.inference import AnomalyFusionEngine, SpatialScorer, TemporalScorer
from mad_ai.models.spatial import CNNAnomalyModel
from mad_ai.models.temporal import LSTMAnomalyModel
from mad_ai.utils.sample_data import make_sample_sensor_data
from mad_ai.wmm import WMMMagneticModel


def main() -> None:
    raw = make_sample_sensor_data()
    enriched = ResidualFeatureBuilder(WMMMagneticModel()).transform(raw)
    temporal = TemporalFeatureBuilder().transform(enriched)

    grid = SpatialGridBuilder().build(temporal)
    sequences = TemporalSequenceBuilder().build(temporal)

    spatial_model = CNNAnomalyModel()
    spatial_model.train(grid)
    temporal_model = LSTMAnomalyModel()
    temporal_model.train(sequences)

    spatial_score = SpatialScorer(spatial_model).score(grid)
    temporal_score = TemporalScorer(temporal_model).score(sequences[0])

    calibration_path = Path("outputs/calibration/thresholds.json")
    engine = AnomalyFusionEngine.from_json(calibration_path) if calibration_path.exists() else AnomalyFusionEngine()
    result = engine.fuse_to_dict(spatial_score, temporal_score)
    print(result)


if __name__ == "__main__":
    main()
