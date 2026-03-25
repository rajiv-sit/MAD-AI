from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mad_ai.datasets import SpatialGridBuilder, TemporalSequenceBuilder
from mad_ai.features import ResidualFeatureBuilder, TemporalFeatureBuilder
from mad_ai.inference import SpatialScorer, TemporalScorer, ThresholdCalibrator, summarize_scores
from mad_ai.models.spatial import CNNAnomalyModel
from mad_ai.models.temporal import LSTMAnomalyModel
from mad_ai.utils import ArtifactStore
from mad_ai.utils.sample_data import make_sample_sensor_data
from mad_ai.wmm import WMMMagneticModel


def main() -> None:
    store = ArtifactStore("outputs/calibration")
    nominal = make_sample_sensor_data(anomaly_magnitude=0.0)
    temporal = TemporalFeatureBuilder().transform(ResidualFeatureBuilder(WMMMagneticModel()).transform(nominal))

    grid = SpatialGridBuilder().build(temporal)
    sequences = TemporalSequenceBuilder().build(temporal)
    spatial_batch = np.stack([grid, grid, grid], axis=0)

    spatial_model = CNNAnomalyModel()
    spatial_model.train(spatial_batch)
    temporal_model = LSTMAnomalyModel()
    temporal_model.train(sequences)

    spatial_scores = np.asarray(spatial_model.score(spatial_batch), dtype=float)
    temporal_scores = np.asarray(temporal_model.score(sequences), dtype=float)
    combined_scores = 0.5 * spatial_scores.mean() + 0.5 * temporal_scores

    calibrator = ThresholdCalibrator(percentile=95.0)
    summary = calibrator.calibrate(combined_scores)

    payload = {
        "threshold": summary.threshold,
        "percentile": summary.percentile,
        "spatial_weight": 0.5,
        "temporal_weight": 0.5,
        "combined_summary": summarize_scores(combined_scores),
        "spatial_summary": summarize_scores(spatial_scores),
        "temporal_summary": summarize_scores(temporal_scores),
    }
    path = store.save_json("thresholds", payload)
    print(f"Saved calibration artifact to {path}")


if __name__ == "__main__":
    main()
