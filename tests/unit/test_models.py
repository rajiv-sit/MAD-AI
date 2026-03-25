from __future__ import annotations

import unittest

import numpy as np

from mad_ai.datasets import SpatialGridBuilder, TemporalSequenceBuilder
from mad_ai.features import ResidualFeatureBuilder, TemporalFeatureBuilder
from mad_ai.models.spatial import CNNAnomalyModel
from mad_ai.models.temporal import LSTMAnomalyModel
from mad_ai.utils.sample_data import make_sample_sensor_data
from mad_ai.wmm import AnalyticMagneticModel
from tests.unit.helpers import workspace_temp_dir


class ModelsTestCase(unittest.TestCase):
    def test_spatial_model_train_score_save_and_load(self) -> None:
        raw = make_sample_sensor_data(16)
        enriched = ResidualFeatureBuilder(AnalyticMagneticModel()).transform(raw)
        grid = SpatialGridBuilder(rows=4, cols=4).build(enriched)
        batch = np.stack([grid, grid], axis=0)

        model = CNNAnomalyModel(epochs=1)
        model.train(batch)
        score = model.score(grid)
        self.assertGreaterEqual(score, 0.0)

        with workspace_temp_dir() as tmp_dir:
            path = tmp_dir / "spatial.pt"
            model.save(path)
            loaded = CNNAnomalyModel(epochs=1)
            loaded.load(path)
            self.assertGreaterEqual(loaded.score(grid), 0.0)

    def test_temporal_model_train_score_save_and_load(self) -> None:
        raw = make_sample_sensor_data(16)
        enriched = ResidualFeatureBuilder(AnalyticMagneticModel()).transform(raw)
        temporal = TemporalFeatureBuilder().transform(enriched)
        sequences = TemporalSequenceBuilder(sequence_length=4).build(temporal)

        model = LSTMAnomalyModel(epochs=1)
        model.train(sequences)
        score = model.score(sequences[0])
        self.assertGreaterEqual(score, 0.0)

        with workspace_temp_dir() as tmp_dir:
            path = tmp_dir / "temporal.pt"
            model.save(path)
            loaded = LSTMAnomalyModel(epochs=1)
            loaded.load(path)
            self.assertGreaterEqual(loaded.score(sequences[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
