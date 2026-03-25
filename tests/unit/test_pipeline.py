from __future__ import annotations

import unittest

from mad_ai.datasets import SpatialGridBuilder, TemporalSequenceBuilder
from mad_ai.features import ResidualFeatureBuilder, TemporalFeatureBuilder
from mad_ai.inference import AnomalyFusionEngine
from mad_ai.models.spatial import CNNAnomalyModel
from mad_ai.models.temporal import LSTMAnomalyModel
from mad_ai.utils.sample_data import make_sample_sensor_data
from mad_ai.wmm import WMMMagneticModel


class PipelineTestCase(unittest.TestCase):
    def test_wmm_model_returns_expected_fields(self) -> None:
        field = WMMMagneticModel().get_field(43.7, -79.4, 0.0)
        self.assertIn("source", field)
        self.assertIn(field["source"], {"geomag", "wmm2020", "analytic-fallback"})
        self.assertIn("total_intensity_nt", field)
        self.assertIn("declination_deg", field)
        self.assertIn("inclination_deg", field)

    def test_end_to_end_shapes_and_scores(self) -> None:
        raw = make_sample_sensor_data(32)
        enriched = ResidualFeatureBuilder(WMMMagneticModel()).transform(raw)
        temporal = TemporalFeatureBuilder().transform(enriched)

        grid = SpatialGridBuilder(rows=8, cols=8).build(temporal)
        sequences = TemporalSequenceBuilder(sequence_length=6).build(temporal)

        self.assertEqual(grid.shape, (8, 8, 2))
        self.assertEqual(sequences.shape[1:], (6, 2))

        spatial_model = CNNAnomalyModel()
        spatial_model.train(grid)
        temporal_model = LSTMAnomalyModel()
        temporal_model.train(sequences)

        fusion = AnomalyFusionEngine(threshold=0.01)
        result = fusion.fuse(float(spatial_model.score(grid)), float(temporal_model.score(sequences[0])))
        self.assertGreaterEqual(result.final_score, 0.0)


if __name__ == "__main__":
    unittest.main()
