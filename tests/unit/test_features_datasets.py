from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from mad_ai.datasets import SpatialGridBuilder, TemporalSequenceBuilder
from mad_ai.features import ResidualFeatureBuilder, TemporalFeatureBuilder
from mad_ai.utils.sample_data import make_sample_sensor_data
from mad_ai.wmm import AnalyticMagneticModel


class FeaturesAndDatasetsTestCase(unittest.TestCase):
    def test_residual_feature_builder_adds_baseline_and_residual_columns(self) -> None:
        raw = make_sample_sensor_data(12)
        enriched = ResidualFeatureBuilder(AnalyticMagneticModel()).transform(raw)
        self.assertIn("baseline_total_nt", enriched.columns)
        self.assertIn("residual_total_nt", enriched.columns)
        self.assertEqual(len(enriched), 12)

    def test_temporal_feature_builder_adds_delta_and_rolling_values(self) -> None:
        raw = make_sample_sensor_data(12)
        enriched = ResidualFeatureBuilder(AnalyticMagneticModel()).transform(raw)
        temporal = TemporalFeatureBuilder().transform(enriched)
        self.assertIn("delta_residual_total_nt", temporal.columns)
        self.assertIn("rolling_residual_total_nt", temporal.columns)
        self.assertEqual(float(temporal["delta_residual_total_nt"].iloc[0]), 0.0)

    def test_spatial_grid_builder_returns_expected_shape(self) -> None:
        raw = make_sample_sensor_data(16)
        enriched = ResidualFeatureBuilder(AnalyticMagneticModel()).transform(raw)
        grid = SpatialGridBuilder(rows=4, cols=4).build(enriched)
        self.assertEqual(grid.shape, (4, 4, 2))

    def test_temporal_sequence_builder_handles_padding_and_windows(self) -> None:
        raw = make_sample_sensor_data(5)
        enriched = ResidualFeatureBuilder(AnalyticMagneticModel()).transform(raw)
        temporal = TemporalFeatureBuilder().transform(enriched)

        padded = TemporalSequenceBuilder(sequence_length=8).build(temporal)
        self.assertEqual(padded.shape, (1, 8, 2))

        windows = TemporalSequenceBuilder(sequence_length=3).build(temporal)
        self.assertEqual(windows.shape, (3, 3, 2))

    def test_residual_feature_builder_uses_batched_queries_when_available(self) -> None:
        raw = make_sample_sensor_data(6)
        model = MagicMock()
        model.get_fields.return_value = [
            {
                "total_intensity_nt": 50000.0 + idx,
                "declination_deg": -8.0,
                "inclination_deg": 58.0,
            }
            for idx in range(len(raw))
        ]

        enriched = ResidualFeatureBuilder(model).transform(raw)

        model.get_fields.assert_called_once()
        model.get_field.assert_not_called()
        self.assertEqual(len(enriched), len(raw))


if __name__ == "__main__":
    unittest.main()
