from __future__ import annotations

import unittest

from mad_ai.inference import (
    build_spatial_training_samples,
    build_temporal_training_samples,
    prepare_observed_features,
    score_observed_features,
    score_spatial_rows,
    score_temporal_rows,
    train_observed_models,
)
from mad_ai.utils.sample_data import make_observed_residual_datasets
from mad_ai.wmm import AnalyticMagneticModel


class ObservedScoringTestCase(unittest.TestCase):
    def test_observed_training_and_scoring_pipeline_returns_threshold_and_metrics(self) -> None:
        datasets = make_observed_residual_datasets(
            train_runs=3,
            calibration_runs=2,
            nominal_eval_runs=2,
            anomalous_eval_runs=2,
            rows_per_run=24,
        )
        prepared = {name: prepare_observed_features(frame, AnalyticMagneticModel()) for name, frame in datasets.items()}

        spatial_model, temporal_model, training_summary = train_observed_models(
            prepared["train"],
            spatial_window_size=12,
            temporal_sequence_length=6,
            stride=3,
        )
        nominal_scored, nominal_summary = score_observed_features(
            prepared["nominal_eval"],
            spatial_model=spatial_model,
            temporal_model=temporal_model,
            calibration_features=prepared["calibration"],
            spatial_window_size=12,
            temporal_sequence_length=6,
            stride=3,
        )
        anomalous_scored, anomalous_summary = score_observed_features(
            prepared["anomalous_eval"],
            spatial_model=spatial_model,
            temporal_model=temporal_model,
            threshold=float(nominal_summary["threshold"]),
            spatial_window_size=12,
            temporal_sequence_length=6,
            stride=3,
        )

        self.assertGreater(training_summary["spatial_sample_count"], 0)
        self.assertGreater(training_summary["temporal_sample_count"], 0)
        self.assertIn("calibration", nominal_summary)
        self.assertIn("metrics", anomalous_summary)
        self.assertIn("final_anomaly_score", nominal_scored.columns)
        self.assertIn("final_anomaly_score", anomalous_scored.columns)
        self.assertEqual(len(anomalous_scored), len(prepared["anomalous_eval"]))

    def test_public_sample_builders_and_row_scorers_return_aligned_outputs(self) -> None:
        datasets = make_observed_residual_datasets(
            train_runs=2,
            calibration_runs=1,
            nominal_eval_runs=1,
            anomalous_eval_runs=1,
            rows_per_run=24,
        )
        prepared = {name: prepare_observed_features(frame, AnalyticMagneticModel()) for name, frame in datasets.items()}
        spatial_samples = build_spatial_training_samples(prepared["train"], window_size=12, stride=3)
        temporal_samples = build_temporal_training_samples(prepared["train"], sequence_length=6, stride=3)
        spatial_model, temporal_model, _ = train_observed_models(
            prepared["train"],
            spatial_window_size=12,
            temporal_sequence_length=6,
            stride=3,
        )

        spatial_scores = score_spatial_rows(prepared["nominal_eval"], spatial_model, window_size=12, stride=3)
        temporal_scores = score_temporal_rows(prepared["nominal_eval"], temporal_model, sequence_length=6, stride=3)

        self.assertGreater(len(spatial_samples), 0)
        self.assertGreater(len(temporal_samples), 0)
        self.assertEqual(len(spatial_scores), len(prepared["nominal_eval"]))
        self.assertEqual(len(temporal_scores), len(prepared["nominal_eval"]))


if __name__ == "__main__":
    unittest.main()
