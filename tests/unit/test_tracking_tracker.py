from __future__ import annotations

import unittest

import pandas as pd

from mad_ai.tracking import ConstantVelocityMagneticTracker, MultiHypothesisMagneticTracker, observations_from_bahamas_frame


class TrackingTrackerTestCase(unittest.TestCase):
    def _sample_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "timestamp": "2008-02-19T15:32:00",
                    "latitude_deg": 24.7,
                    "longitude_deg": -76.8,
                    "altitude_m": 1120.0,
                    "aircraft_heading_deg": 113.5,
                    "aircraft_speed_mps": 92.0,
                    "observed_total_nt": 22.7,
                    "baseline_total_nt": 20.0,
                    "residual_total_nt": 2.7,
                    "final_anomaly_score": 0.2,
                    "track_id": "bahamas_flight",
                    "vessel_latitude_deg": 24.5,
                    "vessel_longitude_deg": -76.0,
                    "vessel_heading_deg": 180.0,
                    "vessel_speed_mps": 7.5,
                    "range_to_vessel_m": 82000.0,
                },
                {
                    "timestamp": "2008-02-19T15:32:10",
                    "latitude_deg": 24.7005,
                    "longitude_deg": -76.7995,
                    "altitude_m": 1120.1,
                    "aircraft_heading_deg": 113.4,
                    "aircraft_speed_mps": 92.2,
                    "observed_total_nt": 25.3,
                    "baseline_total_nt": 20.1,
                    "residual_total_nt": 5.2,
                    "final_anomaly_score": 0.9,
                    "track_id": "bahamas_flight",
                    "vessel_latitude_deg": 24.4994,
                    "vessel_longitude_deg": -75.9998,
                    "vessel_heading_deg": 180.0,
                    "vessel_speed_mps": 7.4,
                    "range_to_vessel_m": 81800.0,
                },
            ]
        )

    def test_tracker_initializes_and_steps(self) -> None:
        observations = observations_from_bahamas_frame(self._sample_frame())
        tracker = ConstantVelocityMagneticTracker(initialization_strategy="magnetic_bearing_grid")
        first = tracker.initialize(observations)
        second = tracker.step(observations[1])

        self.assertGreaterEqual(first.confidence, 0.0)
        self.assertEqual(first.vessel_state.metadata["initialization_source"], "magnetic_bearing_grid")
        self.assertIsInstance(second.innovation_nt, float)
        self.assertGreaterEqual(second.confidence, first.confidence)
        self.assertIsInstance(second.vessel_state.latitude_deg, float)
        self.assertIsInstance(second.vessel_state.longitude_deg, float)

    def test_multi_hypothesis_tracker_reports_hypothesis_metadata(self) -> None:
        observations = observations_from_bahamas_frame(self._sample_frame())
        tracker = MultiHypothesisMagneticTracker(initialization_strategy="magnetic_bearing_grid", beam_width=3)
        first = tracker.initialize(observations)
        second = tracker.step(observations[1])

        self.assertGreaterEqual(int(first.metadata["hypothesis_count"]), 1)
        self.assertGreaterEqual(int(second.metadata["hypothesis_count"]), 1)
        self.assertIn("cumulative_cost", second.metadata)
        self.assertIn("rolling_innovation_cost", second.metadata)
        self.assertIsInstance(second.innovation_nt, float)
        self.assertGreater(second.vessel_state.magnetic_moment_am2, 0.0)


if __name__ == "__main__":
    unittest.main()
