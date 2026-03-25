from __future__ import annotations

import unittest

import pandas as pd

from mad_ai.tracking import (
    DipoleMagneticForwardModel,
    SensorState,
    VesselState,
    initialize_vessel_state_candidates,
    observations_from_bahamas_frame,
    propagate_constant_velocity,
)


class TrackingInitializationTestCase(unittest.TestCase):
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
                    "vessel_heading_deg": -165.8,
                    "vessel_speed_mps": 7.5,
                    "range_to_vessel_m": 82000.0,
                },
                {
                    "timestamp": "2008-02-19T15:32:01",
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
                    "vessel_latitude_deg": 24.5002,
                    "vessel_longitude_deg": -75.9998,
                    "vessel_heading_deg": -165.7,
                    "vessel_speed_mps": 7.4,
                    "range_to_vessel_m": 81800.0,
                },
            ]
        )

    def test_observations_from_bahamas_frame_carries_reference_metadata(self) -> None:
        observations = observations_from_bahamas_frame(self._sample_frame())
        self.assertEqual(len(observations), 2)
        self.assertAlmostEqual(float(observations[0].metadata["reference_vessel_latitude_deg"]), 24.5)
        self.assertEqual(observations[0].track_id, "bahamas_flight")

    def test_initialize_vessel_state_candidates_prefers_top_anomaly_peaks(self) -> None:
        candidates = initialize_vessel_state_candidates(
            observations_from_bahamas_frame(self._sample_frame()),
            top_k=1,
            strategy="reference_peaks",
        )
        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(candidates[0].latitude_deg, 24.5002)
        self.assertEqual(candidates[0].metadata["initialization_source"], "anomaly_peak_reference_seed")

    def test_initialize_vessel_state_candidates_can_use_magnetic_bearing_grid(self) -> None:
        candidates = initialize_vessel_state_candidates(
            observations_from_bahamas_frame(self._sample_frame()),
            top_k=2,
            strategy="magnetic_bearing_grid",
        )
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0].metadata["initialization_source"], "magnetic_bearing_grid")

    def test_dipole_forward_model_decay_with_range(self) -> None:
        model = DipoleMagneticForwardModel()
        sensor = SensorState(latitude_deg=24.7, longitude_deg=-76.8, altitude_m=1000.0)
        near = VesselState(latitude_deg=24.69, longitude_deg=-76.79, magnetic_moment_am2=1.0e6)
        far = VesselState(latitude_deg=24.2, longitude_deg=-76.2, magnetic_moment_am2=1.0e6)
        self.assertGreater(model.predict_residual_nt(sensor, near), model.predict_residual_nt(sensor, far))

    def test_propagate_constant_velocity_moves_state(self) -> None:
        state = VesselState(latitude_deg=24.5, longitude_deg=-76.0, speed_mps=10.0, heading_deg=90.0)
        advanced = propagate_constant_velocity(state, dt_seconds=60.0)
        self.assertAlmostEqual(advanced.latitude_deg, state.latitude_deg, places=3)
        self.assertGreater(advanced.longitude_deg, state.longitude_deg)


if __name__ == "__main__":
    unittest.main()
