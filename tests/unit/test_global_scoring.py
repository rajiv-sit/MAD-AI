from __future__ import annotations

import unittest
from datetime import datetime

from mad_ai.inference import score_global_dataset
from mad_ai.utils.sample_data import make_multi_altitude_global_wmm_grid
from mad_ai.wmm import AnalyticMagneticModel


class GlobalScoringTestCase(unittest.TestCase):
    def test_score_global_dataset_adds_anomaly_columns(self) -> None:
        data = make_multi_altitude_global_wmm_grid(
            altitudes_m=[0.0, 1000.0],
            timestamp=datetime(2026, 3, 24),
            lat_step_deg=90.0,
            lon_step_deg=180.0,
            magnetic_model=AnalyticMagneticModel(),
        )

        scored, summary = score_global_dataset(data)

        self.assertEqual(len(scored), len(data))
        self.assertIn("spatial_anomaly_score", scored.columns)
        self.assertIn("temporal_anomaly_score", scored.columns)
        self.assertIn("final_anomaly_score", scored.columns)
        self.assertIn("is_anomaly", scored.columns)
        self.assertIn("calibration", summary)
        self.assertEqual(summary["patch_radius"], 1)


if __name__ == "__main__":
    unittest.main()
