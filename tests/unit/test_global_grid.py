from __future__ import annotations

import unittest
from datetime import datetime

from mad_ai.utils.sample_data import make_global_wmm_grid
from mad_ai.wmm import AnalyticMagneticModel


class _StubGlobalModel(AnalyticMagneticModel):
    def __init__(self) -> None:
        self.batch_calls = 0

    def get_fields(self, queries):
        self.batch_calls += 1
        return [self.get_field(lat, lon, alt, timestamp) for lat, lon, alt, timestamp in queries]


class GlobalGridTestCase(unittest.TestCase):
    def test_make_global_wmm_grid_returns_world_sample_frame(self) -> None:
        model = _StubGlobalModel()
        grid = make_global_wmm_grid(
            lat_step_deg=90.0,
            lon_step_deg=180.0,
            altitude_m=0.0,
            timestamp=datetime(2026, 3, 24),
            magnetic_model=model,
        )
        self.assertEqual(len(grid), 9)
        self.assertIn("baseline_total_nt", grid.columns)
        self.assertIn("baseline_declination_deg", grid.columns)
        self.assertIn("source", grid.columns)
        self.assertEqual(model.batch_calls, 1)


if __name__ == "__main__":
    unittest.main()
