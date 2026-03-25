from __future__ import annotations

import unittest
from datetime import datetime

from mad_ai.utils.sample_data import make_global_wmm_grid
from mad_ai.wmm import AnalyticMagneticModel


class _StubGlobalModel(AnalyticMagneticModel):
    pass


class GlobalGridTestCase(unittest.TestCase):
    def test_make_global_wmm_grid_returns_world_sample_frame(self) -> None:
        grid = make_global_wmm_grid(
            lat_step_deg=90.0,
            lon_step_deg=180.0,
            altitude_m=0.0,
            timestamp=datetime(2026, 3, 24),
            magnetic_model=_StubGlobalModel(),
        )
        self.assertEqual(len(grid), 9)
        self.assertIn("baseline_total_nt", grid.columns)
        self.assertIn("baseline_declination_deg", grid.columns)
        self.assertIn("source", grid.columns)


if __name__ == "__main__":
    unittest.main()
