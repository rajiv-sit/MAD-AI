from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import pandas as pd


def _load_module():
    script_path = Path("scripts/compare_bahamas_tracking_backends.py").resolve()
    spec = importlib.util.spec_from_file_location("compare_bahamas_tracking_backends", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load compare_bahamas_tracking_backends.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BahamasBackendComparisonScriptTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module()

    def test_score_and_evaluate_returns_summary_fields(self) -> None:
        raw = pd.DataFrame(
            {
                "timestamp": ["2008-02-19T15:32:00", "2008-02-19T15:32:10", "2008-02-19T15:32:20"],
                "observed_total_nt": [22.7, 25.3, 24.1],
                "latitude_deg": [24.7, 24.7005, 24.7010],
                "longitude_deg": [-76.8, -76.7995, -76.7990],
                "altitude_m": [1120.0, 1120.1, 1120.2],
                "aircraft_heading_deg": [113.5, 113.4, 113.3],
                "aircraft_speed_mps": [92.0, 92.2, 92.1],
                "vessel_latitude_deg": [24.5, 24.4994, 24.4988],
                "vessel_longitude_deg": [-76.0, -75.9998, -75.9996],
                "vessel_heading_deg": [180.0, 180.0, 180.0],
                "vessel_speed_mps": [7.5, 7.4, 7.4],
                "range_to_vessel_m": [82000.0, 81800.0, 81600.0],
            }
        )
        summary = self.module._score_and_evaluate(raw, backend="analytic")
        self.assertEqual(summary["backend"], "analytic")
        self.assertEqual(summary["rows"], 3)
        self.assertIn("mean_tracking_error_m", summary)
        self.assertIn("runtime_seconds", summary)


if __name__ == "__main__":
    unittest.main()
