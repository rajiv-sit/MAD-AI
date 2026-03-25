from __future__ import annotations

import unittest

from mad_ai.ingest import load_bahamas_mad_ascii
from mad_ai.viz import RealtimeAnomalyDashboardVisualizer
from tests.unit.helpers import workspace_temp_dir


class BahamasDashboardTestCase(unittest.TestCase):
    def test_bahamas_loader_derives_range_to_vessel(self) -> None:
        with workspace_temp_dir() as tmp_dir:
            source = tmp_dir / "mad_data.asc"
            source.write_text(
                "\n".join(
                    [
                        "55920.0 22.7 24.7000 -76.8000 3676.45 113.5 24.5000 -76.0000 0.0 -165.8",
                        "55920.03125 22.6 24.7005 -76.7995 3676.48 113.4 24.5002 -75.9998 0.0 -165.7",
                    ]
                ),
                encoding="utf-8",
            )
            frame = load_bahamas_mad_ascii(source)
            self.assertIn("range_to_vessel_m", frame.columns)
            self.assertGreater(float(frame["range_to_vessel_m"].iloc[0]), 0.0)

    def test_dashboard_visualizer_writes_png(self) -> None:
        with workspace_temp_dir() as tmp_dir:
            source = tmp_dir / "mad_data.asc"
            source.write_text(
                "\n".join(
                    [
                        "55920.0 22.7 24.7000 -76.8000 3676.45 113.5 24.5000 -76.0000 0.0 -165.8",
                        "55920.03125 22.6 24.7005 -76.7995 3676.48 113.4 24.5002 -75.9998 0.0 -165.7",
                        "55920.06250 23.2 24.7010 -76.7990 3676.51 113.3 24.5004 -75.9996 0.0 -165.6",
                    ]
                ),
                encoding="utf-8",
            )
            frame = load_bahamas_mad_ascii(source)
            frame["baseline_total_nt"] = [20.0, 20.0, 20.0]
            frame["residual_total_nt"] = frame["observed_total_nt"] - frame["baseline_total_nt"]
            frame["final_anomaly_score"] = [0.1, 0.2, 0.8]
            frame["final_anomaly_threshold"] = 0.5
            output = RealtimeAnomalyDashboardVisualizer(threshold_column="final_anomaly_threshold").render(
                frame,
                tmp_dir / "dashboard.png",
            )
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
