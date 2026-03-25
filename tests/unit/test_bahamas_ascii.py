from __future__ import annotations

import unittest

import pandas as pd

from mad_ai.ingest import load_bahamas_mad_ascii
from mad_ai.inference.bahamas_realtime import score_bahamas_realtime
from mad_ai.wmm import AnalyticMagneticModel
from scripts.build_bahamas_realtime_cesium_viewer import _prepare_viewer_frame
from tests.unit.helpers import workspace_temp_dir


class BahamasAsciiTestCase(unittest.TestCase):
    def test_bahamas_ascii_loader_maps_columns_and_timestamps(self) -> None:
        with workspace_temp_dir() as tmp_dir:
            source = tmp_dir / "mad_data.asc"
            source.write_text(
                "\n".join(
                    [
                        "55920.0 22.7 24.7 -76.8 3676.45 113.5 24.5 -76.0 0.0 -165.8",
                        "55920.03125 22.6 24.7 -76.8 3676.48 113.4 24.5 -76.0 0.0 -165.7",
                    ]
                ),
                encoding="utf-8",
            )
            frame = load_bahamas_mad_ascii(source)
            self.assertIn("latitude_deg", frame.columns)
            self.assertIn("vessel_latitude_deg", frame.columns)
            self.assertIn("observed_total_nt", frame.columns)
            self.assertEqual(frame["track_id"].iloc[0], "bahamas_flight")
            self.assertEqual(str(frame["timestamp"].iloc[0]), "2008-02-19 15:32:00")

    def test_bahamas_realtime_scoring_returns_expected_columns(self) -> None:
        with workspace_temp_dir() as tmp_dir:
            source = tmp_dir / "Feb19_2008_time_TF_ACnav_shipnav.asc"
            lines = []
            for index in range(40):
                lines.append(
                    f"{55920.0 + (index / 32.0):.5f} {22.0 + (0.05 if index > 30 else 0.0):.6f} "
                    f"{24.70 + (index * 0.0001):.6f} {-76.80 + (index * 0.0001):.6f} 3676.45 113.5 "
                    f"{24.50 + (index * 0.00005):.6f} {-76.00 + (index * 0.00005):.6f} 0.0 -165.8"
                )
            source.write_text("\n".join(lines), encoding="utf-8")
            raw = load_bahamas_mad_ascii(source)
            scored, summary = score_bahamas_realtime(
                raw,
                magnetic_model=AnalyticMagneticModel(),
                warmup_fraction=0.4,
                calibration_fraction_of_warmup=0.25,
                spatial_window_size=8,
                temporal_sequence_length=4,
                stride=2,
            )
            self.assertIn("baseline_residual_score", scored.columns)
            self.assertIn("baseline_is_anomaly", scored.columns)
            self.assertIn("final_anomaly_score", scored.columns)
            self.assertIn("realtime_phase", scored.columns)
            self.assertIn("reference_strategy", summary)
            self.assertEqual(len(scored), len(raw))

    def test_bahamas_realtime_prefers_stable_reference_window(self) -> None:
        with workspace_temp_dir() as tmp_dir:
            source = tmp_dir / "mad_data.asc"
            lines = []
            for index in range(120):
                measurement = 24.0 if index < 40 else (21.5 if index < 80 else 35.0)
                lines.append(
                    f"{55920.0 + (index / 32.0):.5f} {measurement:.6f} "
                    f"{24.70 + (index * 0.0001):.6f} {-76.80 + (index * 0.0001):.6f} 3676.45 113.5 "
                    f"{24.50 + (index * 0.00005):.6f} {-76.00 + (index * 0.00005):.6f} 0.0 -165.8"
                )
            source.write_text("\n".join(lines), encoding="utf-8")
            raw = load_bahamas_mad_ascii(source)
            _, summary = score_bahamas_realtime(
                raw,
                magnetic_model=AnalyticMagneticModel(),
                warmup_fraction=0.25,
                reference_window_mode="stable_window",
                spatial_window_size=8,
                temporal_sequence_length=4,
                stride=2,
            )
            reference = summary["reference_strategy"]
            self.assertEqual(reference["mode"], "stable_window")
            self.assertGreaterEqual(reference["reference_start_row"], 0)
            self.assertGreater(reference["reference_end_row"], reference["reference_start_row"])

    def test_viewer_preparation_preserves_anomalies_and_limits_altitude_layers(self) -> None:
        raw = {
            "latitude_deg": [24.0 + idx * 0.001 for idx in range(100)],
            "longitude_deg": [-76.0 + idx * 0.001 for idx in range(100)],
            "altitude_m": [1000.0 + idx for idx in range(100)],
            "timestamp": [f"2008-02-19 15:32:{idx:02d}" for idx in range(100)],
            "observed_total_nt": [10.0] * 100,
            "baseline_total_nt": [9.0] * 100,
            "residual_total_nt": [1.0] * 100,
            "baseline_residual_score": [1.0] * 100,
            "spatial_anomaly_score": [0.1] * 100,
            "temporal_anomaly_score": [0.2] * 100,
            "final_anomaly_score": [float(idx) / 10.0 for idx in range(100)],
            "is_anomaly": [idx % 10 == 0 for idx in range(100)],
            "realtime_phase": ["reference" if idx < 20 else "online" for idx in range(100)],
        }
        reduced = _prepare_viewer_frame(pd.DataFrame(raw), max_points=24, altitude_bins=4)
        self.assertLessEqual(len(reduced), 24)
        self.assertTrue(reduced["is_anomaly"].any())
        self.assertIn("reference", set(reduced["realtime_phase"]))
        self.assertLessEqual(reduced["altitude_m"].nunique(), 4)


if __name__ == "__main__":
    unittest.main()
