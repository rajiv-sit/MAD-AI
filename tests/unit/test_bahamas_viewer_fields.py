from __future__ import annotations

import unittest

from mad_ai.ingest import load_bahamas_mad_ascii
from mad_ai.visualizer import CesiumGlobeViewerBuilder
from tests.unit.helpers import workspace_temp_dir


class BahamasViewerFieldsTestCase(unittest.TestCase):
    def test_bahamas_loader_derives_aircraft_and_vessel_speed(self) -> None:
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
            self.assertIn("aircraft_speed_mps", frame.columns)
            self.assertIn("vessel_speed_mps", frame.columns)
            self.assertGreaterEqual(float(frame["aircraft_speed_mps"].iloc[0]), 0.0)

    def test_viewer_payload_contains_track_state_fields(self) -> None:
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
            frame["estimated_vessel_latitude_deg"] = frame["vessel_latitude_deg"] + 0.01
            frame["estimated_vessel_longitude_deg"] = frame["vessel_longitude_deg"] - 0.01
            frame["tracking_error_m"] = [1200.0, 1400.0]
            frame["confidence"] = [0.2, 0.3]
            payload = CesiumGlobeViewerBuilder()._to_payload(frame)
            first = payload[0]
            self.assertIn("trackId", first)
            self.assertIn("aircraftSpeedMps", first)
            self.assertIn("vesselSpeedMps", first)
            self.assertIn("vesselLat", first)
            self.assertIn("rangeToVesselM", first)
            self.assertIn("estimatedVesselLat", first)
            self.assertIn("trackingErrorM", first)

    def test_viewer_html_contains_time_sync_bridge(self) -> None:
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
            frame["baseline_total_nt"] = [20.0, 20.0]
            frame["residual_total_nt"] = frame["observed_total_nt"] - frame["baseline_total_nt"]
            frame["final_anomaly_score"] = [0.1, 0.2]
            frame["is_anomaly"] = [False, True]

            output = CesiumGlobeViewerBuilder().build(frame, tmp_dir / "viewer.html")
            content = output.read_text(encoding="utf-8")
            self.assertIn("mad-ai-time-change", content)
            self.assertIn("mad-ai-set-timestamp", content)


if __name__ == "__main__":
    unittest.main()
