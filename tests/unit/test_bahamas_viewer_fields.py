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
            payload = CesiumGlobeViewerBuilder()._to_payload(frame)
            first = payload[0]
            self.assertIn("trackId", first)
            self.assertIn("aircraftSpeedMps", first)
            self.assertIn("vesselSpeedMps", first)
            self.assertIn("vesselLat", first)


if __name__ == "__main__":
    unittest.main()
