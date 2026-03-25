from __future__ import annotations

import unittest

import pandas as pd

from mad_ai.ingest import CsvSensorIngestor
from mad_ai.utils import ArtifactStore, build_processed_artifacts
from mad_ai.wmm import WMMMagneticModel
from tests.unit.helpers import workspace_temp_dir


class RealDataPipelineTestCase(unittest.TestCase):
    def test_csv_sensor_ingestor_parses_timestamp_column(self) -> None:
        with workspace_temp_dir() as tmp_dir:
            path = tmp_dir / "sensor.csv"
            pd.DataFrame(
                {
                    "latitude_deg": [43.7, 43.8],
                    "longitude_deg": [-79.4, -79.3],
                    "altitude_m": [0.0, 1.0],
                    "timestamp": ["2026-03-24T00:00:00", "2026-03-24T00:05:00"],
                    "observed_total_nt": [52000.0, 52010.0],
                }
            ).to_csv(path, index=False)

            loaded = CsvSensorIngestor().load(path)
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(loaded["timestamp"]))

    def test_build_processed_artifacts_saves_outputs_from_loaded_csv_data(self) -> None:
        with workspace_temp_dir() as tmp_dir:
            csv_path = tmp_dir / "sensor.csv"
            pd.DataFrame(
                {
                    "latitude_deg": [43.7, 43.8, 43.9],
                    "longitude_deg": [-79.4, -79.3, -79.2],
                    "altitude_m": [0.0, 1.0, 2.0],
                    "timestamp": ["2026-03-24T00:00:00", "2026-03-24T00:05:00", "2026-03-24T00:10:00"],
                    "observed_total_nt": [52000.0, 52010.0, 52020.0],
                    "observed_declination_deg": [-8.0, -8.1, -8.2],
                    "observed_inclination_deg": [58.0, 58.1, 58.2],
                }
            ).to_csv(csv_path, index=False)

            raw = CsvSensorIngestor().load(csv_path)
            store = ArtifactStore(tmp_dir / "processed")
            paths = build_processed_artifacts(raw, store, WMMMagneticModel(cache_path=tmp_dir / "wmm_cache.json"))

            self.assertTrue(paths["enriched_samples"].exists())
            self.assertTrue(paths["temporal_samples"].exists())
            self.assertTrue(paths["spatial_grid"].exists())
            self.assertTrue(paths["temporal_sequences"].exists())
            self.assertTrue(paths["dataset_metadata"].exists())


if __name__ == "__main__":
    unittest.main()
