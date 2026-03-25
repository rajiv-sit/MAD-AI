from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from mad_ai.ingest import CsvSensorIngestor, ParquetSensorIngestor
from tests.unit.helpers import workspace_temp_dir


class IngestTestCase(unittest.TestCase):
    def test_csv_sensor_ingestor_loads_required_columns(self) -> None:
        with workspace_temp_dir() as tmp_dir:
            path = tmp_dir / "sensor.csv"
            pd.DataFrame(
                {
                    "latitude_deg": [43.7],
                    "longitude_deg": [-79.4],
                    "altitude_m": [0.0],
                }
            ).to_csv(path, index=False)
            loaded = CsvSensorIngestor().load(path)
            self.assertEqual(list(loaded.columns), ["latitude_deg", "longitude_deg", "altitude_m"])

    def test_csv_sensor_ingestor_rejects_missing_columns(self) -> None:
        with workspace_temp_dir() as tmp_dir:
            path = tmp_dir / "sensor.csv"
            pd.DataFrame({"latitude_deg": [43.7]}).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                CsvSensorIngestor().load(path)

    def test_parquet_sensor_ingestor_uses_dataframe_validation(self) -> None:
        frame = pd.DataFrame(
            {
                "latitude_deg": [43.7],
                "longitude_deg": [-79.4],
                "altitude_m": [0.0],
            }
        )
        with patch("pandas.read_parquet", return_value=frame) as mocked_read:
            loaded = ParquetSensorIngestor().load("stub.parquet")
            mocked_read.assert_called_once()
            self.assertEqual(loaded.shape, (1, 3))


if __name__ == "__main__":
    unittest.main()
