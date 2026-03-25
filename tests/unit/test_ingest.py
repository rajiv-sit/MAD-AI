from __future__ import annotations

import sqlite3
import unittest
from unittest.mock import patch

import pandas as pd

from mad_ai.ingest import (
    BatchSensorIngestor,
    CsvSensorIngestor,
    JsonlSensorIngestor,
    ParquetSensorIngestor,
    SchemaMappedSensorIngestor,
    SplitBatchSensorIngestor,
    SqliteSensorIngestor,
)
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

    def test_schema_mapped_sensor_ingestor_normalizes_real_sensor_columns(self) -> None:
        with workspace_temp_dir() as tmp_dir:
            path = tmp_dir / "mapped.csv"
            pd.DataFrame(
                {
                    "latitude": [43.7],
                    "longitude": [-79.4],
                    "altitude": [25.0],
                    "time": ["2026-03-24T00:00:00"],
                    "total_field_nt": [52000.0],
                }
            ).to_csv(path, index=False)
            loaded = SchemaMappedSensorIngestor().load(path)
            self.assertIn("latitude_deg", loaded.columns)
            self.assertIn("longitude_deg", loaded.columns)
            self.assertIn("altitude_m", loaded.columns)
            self.assertIn("timestamp", loaded.columns)
            self.assertIn("observed_total_nt", loaded.columns)

    def test_batch_sensor_ingestor_loads_multiple_files(self) -> None:
        with workspace_temp_dir() as tmp_dir:
            first = tmp_dir / "a.csv"
            second = tmp_dir / "nested" / "b.csv"
            second.parent.mkdir(parents=True, exist_ok=True)
            frame = pd.DataFrame(
                {
                    "latitude": [43.7],
                    "longitude": [-79.4],
                    "altitude": [0.0],
                }
            )
            frame.to_csv(first, index=False)
            frame.to_csv(second, index=False)
            loaded = BatchSensorIngestor().load(tmp_dir)
            self.assertEqual(len(loaded), 2)
            self.assertIn("source_file", loaded.columns)

    def test_split_batch_sensor_ingestor_loads_named_splits(self) -> None:
        with workspace_temp_dir() as tmp_dir:
            train_dir = tmp_dir / "train"
            eval_dir = tmp_dir / "eval"
            train_dir.mkdir()
            eval_dir.mkdir()
            pd.DataFrame(
                {
                    "latitude": [43.7],
                    "longitude": [-79.4],
                    "altitude": [0.0],
                }
            ).to_csv(train_dir / "train.csv", index=False)
            pd.DataFrame(
                {
                    "latitude": [43.8],
                    "longitude": [-79.3],
                    "altitude": [10.0],
                }
            ).to_csv(eval_dir / "eval.csv", index=False)
            splits = SplitBatchSensorIngestor().load_splits({"train": train_dir, "eval": eval_dir})
            self.assertEqual(sorted(splits.keys()), ["eval", "train"])
            self.assertEqual(len(splits["train"]), 1)
            self.assertEqual(len(splits["eval"]), 1)

    def test_jsonl_sensor_ingestor_loads_lines_file(self) -> None:
        with workspace_temp_dir() as tmp_dir:
            path = tmp_dir / "sensor.jsonl"
            pd.DataFrame(
                {
                    "latitude_deg": [43.7],
                    "longitude_deg": [-79.4],
                    "altitude_m": [0.0],
                }
            ).to_json(path, orient="records", lines=True)
            loaded = JsonlSensorIngestor().load(path)
            self.assertEqual(len(loaded), 1)

    def test_sqlite_sensor_ingestor_loads_sensor_table(self) -> None:
        with workspace_temp_dir() as tmp_dir:
            path = tmp_dir / "sensor.sqlite"
            with sqlite3.connect(path) as connection:
                pd.DataFrame(
                    {
                        "latitude_deg": [43.7],
                        "longitude_deg": [-79.4],
                        "altitude_m": [5.0],
                    }
                ).to_sql("sensor_readings", connection, index=False, if_exists="replace")
            loaded = SqliteSensorIngestor().load(path)
            self.assertEqual(len(loaded), 1)
            self.assertIn("altitude_m", loaded.columns)


if __name__ == "__main__":
    unittest.main()
