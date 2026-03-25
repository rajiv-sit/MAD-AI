from __future__ import annotations

from pathlib import Path

import pandas as pd

from mad_ai.core.base import BaseDataIngestor


REQUIRED_COLUMNS = {"latitude_deg", "longitude_deg", "altitude_m"}


class CsvSensorIngestor(BaseDataIngestor):
    def load(self, source: str | Path) -> pd.DataFrame:
        data = pd.read_csv(source)
        data = _normalize_optional_columns(data)
        _validate_columns(data)
        return data


class ParquetSensorIngestor(BaseDataIngestor):
    def load(self, source: str | Path) -> pd.DataFrame:
        data = pd.read_parquet(source)
        data = _normalize_optional_columns(data)
        _validate_columns(data)
        return data


def _validate_columns(data: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _normalize_optional_columns(data: pd.DataFrame) -> pd.DataFrame:
    normalized = data.copy()
    if "timestamp" in normalized.columns:
        normalized["timestamp"] = pd.to_datetime(normalized["timestamp"])
    return normalized
