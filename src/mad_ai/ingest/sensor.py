from __future__ import annotations

from pathlib import Path

import pandas as pd

from mad_ai.core.base import BaseDataIngestor


REQUIRED_COLUMNS = {"latitude_deg", "longitude_deg", "altitude_m"}
DEFAULT_SCHEMA_MAPPING = {
    "latitude_deg": ["latitude_deg", "latitude", "lat", "sensor_lat"],
    "longitude_deg": ["longitude_deg", "longitude", "lon", "sensor_lon"],
    "altitude_m": ["altitude_m", "altitude", "alt_m", "alt"],
    "timestamp": ["timestamp", "time", "datetime", "utc_time"],
    "observed_total_nt": ["observed_total_nt", "total_field_nt", "mag_total_nt", "total_nt"],
    "observed_declination_deg": ["observed_declination_deg", "declination_deg", "mag_declination_deg"],
    "observed_inclination_deg": ["observed_inclination_deg", "inclination_deg", "mag_inclination_deg"],
    "track_id": ["track_id", "platform_id", "sensor_id"],
    "is_injected_anomaly": ["is_injected_anomaly", "is_anomaly_label", "label_anomaly"],
}


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


class SchemaMappedSensorIngestor(BaseDataIngestor):
    def __init__(self, schema_mapping: dict[str, list[str] | str] | None = None) -> None:
        self.schema_mapping = schema_mapping or DEFAULT_SCHEMA_MAPPING
        self.csv_ingestor = CsvSensorIngestor()
        self.parquet_ingestor = ParquetSensorIngestor()

    def load(self, source: str | Path) -> pd.DataFrame:
        path = Path(source)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            data = pd.read_csv(path)
        elif suffix == ".parquet":
            data = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported sensor file format: {suffix}")

        normalized = _apply_schema_mapping(data, self.schema_mapping)
        normalized = _normalize_optional_columns(normalized)
        _validate_columns(normalized)
        return normalized


class BatchSensorIngestor(BaseDataIngestor):
    def __init__(
        self,
        schema_mapping: dict[str, list[str] | str] | None = None,
        file_extensions: tuple[str, ...] = (".csv", ".parquet"),
    ) -> None:
        self.schema_mapping = schema_mapping or DEFAULT_SCHEMA_MAPPING
        self.file_extensions = tuple(ext.lower() for ext in file_extensions)
        self.file_ingestor = SchemaMappedSensorIngestor(schema_mapping=self.schema_mapping)

    def load(self, source: str | Path) -> pd.DataFrame:
        source_path = Path(source)
        if source_path.is_file():
            frame = self.file_ingestor.load(source_path)
            frame["source_file"] = source_path.name
            return frame

        files = sorted(
            path for path in source_path.rglob("*") if path.is_file() and path.suffix.lower() in self.file_extensions
        )
        if not files:
            raise FileNotFoundError(f"No supported sensor files found under {source_path}")

        frames: list[pd.DataFrame] = []
        for path in files:
            frame = self.file_ingestor.load(path)
            frame["source_file"] = path.name
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)


class SplitBatchSensorIngestor:
    def __init__(self, schema_mapping: dict[str, list[str] | str] | None = None) -> None:
        self.batch_ingestor = BatchSensorIngestor(schema_mapping=schema_mapping)

    def load_splits(self, split_sources: dict[str, str | Path]) -> dict[str, pd.DataFrame]:
        return {split_name: self.batch_ingestor.load(source) for split_name, source in split_sources.items()}


def _validate_columns(data: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _normalize_optional_columns(data: pd.DataFrame) -> pd.DataFrame:
    normalized = data.copy()
    if "timestamp" in normalized.columns:
        normalized["timestamp"] = pd.to_datetime(normalized["timestamp"])
    if "is_injected_anomaly" in normalized.columns:
        normalized["is_injected_anomaly"] = normalized["is_injected_anomaly"].astype(bool)
    return normalized


def _apply_schema_mapping(
    data: pd.DataFrame,
    schema_mapping: dict[str, list[str] | str],
) -> pd.DataFrame:
    normalized = data.copy()
    rename_map: dict[str, str] = {}
    for canonical_name, candidate_names in schema_mapping.items():
        candidates = [candidate_names] if isinstance(candidate_names, str) else list(candidate_names)
        if canonical_name in normalized.columns:
            continue
        matched_name = next((name for name in candidates if name in normalized.columns), None)
        if matched_name is not None:
            rename_map[matched_name] = canonical_name
    normalized = normalized.rename(columns=rename_map)
    return normalized
