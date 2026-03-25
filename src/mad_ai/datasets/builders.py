from __future__ import annotations

import numpy as np
import pandas as pd

from mad_ai.core.base import BaseDatasetBuilder


class SpatialGridBuilder(BaseDatasetBuilder):
    def __init__(self, rows: int = 24, cols: int = 24, feature_columns: list[str] | None = None) -> None:
        self.rows = rows
        self.cols = cols
        self.feature_columns = feature_columns or ["residual_total_nt", "baseline_total_nt"]

    def build(self, data: pd.DataFrame) -> np.ndarray:
        grid = np.zeros((self.rows, self.cols, len(self.feature_columns)), dtype=float)
        counts = np.zeros((self.rows, self.cols), dtype=float)

        lat_min = float(data["latitude_deg"].min())
        lat_max = float(data["latitude_deg"].max())
        lon_min = float(data["longitude_deg"].min())
        lon_max = float(data["longitude_deg"].max())

        lat_span = max(lat_max - lat_min, 1e-9)
        lon_span = max(lon_max - lon_min, 1e-9)

        for row in data.itertuples(index=False):
            grid_r = min(self.rows - 1, int(((float(row.latitude_deg) - lat_min) / lat_span) * (self.rows - 1)))
            grid_c = min(self.cols - 1, int(((float(row.longitude_deg) - lon_min) / lon_span) * (self.cols - 1)))
            for idx, column in enumerate(self.feature_columns):
                grid[grid_r, grid_c, idx] += float(getattr(row, column, 0.0))
            counts[grid_r, grid_c] += 1.0

        nonzero = counts > 0
        for idx in range(len(self.feature_columns)):
            grid[:, :, idx][nonzero] /= counts[nonzero]
        return grid


class TemporalSequenceBuilder(BaseDatasetBuilder):
    def __init__(self, sequence_length: int = 12, feature_columns: list[str] | None = None) -> None:
        self.sequence_length = sequence_length
        self.feature_columns = feature_columns or ["residual_total_nt", "delta_residual_total_nt"]

    def build(self, data: pd.DataFrame) -> np.ndarray:
        ordered = data.sort_values("timestamp").reset_index(drop=True) if "timestamp" in data.columns else data.reset_index(drop=True)
        values = ordered[self.feature_columns].fillna(0.0).to_numpy(dtype=float)
        if len(values) < self.sequence_length:
            padded = np.zeros((self.sequence_length, values.shape[1]), dtype=float)
            padded[: len(values)] = values
            return padded[np.newaxis, ...]

        sequences = []
        for start in range(0, len(values) - self.sequence_length + 1):
            sequences.append(values[start : start + self.sequence_length])
        return np.stack(sequences, axis=0)
