from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class ArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_dataframe(self, name: str, data: pd.DataFrame) -> Path:
        path = self.root / f"{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, index=False)
        return path

    def load_dataframe(self, name: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
        path = self.root / f"{name}.csv"
        return pd.read_csv(path, parse_dates=parse_dates)

    def save_array(self, name: str, data: np.ndarray) -> Path:
        path = self.root / f"{name}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, data)
        return path

    def load_array(self, name: str) -> np.ndarray:
        path = self.root / f"{name}.npy"
        return np.load(path, allow_pickle=False)

    def save_json(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.root / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path
