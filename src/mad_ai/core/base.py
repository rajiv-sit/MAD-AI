from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseMagneticModel(ABC):
    @abstractmethod
    def get_field(self, lat: float, lon: float, alt: float, timestamp: Any = None) -> dict[str, Any]:
        raise NotImplementedError


class BaseDataIngestor(ABC):
    @abstractmethod
    def load(self, source: str | Path):
        raise NotImplementedError


class BaseFeatureBuilder(ABC):
    @abstractmethod
    def transform(self, data):
        raise NotImplementedError


class BaseDatasetBuilder(ABC):
    @abstractmethod
    def build(self, data):
        raise NotImplementedError


class BaseAnomalyModel(ABC):
    @abstractmethod
    def train(self, train_data, val_data=None) -> None:
        raise NotImplementedError

    @abstractmethod
    def score(self, batch):
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str | Path) -> None:
        raise NotImplementedError


class BaseVisualizer(ABC):
    @abstractmethod
    def render(self, data, output_path: str | Path) -> Path:
        raise NotImplementedError


class BaseViewModel(ABC):
    @abstractmethod
    def build_view_state(self, data):
        raise NotImplementedError


class BaseMagneticForwardModel(ABC):
    @abstractmethod
    def predict_total_field_nt(self, sensor_state, vessel_state, baseline_field_nt: float) -> float:
        raise NotImplementedError


class BaseTracker(ABC):
    @abstractmethod
    def initialize(self, observations):
        raise NotImplementedError

    @abstractmethod
    def step(self, observation):
        raise NotImplementedError
