from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mad_ai.core.base import BaseViewModel


@dataclass(slots=True)
class MapViewState:
    title: str
    data: pd.DataFrame
    value_column: str


@dataclass(slots=True)
class TimelineViewState:
    title: str
    data: pd.DataFrame
    x_column: str
    y_column: str


@dataclass(slots=True)
class AnomalyReviewState:
    title: str
    data: pd.DataFrame
    events: pd.DataFrame
    residual_column: str
    threshold: float


class MapViewModel(BaseViewModel):
    def __init__(self, value_column: str = "baseline_total_nt", title: str = "Magnetic Heatmap") -> None:
        self.value_column = value_column
        self.title = title

    def build_view_state(self, data: pd.DataFrame) -> MapViewState:
        return MapViewState(title=self.title, data=data.copy(), value_column=self.value_column)


class TimelineViewModel(BaseViewModel):
    def __init__(self, x_column: str = "timestamp", y_column: str = "residual_total_nt", title: str = "Residual Timeline") -> None:
        self.x_column = x_column
        self.y_column = y_column
        self.title = title

    def build_view_state(self, data: pd.DataFrame) -> TimelineViewState:
        return TimelineViewState(title=self.title, data=data.copy(), x_column=self.x_column, y_column=self.y_column)


class AnomalyReviewViewModel(BaseViewModel):
    def __init__(self, residual_column: str = "residual_total_nt", threshold: float = 250.0, title: str = "Anomaly Review") -> None:
        self.residual_column = residual_column
        self.threshold = threshold
        self.title = title

    def build_view_state(self, data: pd.DataFrame) -> AnomalyReviewState:
        review_data = data.copy()
        review_data["is_anomaly_event"] = review_data[self.residual_column].abs() >= self.threshold
        events = review_data.loc[review_data["is_anomaly_event"]].copy()
        return AnomalyReviewState(
            title=self.title,
            data=review_data,
            events=events,
            residual_column=self.residual_column,
            threshold=self.threshold,
        )
