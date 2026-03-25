from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class CalibrationSummary:
    threshold: float
    percentile: float
    mean_score: float
    std_score: float
    min_score: float
    max_score: float
    sample_count: int


class ThresholdCalibrator:
    def __init__(self, percentile: float = 95.0) -> None:
        if not 0.0 < percentile <= 100.0:
            raise ValueError("percentile must be in (0, 100].")
        self.percentile = percentile

    def calibrate(self, scores) -> CalibrationSummary:
        array = np.asarray(scores, dtype=float).reshape(-1)
        if array.size == 0:
            raise ValueError("scores must not be empty.")
        threshold = float(np.percentile(array, self.percentile))
        return CalibrationSummary(
            threshold=threshold,
            percentile=self.percentile,
            mean_score=float(array.mean()),
            std_score=float(array.std()),
            min_score=float(array.min()),
            max_score=float(array.max()),
            sample_count=int(array.size),
        )


def summarize_scores(scores) -> dict[str, float | int]:
    array = np.asarray(scores, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError("scores must not be empty.")
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "count": int(array.size),
    }
