from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from mad_ai.core.types import AnomalyResult


class SpatialScorer:
    def __init__(self, model) -> None:
        self.model = model

    def score(self, grid) -> float:
        value = self.model.score(grid)
        return float(value[0] if hasattr(value, "__len__") and not isinstance(value, (str, bytes)) else value)


class TemporalScorer:
    def __init__(self, model) -> None:
        self.model = model

    def score(self, sequence) -> float:
        value = self.model.score(sequence)
        return float(value[0] if hasattr(value, "__len__") and not isinstance(value, (str, bytes)) else value)


class AnomalyFusionEngine:
    def __init__(self, spatial_weight: float = 0.5, temporal_weight: float = 0.5, threshold: float = 0.35) -> None:
        total = spatial_weight + temporal_weight
        self.spatial_weight = spatial_weight / total
        self.temporal_weight = temporal_weight / total
        self.threshold = threshold

    def fuse(self, spatial_score: float, temporal_score: float) -> AnomalyResult:
        final_score = (self.spatial_weight * spatial_score) + (self.temporal_weight * temporal_score)
        return AnomalyResult(
            spatial_score=spatial_score,
            temporal_score=temporal_score,
            final_score=final_score,
            is_anomaly=final_score >= self.threshold,
            metadata={"threshold": self.threshold},
        )

    def fuse_to_dict(self, spatial_score: float, temporal_score: float) -> dict[str, object]:
        return asdict(self.fuse(spatial_score, temporal_score))

    @classmethod
    def from_json(cls, path: str | Path, default_threshold: float = 0.35) -> "AnomalyFusionEngine":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            spatial_weight=float(payload.get("spatial_weight", 0.5)),
            temporal_weight=float(payload.get("temporal_weight", 0.5)),
            threshold=float(payload.get("threshold", default_threshold)),
        )
