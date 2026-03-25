from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class MagneticField:
    declination_deg: float
    inclination_deg: float
    total_intensity_nt: float
    horizontal_intensity_nt: float
    north_nt: float
    east_nt: float
    down_nt: float
    source: str = "unknown"


@dataclass(slots=True)
class MagneticSample:
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    timestamp: datetime | None = None
    observed_total_nt: float | None = None
    observed_declination_deg: float | None = None
    observed_inclination_deg: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AnomalyResult:
    spatial_score: float
    temporal_score: float
    final_score: float
    is_anomaly: bool
    metadata: dict[str, Any] = field(default_factory=dict)
