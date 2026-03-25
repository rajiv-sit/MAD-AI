from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class SensorState:
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    timestamp: datetime | None = None
    heading_deg: float | None = None
    speed_mps: float | None = None


@dataclass(slots=True)
class VesselState:
    latitude_deg: float
    longitude_deg: float
    speed_mps: float = 0.0
    heading_deg: float = 0.0
    depth_m: float = 0.0
    magnetic_moment_am2: float = 1.0e6
    timestamp: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MagneticTrackingObservation:
    sensor_state: SensorState
    observed_total_nt: float
    baseline_total_nt: float
    residual_total_nt: float
    anomaly_score: float
    track_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VesselTrackEstimate:
    vessel_state: VesselState
    expected_total_nt: float
    expected_residual_nt: float
    innovation_nt: float
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)
