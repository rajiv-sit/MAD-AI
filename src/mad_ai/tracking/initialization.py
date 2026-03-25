from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .forward_model import DipoleMagneticForwardModel, project_position
from .types import MagneticTrackingObservation, SensorState, VesselState


def observations_from_bahamas_frame(frame: pd.DataFrame) -> list[MagneticTrackingObservation]:
    observations: list[MagneticTrackingObservation] = []
    for row in frame.itertuples(index=False):
        observations.append(
            MagneticTrackingObservation(
                sensor_state=SensorState(
                    latitude_deg=float(row.latitude_deg),
                    longitude_deg=float(row.longitude_deg),
                    altitude_m=float(getattr(row, "altitude_m", 0.0)),
                    timestamp=pd.Timestamp(getattr(row, "timestamp", None)).to_pydatetime() if getattr(row, "timestamp", None) is not None else None,
                    heading_deg=float(getattr(row, "aircraft_heading_deg", 0.0)) if hasattr(row, "aircraft_heading_deg") else None,
                    speed_mps=float(getattr(row, "aircraft_speed_mps", 0.0)) if hasattr(row, "aircraft_speed_mps") else None,
                ),
                observed_total_nt=float(getattr(row, "observed_total_nt", 0.0)),
                baseline_total_nt=float(getattr(row, "baseline_total_nt", 0.0)),
                residual_total_nt=float(getattr(row, "residual_total_nt", 0.0)),
                anomaly_score=float(getattr(row, "final_anomaly_score", 0.0)),
                track_id=str(getattr(row, "track_id", "")),
                metadata={
                    "reference_vessel_latitude_deg": float(getattr(row, "vessel_latitude_deg", 0.0)) if hasattr(row, "vessel_latitude_deg") else None,
                    "reference_vessel_longitude_deg": float(getattr(row, "vessel_longitude_deg", 0.0)) if hasattr(row, "vessel_longitude_deg") else None,
                    "reference_vessel_heading_deg": float(getattr(row, "vessel_heading_deg", 0.0)) if hasattr(row, "vessel_heading_deg") else None,
                    "reference_vessel_speed_mps": float(getattr(row, "vessel_speed_mps", 0.0)) if hasattr(row, "vessel_speed_mps") else None,
                    "range_to_vessel_m": float(getattr(row, "range_to_vessel_m", 0.0)) if hasattr(row, "range_to_vessel_m") else None,
                },
            )
        )
    return observations


def initialize_vessel_state_candidates(
    observations: list[MagneticTrackingObservation],
    top_k: int = 5,
    score_column_floor: float = 0.0,
    strategy: str = "reference_peaks",
    forward_model: DipoleMagneticForwardModel | None = None,
) -> list[VesselState]:
    if not observations:
        return []

    strategy = strategy.strip().lower()
    if strategy == "magnetic_bearing_grid":
        return _initialize_from_magnetic_bearing_grid(
            observations,
            top_k=top_k,
            score_column_floor=score_column_floor,
            forward_model=forward_model or DipoleMagneticForwardModel(),
        )

    ranked = sorted(observations, key=lambda observation: observation.anomaly_score, reverse=True)
    candidates: list[VesselState] = []
    seen: set[tuple[float, float]] = set()

    for observation in ranked:
        if observation.anomaly_score < score_column_floor:
            continue
        vessel_lat = observation.metadata.get("reference_vessel_latitude_deg")
        vessel_lon = observation.metadata.get("reference_vessel_longitude_deg")
        if vessel_lat is None or vessel_lon is None:
            continue
        rounded_key = (round(float(vessel_lat), 4), round(float(vessel_lon), 4))
        if rounded_key in seen:
            continue
        seen.add(rounded_key)
        candidates.append(
            VesselState(
                latitude_deg=float(vessel_lat),
                longitude_deg=float(vessel_lon),
                speed_mps=float(observation.metadata.get("reference_vessel_speed_mps") or 0.0),
                heading_deg=float(observation.metadata.get("reference_vessel_heading_deg") or 0.0),
                timestamp=observation.sensor_state.timestamp,
                metadata={
                    "initialization_source": "anomaly_peak_reference_seed",
                    "seed_anomaly_score": observation.anomaly_score,
                    "seed_range_to_vessel_m": observation.metadata.get("range_to_vessel_m"),
                    "track_id": observation.track_id,
                },
            )
        )
        if len(candidates) >= top_k:
            break

    if candidates:
        return candidates

    first = observations[0]
    return [
        VesselState(
            latitude_deg=first.sensor_state.latitude_deg,
            longitude_deg=first.sensor_state.longitude_deg,
            timestamp=first.sensor_state.timestamp,
            metadata={"initialization_source": "sensor_position_fallback", "track_id": first.track_id},
        )
    ]


def _initialize_from_magnetic_bearing_grid(
    observations: list[MagneticTrackingObservation],
    top_k: int,
    score_column_floor: float,
    forward_model: DipoleMagneticForwardModel,
) -> list[VesselState]:
    ranked = sorted(observations, key=lambda observation: observation.anomaly_score, reverse=True)
    seed_observation = next((observation for observation in ranked if observation.anomaly_score >= score_column_floor), observations[0])
    base_heading = float(seed_observation.sensor_state.heading_deg or 0.0)
    candidate_bearings = [
        (base_heading - 120.0) % 360.0,
        (base_heading - 90.0) % 360.0,
        (base_heading - 60.0) % 360.0,
        (base_heading + 60.0) % 360.0,
        (base_heading + 90.0) % 360.0,
        (base_heading + 120.0) % 360.0,
    ]
    base_vessel_state = VesselState(
        latitude_deg=seed_observation.sensor_state.latitude_deg,
        longitude_deg=seed_observation.sensor_state.longitude_deg,
        heading_deg=base_heading,
        speed_mps=0.0,
        timestamp=seed_observation.sensor_state.timestamp,
        metadata={"initialization_source": "magnetic_bearing_grid", "track_id": seed_observation.track_id},
    )
    slant_range_m = forward_model.infer_slant_range_m(seed_observation.residual_total_nt, base_vessel_state)
    vertical_separation_m = max(seed_observation.sensor_state.altitude_m + base_vessel_state.depth_m, 1.0)
    horizontal_range_m = max(np.sqrt(max(slant_range_m**2 - vertical_separation_m**2, 0.0)), forward_model.minimum_range_m)

    candidates: list[VesselState] = []
    for bearing in candidate_bearings[: max(1, top_k)]:
        latitude_deg, longitude_deg = project_position(
            seed_observation.sensor_state.latitude_deg,
            seed_observation.sensor_state.longitude_deg,
            bearing,
            horizontal_range_m,
        )
        candidates.append(
            VesselState(
                latitude_deg=latitude_deg,
                longitude_deg=longitude_deg,
                heading_deg=bearing,
                speed_mps=0.0,
                timestamp=seed_observation.sensor_state.timestamp,
                metadata={
                    "initialization_source": "magnetic_bearing_grid",
                    "seed_anomaly_score": seed_observation.anomaly_score,
                    "seed_slant_range_m": slant_range_m,
                    "seed_horizontal_range_m": horizontal_range_m,
                    "track_id": seed_observation.track_id,
                },
            )
        )
    return candidates


def propagate_constant_velocity(state: VesselState, dt_seconds: float) -> VesselState:
    if dt_seconds <= 0.0 or state.speed_mps <= 0.0:
        return state

    heading_rad = np.deg2rad(state.heading_deg)
    north_m = state.speed_mps * dt_seconds * np.cos(heading_rad)
    east_m = state.speed_mps * dt_seconds * np.sin(heading_rad)
    lat_deg = state.latitude_deg + (north_m / 111_320.0)
    lon_scale = max(np.cos(np.deg2rad(state.latitude_deg)), 1.0e-6)
    lon_deg = state.longitude_deg + (east_m / (111_320.0 * lon_scale))
    return replace(state, latitude_deg=float(lat_deg), longitude_deg=float(lon_deg))
