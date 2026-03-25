from __future__ import annotations

from math import asin, atan2, cos, degrees, radians, sin, sqrt

from mad_ai.core.base import BaseMagneticForwardModel

from .types import SensorState, VesselState


EARTH_RADIUS_M = 6_371_000.0


def surface_range_m(
    latitude_a_deg: float,
    longitude_a_deg: float,
    latitude_b_deg: float,
    longitude_b_deg: float,
) -> float:
    lat1 = radians(latitude_a_deg)
    lon1 = radians(longitude_a_deg)
    lat2 = radians(latitude_b_deg)
    lon2 = radians(longitude_b_deg)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2.0) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2.0) ** 2
    c = 2.0 * atan2(sqrt(a), sqrt(max(1.0e-12, 1.0 - a)))
    return EARTH_RADIUS_M * c


def bearing_deg(
    latitude_a_deg: float,
    longitude_a_deg: float,
    latitude_b_deg: float,
    longitude_b_deg: float,
) -> float:
    lat1 = radians(latitude_a_deg)
    lon1 = radians(longitude_a_deg)
    lat2 = radians(latitude_b_deg)
    lon2 = radians(longitude_b_deg)
    y = sin(lon2 - lon1) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(lon2 - lon1)
    return (degrees(atan2(y, x)) + 360.0) % 360.0


def project_position(
    latitude_deg: float,
    longitude_deg: float,
    bearing_degrees: float,
    range_m: float,
) -> tuple[float, float]:
    angular_distance = range_m / EARTH_RADIUS_M
    bearing_rad = radians(bearing_degrees)
    lat1 = radians(latitude_deg)
    lon1 = radians(longitude_deg)

    lat2 = asin(sin(lat1) * cos(angular_distance) + cos(lat1) * sin(angular_distance) * cos(bearing_rad))
    lon2 = lon1 + atan2(
        sin(bearing_rad) * sin(angular_distance) * cos(lat1),
        cos(angular_distance) - sin(lat1) * sin(lat2),
    )
    return degrees(lat2), ((degrees(lon2) + 540.0) % 360.0) - 180.0


class DipoleMagneticForwardModel(BaseMagneticForwardModel):
    """Simple first-pass forward model for tracking scaffolding.

    This is not an operationally validated vessel magnetic model. It gives the
    tracking layer a stable interface and a distance-decay response that can be
    replaced later with a stronger physics-based model.
    """

    def __init__(self, minimum_range_m: float = 25.0) -> None:
        self.minimum_range_m = minimum_range_m

    def predict_total_field_nt(self, sensor_state: SensorState, vessel_state: VesselState, baseline_field_nt: float) -> float:
        residual = self.predict_residual_nt(sensor_state, vessel_state)
        return float(baseline_field_nt) + residual

    def predict_residual_nt(self, sensor_state: SensorState, vessel_state: VesselState) -> float:
        range_m = max(
            surface_range_m(
                sensor_state.latitude_deg,
                sensor_state.longitude_deg,
                vessel_state.latitude_deg,
                vessel_state.longitude_deg,
            ),
            self.minimum_range_m,
        )
        vertical_separation_m = max(sensor_state.altitude_m + vessel_state.depth_m, 1.0)
        slant_range_m = sqrt(range_m**2 + vertical_separation_m**2)
        return float(vessel_state.magnetic_moment_am2 / (slant_range_m**3))

    def infer_slant_range_m(self, residual_total_nt: float, vessel_state: VesselState) -> float:
        magnitude_nt = max(abs(float(residual_total_nt)), 1.0e-9)
        return max((vessel_state.magnetic_moment_am2 / magnitude_nt) ** (1.0 / 3.0), self.minimum_range_m)

    def infer_magnetic_moment_am2(self, residual_total_nt: float, slant_range_m: float) -> float:
        return max(abs(float(residual_total_nt)) * max(slant_range_m, self.minimum_range_m) ** 3, 1.0)


class GeometryAwareDipoleMagneticForwardModel(DipoleMagneticForwardModel):
    """Dipole model with simple platform-geometry weighting.

    This is still a lightweight approximation, but it accounts for:
    - relative bearing between aircraft heading and target bearing
    - a modest dependence on aircraft speed
    """

    def __init__(
        self,
        minimum_range_m: float = 25.0,
        min_geometry_gain: float = 0.35,
        speed_gain_scale: float = 0.002,
    ) -> None:
        super().__init__(minimum_range_m=minimum_range_m)
        self.min_geometry_gain = min_geometry_gain
        self.speed_gain_scale = speed_gain_scale

    def predict_residual_nt(self, sensor_state: SensorState, vessel_state: VesselState) -> float:
        base_residual = super().predict_residual_nt(sensor_state, vessel_state)
        geometry_gain = self._geometry_gain(sensor_state, vessel_state)
        return base_residual * geometry_gain

    def infer_slant_range_m(self, residual_total_nt: float, vessel_state: VesselState, sensor_state: SensorState | None = None) -> float:
        magnitude_nt = max(abs(float(residual_total_nt)), 1.0e-9)
        geometry_gain = 1.0 if sensor_state is None else max(self._geometry_gain(sensor_state, vessel_state), 1.0e-6)
        adjusted = magnitude_nt / geometry_gain
        return max((vessel_state.magnetic_moment_am2 / adjusted) ** (1.0 / 3.0), self.minimum_range_m)

    def infer_magnetic_moment_am2(
        self,
        residual_total_nt: float,
        slant_range_m: float,
        sensor_state: SensorState | None = None,
        vessel_state: VesselState | None = None,
    ) -> float:
        geometry_gain = 1.0
        if sensor_state is not None and vessel_state is not None:
            geometry_gain = max(self._geometry_gain(sensor_state, vessel_state), 1.0e-6)
        return max(abs(float(residual_total_nt)) * max(slant_range_m, self.minimum_range_m) ** 3 / geometry_gain, 1.0)

    def _geometry_gain(self, sensor_state: SensorState, vessel_state: VesselState) -> float:
        sensor_heading_deg = float(sensor_state.heading_deg or 0.0)
        target_bearing_deg = bearing_deg(
            sensor_state.latitude_deg,
            sensor_state.longitude_deg,
            vessel_state.latitude_deg,
            vessel_state.longitude_deg,
        )
        relative_bearing_deg = ((target_bearing_deg - sensor_heading_deg + 540.0) % 360.0) - 180.0
        side_gain = max(abs(sin(radians(relative_bearing_deg))), self.min_geometry_gain)
        speed_gain = 1.0 + self.speed_gain_scale * float(sensor_state.speed_mps or 0.0)
        return max(side_gain * speed_gain, self.min_geometry_gain)
