"""Tracking primitives for inverse magnetic vessel estimation."""

from .forward_model import DipoleMagneticForwardModel, GeometryAwareDipoleMagneticForwardModel, surface_range_m
from .initialization import (
    initialize_vessel_state_candidates,
    observations_from_bahamas_frame,
    propagate_constant_velocity,
)
from .tracker import ConstantVelocityMagneticTracker, MultiHypothesisMagneticTracker
from .types import MagneticTrackingObservation, SensorState, VesselState, VesselTrackEstimate

__all__ = [
    "ConstantVelocityMagneticTracker",
    "DipoleMagneticForwardModel",
    "GeometryAwareDipoleMagneticForwardModel",
    "MagneticTrackingObservation",
    "MultiHypothesisMagneticTracker",
    "SensorState",
    "VesselState",
    "VesselTrackEstimate",
    "initialize_vessel_state_candidates",
    "observations_from_bahamas_frame",
    "propagate_constant_velocity",
    "surface_range_m",
]
