from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from datetime import datetime

from mad_ai.core.base import BaseTracker

from .forward_model import DipoleMagneticForwardModel, GeometryAwareDipoleMagneticForwardModel, bearing_deg, project_position, surface_range_m
from .initialization import initialize_vessel_state_candidates, propagate_constant_velocity
from .types import MagneticTrackingObservation, VesselState, VesselTrackEstimate


@dataclass(slots=True)
class _TrackingHypothesis:
    state: VesselState
    cumulative_cost: float
    step_cost: float
    confidence: float
    innovation_history: tuple[float, ...]


class ConstantVelocityMagneticTracker(BaseTracker):
    """First-pass tracker for inverse magnetic tracking experiments.

    This implementation is intentionally simple. It uses:
    - magnetic-bearing-grid initialization
    - constant-velocity propagation
    - residual-magnitude range inference
    - bearing-preserving geometry correction relative to the sensor platform
    """

    def __init__(
        self,
        forward_model: DipoleMagneticForwardModel | None = None,
        candidate_count: int = 5,
        initialization_strategy: str = "magnetic_bearing_grid",
        moment_blend_alpha: float = 0.15,
    ) -> None:
        self.forward_model = forward_model or GeometryAwareDipoleMagneticForwardModel()
        self.candidate_count = candidate_count
        self.initialization_strategy = initialization_strategy
        self.moment_blend_alpha = moment_blend_alpha
        self._state: VesselState | None = None

    def initialize(self, observations: list[MagneticTrackingObservation]) -> VesselTrackEstimate:
        candidates = initialize_vessel_state_candidates(
            observations,
            top_k=self.candidate_count,
            strategy=self.initialization_strategy,
            forward_model=self.forward_model,
        )
        if not candidates:
            raise ValueError("Tracker initialization requires at least one observation.")
        seed = candidates[0]
        self._state = seed
        first_observation = observations[0]
        return self._estimate_for_observation(first_observation, seed, confidence=0.25)

    def step(self, observation: MagneticTrackingObservation) -> VesselTrackEstimate:
        if self._state is None:
            raise RuntimeError("Tracker must be initialized before calling step().")

        dt_seconds = _delta_seconds(self._state.timestamp, observation.sensor_state.timestamp)
        predicted_state = propagate_constant_velocity(self._state, dt_seconds)
        corrected_state = _correct_state_from_geometry(
            predicted_state,
            observation,
            self.forward_model,
            moment_blend_alpha=self.moment_blend_alpha,
        )
        self._state = replace(corrected_state, timestamp=observation.sensor_state.timestamp)
        confidence = max(0.0, min(1.0, float(observation.anomaly_score)))
        return self._estimate_for_observation(observation, self._state, confidence=confidence)

    def _estimate_for_observation(
        self,
        observation: MagneticTrackingObservation,
        vessel_state: VesselState,
        confidence: float,
    ) -> VesselTrackEstimate:
        expected_total_nt = self.forward_model.predict_total_field_nt(
            observation.sensor_state,
            vessel_state,
            observation.baseline_total_nt,
        )
        expected_residual_nt = expected_total_nt - observation.baseline_total_nt
        innovation_nt = observation.observed_total_nt - expected_total_nt
        return VesselTrackEstimate(
            vessel_state=replace(vessel_state),
            expected_total_nt=expected_total_nt,
            expected_residual_nt=expected_residual_nt,
            innovation_nt=innovation_nt,
            confidence=confidence,
            metadata={
                "track_id": observation.track_id,
                "range_to_sensor_m": surface_range_m(
                    observation.sensor_state.latitude_deg,
                    observation.sensor_state.longitude_deg,
                    vessel_state.latitude_deg,
                    vessel_state.longitude_deg,
                ),
            },
        )


def _delta_seconds(previous: datetime | None, current: datetime | None) -> float:
    if previous is None or current is None:
        return 0.0
    return max((current - previous).total_seconds(), 0.0)


def _correct_state_from_geometry(
    predicted_state: VesselState,
    observation: MagneticTrackingObservation,
    forward_model: DipoleMagneticForwardModel,
    moment_blend_alpha: float = 0.15,
) -> VesselState:
    slant_range_m = _infer_slant_range(forward_model, observation, predicted_state)
    vertical_separation_m = max(observation.sensor_state.altitude_m + predicted_state.depth_m, 1.0)
    horizontal_range_m = max((max(slant_range_m**2 - vertical_separation_m**2, 0.0)) ** 0.5, forward_model.minimum_range_m)

    current_surface_range_m = surface_range_m(
        observation.sensor_state.latitude_deg,
        observation.sensor_state.longitude_deg,
        predicted_state.latitude_deg,
        predicted_state.longitude_deg,
    )
    if current_surface_range_m <= forward_model.minimum_range_m:
        bearing_to_target_deg = (float(observation.sensor_state.heading_deg or predicted_state.heading_deg) - 90.0) % 360.0
    else:
        bearing_to_target_deg = bearing_deg(
            observation.sensor_state.latitude_deg,
            observation.sensor_state.longitude_deg,
            predicted_state.latitude_deg,
            predicted_state.longitude_deg,
        )

    latitude_deg, longitude_deg = project_position(
        observation.sensor_state.latitude_deg,
        observation.sensor_state.longitude_deg,
        bearing_to_target_deg,
        horizontal_range_m,
    )
    updated_speed_mps = predicted_state.speed_mps
    updated_heading_deg = predicted_state.heading_deg
    dt_seconds = _delta_seconds(predicted_state.timestamp, observation.sensor_state.timestamp)
    if dt_seconds > 0.0:
        traveled_m = surface_range_m(predicted_state.latitude_deg, predicted_state.longitude_deg, latitude_deg, longitude_deg)
        updated_speed_mps = traveled_m / dt_seconds
        updated_heading_deg = bearing_deg(
            predicted_state.latitude_deg,
            predicted_state.longitude_deg,
            latitude_deg,
            longitude_deg,
        )
    inferred_moment_am2 = _infer_magnetic_moment(forward_model, observation, predicted_state, slant_range_m)
    updated_moment_am2 = _blend_moment(predicted_state.magnetic_moment_am2, inferred_moment_am2, moment_blend_alpha)
    return replace(
        predicted_state,
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
        speed_mps=updated_speed_mps,
        heading_deg=updated_heading_deg,
        magnetic_moment_am2=updated_moment_am2,
        metadata={
            **predicted_state.metadata,
            "track_side": _track_side(
                float(observation.sensor_state.heading_deg or bearing_to_target_deg),
                bearing_to_target_deg,
            ),
        },
    )


class MultiHypothesisMagneticTracker(BaseTracker):
    """Beam-style magnetic tracker.

    Keeps several vessel-state hypotheses alive and selects the lowest-cost
    state at each step based on magnetic innovation and smoothness.
    """

    def __init__(
        self,
        forward_model: DipoleMagneticForwardModel | None = None,
        candidate_count: int = 5,
        beam_width: int = 4,
        initialization_strategy: str = "magnetic_bearing_grid",
        innovation_weight: float = 1.0,
        motion_weight: float = 0.2,
        moment_blend_alpha: float = 0.15,
        innovation_window: int = 12,
    ) -> None:
        self.forward_model = forward_model or GeometryAwareDipoleMagneticForwardModel()
        self.candidate_count = candidate_count
        self.beam_width = max(1, beam_width)
        self.initialization_strategy = initialization_strategy
        self.innovation_weight = innovation_weight
        self.motion_weight = motion_weight
        self.moment_blend_alpha = moment_blend_alpha
        self.innovation_window = max(1, innovation_window)
        self._hypotheses: list[_TrackingHypothesis] = []

    def initialize(self, observations: list[MagneticTrackingObservation]) -> VesselTrackEstimate:
        if not observations:
            raise ValueError("Tracker initialization requires at least one observation.")
        first_observation = observations[0]
        candidates = initialize_vessel_state_candidates(
            observations,
            top_k=max(self.candidate_count, self.beam_width),
            strategy=self.initialization_strategy,
            forward_model=self.forward_model,
        )
        if not candidates:
            raise ValueError("Tracker initialization requires at least one candidate state.")

        seeded: list[_TrackingHypothesis] = []
        for candidate in candidates[: self.beam_width]:
            estimate = _estimate_for_observation(first_observation, candidate, 0.25, self.forward_model)
            innovation_cost = abs(estimate.innovation_nt)
            seeded.append(
                _TrackingHypothesis(
                    state=replace(candidate, timestamp=first_observation.sensor_state.timestamp),
                    cumulative_cost=innovation_cost,
                    step_cost=innovation_cost,
                    confidence=_confidence_from_cost(innovation_cost),
                    innovation_history=(innovation_cost,),
                )
            )
        self._hypotheses = sorted(seeded, key=lambda item: item.cumulative_cost)[: self.beam_width]
        best = self._hypotheses[0]
        return _estimate_for_observation(first_observation, best.state, best.confidence, self.forward_model, hypothesis_count=len(self._hypotheses))

    def step(self, observation: MagneticTrackingObservation) -> VesselTrackEstimate:
        if not self._hypotheses:
            raise RuntimeError("Tracker must be initialized before calling step().")

        expanded: list[_TrackingHypothesis] = []
        for hypothesis in self._hypotheses:
            dt_seconds = _delta_seconds(hypothesis.state.timestamp, observation.sensor_state.timestamp)
            predicted_state = propagate_constant_velocity(hypothesis.state, dt_seconds)
            proposed_states = _propose_measurement_branches(
                predicted_state,
                observation,
                self.forward_model,
                moment_blend_alpha=self.moment_blend_alpha,
            )
            for proposed_state in proposed_states:
                estimate = _estimate_for_observation(observation, proposed_state, hypothesis.confidence, self.forward_model)
                innovation_cost = abs(estimate.innovation_nt)
                motion_cost = _motion_cost(hypothesis.state, proposed_state)
                innovation_history = _append_history(hypothesis.innovation_history, innovation_cost, self.innovation_window)
                rolling_cost = _rolling_innovation_cost(innovation_history)
                step_cost = self.innovation_weight * rolling_cost + self.motion_weight * motion_cost
                expanded.append(
                    _TrackingHypothesis(
                        state=replace(proposed_state, timestamp=observation.sensor_state.timestamp),
                        cumulative_cost=hypothesis.cumulative_cost + step_cost,
                        step_cost=step_cost,
                        confidence=_confidence_from_cost(step_cost),
                        innovation_history=innovation_history,
                    )
                )

        self._hypotheses = sorted(expanded, key=lambda item: item.cumulative_cost)[: self.beam_width]
        best = self._hypotheses[0]
        return _estimate_for_observation(
            observation,
            best.state,
            best.confidence,
            self.forward_model,
            hypothesis_count=len(self._hypotheses),
            cumulative_cost=best.cumulative_cost,
            rolling_innovation_cost=_rolling_innovation_cost(best.innovation_history),
        )


def _estimate_for_observation(
    observation: MagneticTrackingObservation,
    vessel_state: VesselState,
    confidence: float,
    forward_model: DipoleMagneticForwardModel,
    hypothesis_count: int | None = None,
    cumulative_cost: float | None = None,
    rolling_innovation_cost: float | None = None,
) -> VesselTrackEstimate:
    expected_total_nt = forward_model.predict_total_field_nt(
        observation.sensor_state,
        vessel_state,
        observation.baseline_total_nt,
    )
    expected_residual_nt = expected_total_nt - observation.baseline_total_nt
    innovation_nt = observation.observed_total_nt - expected_total_nt
    metadata = {
        "track_id": observation.track_id,
        "range_to_sensor_m": surface_range_m(
            observation.sensor_state.latitude_deg,
            observation.sensor_state.longitude_deg,
            vessel_state.latitude_deg,
            vessel_state.longitude_deg,
        ),
    }
    if hypothesis_count is not None:
        metadata["hypothesis_count"] = hypothesis_count
    if cumulative_cost is not None:
        metadata["cumulative_cost"] = cumulative_cost
    if rolling_innovation_cost is not None:
        metadata["rolling_innovation_cost"] = rolling_innovation_cost
    return VesselTrackEstimate(
        vessel_state=replace(vessel_state),
        expected_total_nt=expected_total_nt,
        expected_residual_nt=expected_residual_nt,
        innovation_nt=innovation_nt,
        confidence=confidence,
        metadata=metadata,
    )


def _propose_measurement_branches(
    predicted_state: VesselState,
    observation: MagneticTrackingObservation,
    forward_model: DipoleMagneticForwardModel,
    moment_blend_alpha: float = 0.15,
) -> list[VesselState]:
    slant_range_m = _infer_slant_range(forward_model, observation, predicted_state)
    vertical_separation_m = max(observation.sensor_state.altitude_m + predicted_state.depth_m, 1.0)
    horizontal_range_m = max((max(slant_range_m**2 - vertical_separation_m**2, 0.0)) ** 0.5, forward_model.minimum_range_m)

    current_surface_range_m = surface_range_m(
        observation.sensor_state.latitude_deg,
        observation.sensor_state.longitude_deg,
        predicted_state.latitude_deg,
        predicted_state.longitude_deg,
    )
    if current_surface_range_m <= forward_model.minimum_range_m:
        base_bearing_deg = (float(observation.sensor_state.heading_deg or predicted_state.heading_deg) - 90.0) % 360.0
    else:
        base_bearing_deg = bearing_deg(
            observation.sensor_state.latitude_deg,
            observation.sensor_state.longitude_deg,
            predicted_state.latitude_deg,
            predicted_state.longitude_deg,
        )

    sensor_heading_deg = float(observation.sensor_state.heading_deg or base_bearing_deg)
    relative_bearing_deg = ((base_bearing_deg - sensor_heading_deg + 540.0) % 360.0) - 180.0
    if abs(relative_bearing_deg) < 20.0:
        side_center_deg = (sensor_heading_deg + (90.0 if relative_bearing_deg >= 0.0 else -90.0)) % 360.0
    else:
        side_center_deg = base_bearing_deg
    candidate_bearings = [
        side_center_deg,
        (side_center_deg - 15.0) % 360.0,
        (side_center_deg + 15.0) % 360.0,
    ]
    inferred_moment_am2 = _infer_magnetic_moment(forward_model, observation, predicted_state, slant_range_m)
    proposed: list[VesselState] = []
    for bearing_to_target_deg in candidate_bearings:
        latitude_deg, longitude_deg = project_position(
            observation.sensor_state.latitude_deg,
            observation.sensor_state.longitude_deg,
            bearing_to_target_deg,
            horizontal_range_m,
        )
        updated_speed_mps = predicted_state.speed_mps
        updated_heading_deg = predicted_state.heading_deg
        dt_seconds = _delta_seconds(predicted_state.timestamp, observation.sensor_state.timestamp)
        if dt_seconds > 0.0:
            traveled_m = surface_range_m(predicted_state.latitude_deg, predicted_state.longitude_deg, latitude_deg, longitude_deg)
            updated_speed_mps = traveled_m / dt_seconds
            updated_heading_deg = bearing_deg(
                predicted_state.latitude_deg,
                predicted_state.longitude_deg,
                latitude_deg,
                longitude_deg,
            )
        proposed.append(
            replace(
                predicted_state,
                latitude_deg=latitude_deg,
                longitude_deg=longitude_deg,
                speed_mps=updated_speed_mps,
                heading_deg=updated_heading_deg,
                magnetic_moment_am2=_blend_moment(predicted_state.magnetic_moment_am2, inferred_moment_am2, moment_blend_alpha),
                metadata={
                    **predicted_state.metadata,
                    "track_side": _track_side(sensor_heading_deg, bearing_to_target_deg),
                },
            )
        )
    return proposed


def _motion_cost(previous_state: VesselState, next_state: VesselState) -> float:
    position_change_m = surface_range_m(
        previous_state.latitude_deg,
        previous_state.longitude_deg,
        next_state.latitude_deg,
        next_state.longitude_deg,
    )
    heading_change_deg = abs(next_state.heading_deg - previous_state.heading_deg)
    side_switch_penalty = 0.0
    if previous_state.metadata.get("track_side") is not None and next_state.metadata.get("track_side") is not None:
        if previous_state.metadata.get("track_side") != next_state.metadata.get("track_side"):
            side_switch_penalty = 5000.0
    return position_change_m + 10.0 * heading_change_deg + side_switch_penalty


def _confidence_from_cost(cost: float) -> float:
    return max(0.0, min(1.0, 1.0 / (1.0 + max(cost, 0.0))))


def _blend_moment(previous_moment_am2: float, inferred_moment_am2: float, alpha: float) -> float:
    alpha = max(0.0, min(1.0, alpha))
    return max((1.0 - alpha) * previous_moment_am2 + alpha * inferred_moment_am2, 1.0)


def _track_side(sensor_heading_deg: float, target_bearing_deg: float) -> str:
    relative_bearing_deg = ((target_bearing_deg - sensor_heading_deg + 540.0) % 360.0) - 180.0
    return "starboard" if relative_bearing_deg >= 0.0 else "port"


def _infer_slant_range(
    forward_model: DipoleMagneticForwardModel,
    observation: MagneticTrackingObservation,
    vessel_state: VesselState,
) -> float:
    if isinstance(forward_model, GeometryAwareDipoleMagneticForwardModel):
        return forward_model.infer_slant_range_m(
            observation.residual_total_nt,
            vessel_state,
            sensor_state=observation.sensor_state,
        )
    return forward_model.infer_slant_range_m(observation.residual_total_nt, vessel_state)


def _infer_magnetic_moment(
    forward_model: DipoleMagneticForwardModel,
    observation: MagneticTrackingObservation,
    vessel_state: VesselState,
    slant_range_m: float,
) -> float:
    if isinstance(forward_model, GeometryAwareDipoleMagneticForwardModel):
        return forward_model.infer_magnetic_moment_am2(
            observation.residual_total_nt,
            slant_range_m,
            sensor_state=observation.sensor_state,
            vessel_state=vessel_state,
        )
    return forward_model.infer_magnetic_moment_am2(observation.residual_total_nt, slant_range_m)


def _append_history(history: tuple[float, ...], innovation_cost: float, window: int) -> tuple[float, ...]:
    updated = history + (innovation_cost,)
    if len(updated) > window:
        updated = updated[-window:]
    return updated


def _rolling_innovation_cost(history: tuple[float, ...]) -> float:
    if not history:
        return 0.0
    mean_cost = sum(history) / len(history)
    worst_cost = max(history)
    return 0.7 * mean_cost + 0.3 * worst_cost
