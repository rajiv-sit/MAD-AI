from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from mad_ai.tracking import ConstantVelocityMagneticTracker, MultiHypothesisMagneticTracker, observations_from_bahamas_frame, surface_range_m


def main() -> None:
    scored_csv = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed/bahamas/bahamas_mad_scored.csv")
    output_json = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs/evaluation/bahamas_tracking_metrics.json")
    estimated_csv = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("outputs/evaluation/bahamas_tracking_estimates.csv")
    tracker_mode = str(sys.argv[4]).strip().lower() if len(sys.argv) > 4 else "multi"

    frame = pd.read_csv(scored_csv)
    observations = observations_from_bahamas_frame(frame)
    if not observations:
        raise SystemExit("No observations available for Bahamas tracking evaluation.")

    if tracker_mode == "single":
        tracker = ConstantVelocityMagneticTracker(initialization_strategy="magnetic_bearing_grid")
    else:
        tracker = MultiHypothesisMagneticTracker(initialization_strategy="magnetic_bearing_grid")
    estimates = [tracker.initialize(observations)]
    for observation in observations[1:]:
        estimates.append(tracker.step(observation))

    estimate_rows = []
    errors_m = []
    for observation, estimate in zip(observations, estimates, strict=True):
        reference_lat = observation.metadata.get("reference_vessel_latitude_deg")
        reference_lon = observation.metadata.get("reference_vessel_longitude_deg")
        if reference_lat is None or reference_lon is None:
            continue
        error_m = surface_range_m(
            estimate.vessel_state.latitude_deg,
            estimate.vessel_state.longitude_deg,
            float(reference_lat),
            float(reference_lon),
        )
        errors_m.append(error_m)
        estimate_rows.append(
            {
                "timestamp": observation.sensor_state.timestamp.isoformat() if observation.sensor_state.timestamp else None,
                "estimated_vessel_latitude_deg": estimate.vessel_state.latitude_deg,
                "estimated_vessel_longitude_deg": estimate.vessel_state.longitude_deg,
                "reference_vessel_latitude_deg": float(reference_lat),
                "reference_vessel_longitude_deg": float(reference_lon),
                "tracking_error_m": error_m,
                "expected_total_nt": estimate.expected_total_nt,
                "expected_residual_nt": estimate.expected_residual_nt,
                "innovation_nt": estimate.innovation_nt,
                "confidence": estimate.confidence,
                "estimated_magnetic_moment_am2": estimate.vessel_state.magnetic_moment_am2,
                "hypothesis_count": estimate.metadata.get("hypothesis_count"),
                "cumulative_cost": estimate.metadata.get("cumulative_cost"),
            }
        )

    estimated_frame = pd.DataFrame(estimate_rows)
    estimated_csv.parent.mkdir(parents=True, exist_ok=True)
    estimated_frame.to_csv(estimated_csv, index=False)

    metrics = {
        "rows": int(len(estimated_frame)),
        "mean_tracking_error_m": float(estimated_frame["tracking_error_m"].mean()) if not estimated_frame.empty else None,
        "median_tracking_error_m": float(estimated_frame["tracking_error_m"].median()) if not estimated_frame.empty else None,
        "max_tracking_error_m": float(estimated_frame["tracking_error_m"].max()) if not estimated_frame.empty else None,
        "mean_innovation_nt": float(estimated_frame["innovation_nt"].mean()) if not estimated_frame.empty else None,
        "mean_estimated_magnetic_moment_am2": float(estimated_frame["estimated_magnetic_moment_am2"].mean()) if not estimated_frame.empty else None,
        "estimated_csv": str(estimated_csv),
        "initialization_strategy": "magnetic_bearing_grid",
        "tracker_mode": tracker_mode,
        "note": "This is a first-pass magnetic tracker. The estimate path uses magnetic residuals and aircraft geometry; the reference vessel track is used for evaluation only.",
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved Bahamas tracking metrics to {output_json}")
    print(f"Saved Bahamas tracking estimates to {estimated_csv}")


if __name__ == "__main__":
    main()
