from __future__ import annotations

import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from mad_ai.ingest import load_bahamas_mad_ascii
from mad_ai.inference.bahamas_realtime import score_bahamas_realtime
from mad_ai.tracking import MultiHypothesisMagneticTracker, observations_from_bahamas_frame, surface_range_m
from mad_ai.wmm import AnalyticMagneticModel, WMMMagneticModel


def main() -> None:
    input_asc = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("inspection/external_mad_repo/data/raw/mad_data.asc")
    output_json = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else Path("outputs/evaluation/bahamas_tracking_backend_comparison_subset.json")
    )
    row_limit = int(sys.argv[3]) if len(sys.argv) > 3 else 2000

    raw = load_bahamas_mad_ascii(input_asc).head(row_limit).copy()
    report = {
        "input_asc": str(input_asc),
        "row_limit": row_limit,
        "analytic": _score_and_evaluate(raw, backend="analytic"),
        "wmm": _score_and_evaluate(raw, backend="wmm"),
    }
    report["delta_mean_tracking_error_m"] = (
        float(report["wmm"]["mean_tracking_error_m"]) - float(report["analytic"]["mean_tracking_error_m"])
        if report["analytic"]["mean_tracking_error_m"] is not None and report["wmm"]["mean_tracking_error_m"] is not None
        else None
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved Bahamas backend comparison to {output_json}")


def _score_and_evaluate(raw: pd.DataFrame, backend: str) -> dict[str, object]:
    started = time.perf_counter()
    prepared_raw = raw.copy()
    if "timestamp" in prepared_raw.columns:
        prepared_raw["timestamp"] = pd.to_datetime(prepared_raw["timestamp"])
    magnetic_model = AnalyticMagneticModel() if backend == "analytic" else WMMMagneticModel(cache_path="data/cache/wmm_cache.json")
    scored, summary = score_bahamas_realtime(prepared_raw, magnetic_model=magnetic_model, reference_window_mode="stable_window")
    observations = observations_from_bahamas_frame(scored)
    tracker = MultiHypothesisMagneticTracker(initialization_strategy="magnetic_bearing_grid")
    estimates = [tracker.initialize(observations)]
    for observation in observations[1:]:
        estimates.append(tracker.step(observation))

    errors_m = []
    for observation, estimate in zip(observations, estimates, strict=True):
        reference_lat = observation.metadata.get("reference_vessel_latitude_deg")
        reference_lon = observation.metadata.get("reference_vessel_longitude_deg")
        if reference_lat is None or reference_lon is None:
            continue
        errors_m.append(
            surface_range_m(
                estimate.vessel_state.latitude_deg,
                estimate.vessel_state.longitude_deg,
                float(reference_lat),
                float(reference_lon),
            )
        )

    return {
        "backend": backend,
        "rows": int(len(scored)),
        "runtime_seconds": time.perf_counter() - started,
        "mean_tracking_error_m": float(pd.Series(errors_m).mean()) if errors_m else None,
        "median_tracking_error_m": float(pd.Series(errors_m).median()) if errors_m else None,
        "mean_innovation_nt": float(pd.Series([estimate.innovation_nt for estimate in estimates]).mean()) if estimates else None,
        "fused_anomaly_count": int(summary.get("fused_anomaly_count", 0)),
        "baseline_anomaly_count": int(summary.get("baseline_anomaly_count", 0)),
    }


if __name__ == "__main__":
    main()
