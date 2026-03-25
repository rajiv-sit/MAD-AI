from __future__ import annotations

from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import numpy as np

from mad_ai.visualizer import CesiumGlobeViewerBuilder


def _prepare_viewer_frame(scored: pd.DataFrame, max_points: int, altitude_bins: int = 8) -> pd.DataFrame:
    reduced = _downsample_for_viewer(scored, max_points=max_points)
    return _quantize_altitudes_for_viewer(reduced, altitude_bins=altitude_bins)


def _downsample_for_viewer(scored: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(scored) <= max_points:
        return scored

    anomaly_points = scored[scored["is_anomaly"].astype(bool)].copy()
    anomaly_points = anomaly_points.sort_values("final_anomaly_score", ascending=False)
    keep_anomalies = anomaly_points.head(min(len(anomaly_points), max_points // 3))

    reference_points = scored[scored["realtime_phase"] == "reference"].copy()
    keep_reference = reference_points.iloc[:: max(1, len(reference_points) // max(1, max_points // 6))]

    remaining_budget = max_points - len(keep_anomalies) - len(keep_reference)
    remaining_budget = max(remaining_budget, max_points // 4)
    stride = max(1, len(scored) // remaining_budget)
    keep_uniform = scored.iloc[::stride].copy()

    reduced = pd.concat([keep_uniform, keep_reference, keep_anomalies], ignore_index=False)
    reduced = reduced.sort_index().drop_duplicates().reset_index(drop=True)
    if len(reduced) > max_points:
        reduced = reduced.iloc[:max_points].copy()
    return reduced


def _quantize_altitudes_for_viewer(scored: pd.DataFrame, altitude_bins: int) -> pd.DataFrame:
    if "altitude_m" not in scored.columns or altitude_bins <= 1 or len(scored) == 0:
        return scored

    frame = scored.copy()
    altitude = frame["altitude_m"].astype(float).to_numpy()
    min_altitude = float(np.nanmin(altitude))
    max_altitude = float(np.nanmax(altitude))
    if np.isclose(min_altitude, max_altitude):
        frame["altitude_m"] = min_altitude
        return frame

    bin_edges = np.linspace(min_altitude, max_altitude, num=max(2, altitude_bins + 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_indices = np.clip(np.digitize(altitude, bin_edges[1:-1], right=False), 0, len(bin_centers) - 1)
    frame["altitude_m"] = bin_centers[bin_indices]
    return frame


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python scripts\\build_bahamas_realtime_cesium_viewer.py <input_asc> [output_html] [backend] [max_points] [altitude_bins]"
        )

    input_asc = Path(sys.argv[1])
    output_html = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs/viewer/cesium_bahamas_mad_globe.html")
    backend = str(sys.argv[3]).strip().lower() if len(sys.argv) > 3 else "wmm"
    max_points = int(sys.argv[4]) if len(sys.argv) > 4 else 15000
    altitude_bins = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    scored_csv = Path("data/processed/bahamas/bahamas_mad_scored.csv")
    summary_json = Path("outputs/evaluation/bahamas_mad_summary.json")

    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[0] / "score_bahamas_realtime_anomalies.py"),
            str(input_asc),
            str(scored_csv),
            str(summary_json),
            backend,
        ],
        check=True,
    )

    scored = pd.read_csv(scored_csv)
    viewer_frame = _prepare_viewer_frame(scored, max_points=max_points, altitude_bins=altitude_bins)
    viewer = CesiumGlobeViewerBuilder(
        title="MAD-AI Bahamas Flight Realtime Anomaly Globe",
        component_columns={
            "Observed Total": "observed_total_nt",
            "Baseline Total": "baseline_total_nt",
            "Residual Total": "residual_total_nt",
            "Baseline Residual Score": "baseline_residual_score",
            "Spatial Anomaly Score": "spatial_anomaly_score",
            "Temporal Anomaly Score": "temporal_anomaly_score",
            "Final Anomaly Score": "final_anomaly_score",
        },
        default_component="final_anomaly_score",
        anomaly_flag_column="is_anomaly",
        anomaly_score_column="final_anomaly_score",
    )
    output = viewer.build(viewer_frame, output_html)
    print(f"Saved Bahamas realtime Cesium globe viewer to {output}")


if __name__ == "__main__":
    main()
