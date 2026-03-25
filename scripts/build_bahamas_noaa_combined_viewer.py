from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from mad_ai.visualizer import CesiumGlobeViewerBuilder


def main() -> None:
    output_html = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/viewer/cesium_bahamas_noaa_combined.html")
    global_grid_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/processed/global_noaa/global_wmm_grid.csv")
    scored_csv = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("data/processed/bahamas/bahamas_mad_scored.csv")

    global_grid = pd.read_csv(global_grid_csv)
    scored = pd.read_csv(scored_csv)
    viewer_frame = _prepare_viewer_frame(scored, max_points=12000, altitude_bins=8)
    overlay_altitudes = sorted(global_grid["altitude_m"].dropna().astype(float).unique().tolist())
    viewer_frame["altitude_m"] = viewer_frame["altitude_m"].map(lambda value: _nearest_altitude(float(value), overlay_altitudes))
    viewer_frame["display_mode"] = "noaa_plus_anomaly"

    component_columns = {
        "Total Field": "baseline_total_nt",
        "Declination": "baseline_declination_deg",
        "Inclination": "baseline_inclination_deg",
        "Residual": "residual_total_nt",
        "Baseline Residual Score": "baseline_residual_score",
        "Spatial Anomaly Score": "spatial_anomaly_score",
        "Temporal Anomaly Score": "temporal_anomaly_score",
        "Final Anomaly Score": "final_anomaly_score",
    }

    output = CesiumGlobeViewerBuilder(
        title="MAD-AI NOAA Baseline With Bahamas Anomaly Overlay",
        component_columns=component_columns,
        default_component="baseline_total_nt",
        anomaly_flag_column="is_anomaly",
        anomaly_score_column="final_anomaly_score",
    ).build_with_overlay_data(
        viewer_frame,
        output_html,
        overlay_data=global_grid,
    )
    print(f"Saved combined NOAA and Bahamas Cesium viewer to {output}")

def _nearest_altitude(value: float, overlay_altitudes: list[float]) -> float:
    if not overlay_altitudes:
        return value
    return min(overlay_altitudes, key=lambda candidate: abs(candidate - value))


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
    min_altitude = float(altitude.min())
    max_altitude = float(altitude.max())
    if min_altitude == max_altitude:
        frame["altitude_m"] = min_altitude
        return frame

    import numpy as np

    bin_edges = np.linspace(min_altitude, max_altitude, num=max(2, altitude_bins + 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_indices = np.clip(np.digitize(altitude, bin_edges[1:-1], right=False), 0, len(bin_centers) - 1)
    frame["altitude_m"] = bin_centers[bin_indices]
    return frame


if __name__ == "__main__":
    main()
