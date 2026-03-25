from __future__ import annotations

from pathlib import Path
import sys
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from mad_ai.visualizer import CesiumGlobeViewerBuilder


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts\\build_observed_anomaly_cesium_viewer.py <input_csv> [output_html]")

    input_csv = Path(sys.argv[1])
    output_html = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs/viewer/cesium_observed_anomaly_globe.html")
    scored_csv = Path("data/processed/observed_scored/observed_anomaly_scored.csv")
    summary_json = Path("outputs/evaluation/observed_anomaly_summary.json")
    observed_thresholds = Path("outputs/calibration/observed_thresholds.json")
    if not observed_thresholds.exists():
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parents[0] / "evaluate_observed_residual_models.py"),
            ],
            check=True,
        )
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[0] / "score_observed_csv_anomalies.py"),
            str(input_csv),
            str(scored_csv),
            str(summary_json),
        ],
        check=True,
    )
    temporal = pd.read_csv(scored_csv)

    viewer = CesiumGlobeViewerBuilder(
        title="MAD-AI Observed Anomaly Globe",
        component_columns={
            "Observed Total": "observed_total_nt",
            "Baseline Total": "baseline_total_nt",
            "Residual Total": "residual_total_nt",
            "Spatial Anomaly Score": "spatial_anomaly_score",
            "Temporal Anomaly Score": "temporal_anomaly_score",
            "Final Anomaly Score": "final_anomaly_score",
        },
        default_component="final_anomaly_score",
        anomaly_flag_column="is_anomaly",
        anomaly_score_column="final_anomaly_score",
    )
    output = viewer.build(temporal, output_html)
    print(f"Saved observed anomaly Cesium globe viewer to {output}")


if __name__ == "__main__":
    main()
