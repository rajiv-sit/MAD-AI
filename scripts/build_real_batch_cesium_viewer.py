from __future__ import annotations

from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from mad_ai.visualizer import CesiumGlobeViewerBuilder


def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config/real_batch.yaml")
    output_html = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs/viewer/cesium_real_batch_globe.html")
    input_dir = Path("data/raw/real_batch/anomalous_eval")
    scored_csv = Path("data/processed/real_batch_scored/real_batch_scored.csv")
    summary_json = Path("outputs/evaluation/real_batch_scored_summary.json")

    if not Path("outputs/calibration/real_batch_thresholds.json").exists():
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parents[0] / "evaluate_real_batch_models.py"), str(config_path)],
            check=True,
        )
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[0] / "score_real_batch_folder.py"),
            str(input_dir),
            str(config_path),
            str(scored_csv),
            str(summary_json),
        ],
        check=True,
    )

    scored = pd.read_csv(scored_csv)
    viewer = CesiumGlobeViewerBuilder(
        title="MAD-AI Real Batch Globe",
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
    output = viewer.build(scored, output_html)
    print(f"Saved real batch Cesium globe viewer to {output}")


if __name__ == "__main__":
    main()
