from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from mad_ai.viz import RealtimeAnomalyDashboardVisualizer


def main() -> None:
    scored_csv = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed/bahamas/bahamas_mad_scored.csv")
    summary_json = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs/evaluation/bahamas_mad_summary.json")
    output_png = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("outputs/figures/bahamas/bahamas_realtime_anomaly_dashboard.png")

    frame = pd.read_csv(scored_csv)
    summary = json.loads(summary_json.read_text(encoding="utf-8")) if summary_json.exists() else {}
    threshold = summary.get("threshold")
    if threshold is not None:
        frame["final_anomaly_threshold"] = float(threshold)

    output = RealtimeAnomalyDashboardVisualizer(
        threshold_column="final_anomaly_threshold" if "final_anomaly_threshold" in frame.columns else None
    ).render(frame, output_png)
    print(f"Saved Bahamas realtime anomaly dashboard to {output}")


if __name__ == "__main__":
    main()
