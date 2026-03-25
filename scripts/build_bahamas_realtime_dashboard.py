from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from mad_ai.viz import BahamasRealtimeDashboardHTMLBuilder


def main() -> None:
    scored_csv = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed/bahamas/bahamas_mad_scored.csv")
    summary_json = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs/evaluation/bahamas_mad_summary.json")
    output_html = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("outputs/viewer/bahamas_realtime_dashboard.html")
    globe_html = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("cesium_bahamas_noaa_combined.html")
    tracking_estimates_csv = (
        Path(sys.argv[5]) if len(sys.argv) > 5 else Path("outputs/evaluation/bahamas_tracking_estimates_multi.csv")
    )

    frame = pd.read_csv(scored_csv)
    if tracking_estimates_csv.exists():
        frame = _merge_tracking_estimates(frame, pd.read_csv(tracking_estimates_csv))
    summary = json.loads(summary_json.read_text(encoding="utf-8")) if summary_json.exists() else {}
    output = BahamasRealtimeDashboardHTMLBuilder(globe_relative_path=globe_html.as_posix()).render(
        frame,
        output_html,
        summary=summary,
    )
    print(f"Saved Bahamas realtime dashboard to {output}")


def _merge_tracking_estimates(frame: pd.DataFrame, estimates: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in frame.columns or "timestamp" not in estimates.columns:
        return frame

    estimate_columns = [
        "timestamp",
        "estimated_vessel_latitude_deg",
        "estimated_vessel_longitude_deg",
        "tracking_error_m",
        "confidence",
    ]
    available = [column for column in estimate_columns if column in estimates.columns]
    merged = frame.copy()
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=False, format="ISO8601")
    estimate_frame = estimates[available].copy()
    estimate_frame["timestamp"] = pd.to_datetime(estimate_frame["timestamp"], utc=False, format="ISO8601")
    merged = merged.merge(estimate_frame, on="timestamp", how="left")
    merged["timestamp"] = merged["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return merged


if __name__ == "__main__":
    main()
