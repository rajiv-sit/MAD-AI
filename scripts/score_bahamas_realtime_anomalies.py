from __future__ import annotations

import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mad_ai.ingest import load_bahamas_mad_ascii
from mad_ai.inference.bahamas_realtime import score_bahamas_realtime
from mad_ai.wmm import AnalyticMagneticModel, WMMMagneticModel


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python scripts\\score_bahamas_realtime_anomalies.py <input_asc> [output_csv] [summary_json] [backend] [reference_mode]"
        )

    input_asc = Path(sys.argv[1])
    output_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/processed/bahamas/bahamas_mad_scored.csv")
    summary_json = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("outputs/evaluation/bahamas_mad_summary.json")
    backend = str(sys.argv[4]).strip().lower() if len(sys.argv) > 4 else "wmm"
    reference_mode = str(sys.argv[5]).strip().lower() if len(sys.argv) > 5 else "stable_window"

    started_at = time.perf_counter()
    raw = load_bahamas_mad_ascii(input_asc)
    magnetic_model = AnalyticMagneticModel() if backend == "analytic" else WMMMagneticModel(cache_path="data/cache/wmm_cache.json")
    scored, summary = score_bahamas_realtime(raw, magnetic_model=magnetic_model, reference_window_mode=reference_mode)
    summary["input_asc"] = str(input_asc)
    summary["output_csv"] = str(output_csv)
    summary["magnetic_backend"] = backend
    summary["reference_window_mode"] = reference_mode
    summary["runtime_seconds"] = time.perf_counter() - started_at

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_csv, index=False)

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved Bahamas scored anomaly CSV to {output_csv}")
    print(f"Saved Bahamas anomaly summary to {summary_json}")


if __name__ == "__main__":
    main()
