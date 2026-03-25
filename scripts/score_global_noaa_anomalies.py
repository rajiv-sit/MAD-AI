from __future__ import annotations

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from mad_ai.inference import score_global_dataset


def main() -> None:
    input_path = Path("data/processed/global_noaa/global_wmm_grid.csv")
    summary_path = Path("outputs/evaluation/global_noaa_anomaly_summary.json")

    data = pd.read_csv(input_path)
    scored, summary = score_global_dataset(data)
    scored.to_csv(input_path, index=False)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Updated scored global NOAA grid at {input_path}")
    print(f"Saved global anomaly summary to {summary_path}")


if __name__ == "__main__":
    main()
