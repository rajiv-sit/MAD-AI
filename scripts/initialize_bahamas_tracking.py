from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from mad_ai.tracking import initialize_vessel_state_candidates, observations_from_bahamas_frame


def main() -> None:
    scored_csv = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed/bahamas/bahamas_mad_scored.csv")
    output_json = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs/evaluation/bahamas_tracking_initialization.json")
    top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    frame = pd.read_csv(scored_csv)
    observations = observations_from_bahamas_frame(frame)
    candidates = initialize_vessel_state_candidates(observations, top_k=top_k)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "latitude_deg": candidate.latitude_deg,
            "longitude_deg": candidate.longitude_deg,
            "speed_mps": candidate.speed_mps,
            "heading_deg": candidate.heading_deg,
            "depth_m": candidate.depth_m,
            "magnetic_moment_am2": candidate.magnetic_moment_am2,
            "timestamp": candidate.timestamp.isoformat() if candidate.timestamp else None,
            "metadata": candidate.metadata,
        }
        for candidate in candidates
    ]
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved Bahamas tracking initialization candidates to {output_json}")


if __name__ == "__main__":
    main()
