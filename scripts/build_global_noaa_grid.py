from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mad_ai.utils import ArtifactStore
from mad_ai.utils.sample_data import make_multi_altitude_global_wmm_grid
from mad_ai.wmm import WMMMagneticModel


def main() -> None:
    store = ArtifactStore("data/processed/global_noaa")
    altitudes_m = [0.0, 1000.0, 5000.0, 10000.0]
    lat_step_deg = 5.0
    lon_step_deg = 5.0
    grid = make_multi_altitude_global_wmm_grid(
        altitudes_m=altitudes_m,
        lat_step_deg=lat_step_deg,
        lon_step_deg=lon_step_deg,
        timestamp=datetime(2026, 3, 24),
        magnetic_model=WMMMagneticModel(cache_path="data/cache/wmm_cache.json"),
    )
    csv_path = store.save_dataframe("global_wmm_grid", grid)
    meta_path = store.save_json(
        "global_wmm_grid_metadata",
        {
            "rows": int(len(grid)),
            "lat_step_deg": lat_step_deg,
            "lon_step_deg": lon_step_deg,
            "altitudes_m": altitudes_m,
            "timestamp": "2026-03-24T00:00:00",
            "source": "NOAA/WMM-derived backend with official NOAA coefficient package present in repo",
        },
    )
    print(f"Saved global grid to {csv_path}")
    print(f"Saved global grid metadata to {meta_path}")


if __name__ == "__main__":
    main()
