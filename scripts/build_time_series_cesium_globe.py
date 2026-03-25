from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mad_ai.utils.sample_data import make_time_series_global_wmm_grid
from mad_ai.visualizer import CesiumGlobeViewerBuilder
from mad_ai.wmm import WMMMagneticModel


def main() -> None:
    timestamps = [
        datetime(2026, 3, 24, 0, 0, 0),
        datetime(2026, 3, 24, 6, 0, 0),
        datetime(2026, 3, 24, 12, 0, 0),
        datetime(2026, 3, 24, 18, 0, 0),
    ]
    data = make_time_series_global_wmm_grid(
        timestamps=timestamps,
        lat_step_deg=20.0,
        lon_step_deg=20.0,
        altitude_m=0.0,
        magnetic_model=WMMMagneticModel(cache_path="data/cache/wmm_cache.json"),
    )
    output = CesiumGlobeViewerBuilder(title="MAD-AI Time-Series Global Magnetic Globe").build(
        data,
        Path("outputs/viewer/cesium_time_series_global_magnetic_globe.html"),
    )
    print(f"Saved time-series Cesium globe viewer to {output}")


if __name__ == "__main__":
    main()
