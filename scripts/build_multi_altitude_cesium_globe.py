from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mad_ai.utils.sample_data import make_multi_altitude_global_wmm_grid
from mad_ai.visualizer import CesiumGlobeViewerBuilder
from mad_ai.wmm import WMMMagneticModel


def main() -> None:
    data = make_multi_altitude_global_wmm_grid(
        altitudes_m=[0.0, 10000.0, 30000.0],
        timestamp=datetime(2026, 3, 24, 0, 0, 0),
        lat_step_deg=20.0,
        lon_step_deg=20.0,
        magnetic_model=WMMMagneticModel(cache_path="data/cache/wmm_cache.json"),
    )
    output = CesiumGlobeViewerBuilder(title="MAD-AI Multi-Altitude Magnetic Globe").build(
        data,
        Path("outputs/viewer/cesium_multi_altitude_magnetic_globe.html"),
    )
    print(f"Saved multi-altitude Cesium globe viewer to {output}")


if __name__ == "__main__":
    main()
