from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mad_ai.features import ResidualFeatureBuilder, TemporalFeatureBuilder
from mad_ai.utils.sample_data import make_sample_sensor_data
from mad_ai.visualizer import CesiumGlobeViewerBuilder
from mad_ai.wmm import WMMMagneticModel


def main() -> None:
    raw = make_sample_sensor_data()
    enriched = ResidualFeatureBuilder(WMMMagneticModel()).transform(raw)
    temporal = TemporalFeatureBuilder().transform(enriched)
    output = CesiumGlobeViewerBuilder(title="MAD-AI NOAA Magnetic Globe").build(
        temporal,
        Path("outputs/viewer/cesium_magnetic_globe.html"),
    )
    print(f"Saved Cesium globe viewer to {output}")


if __name__ == "__main__":
    main()
