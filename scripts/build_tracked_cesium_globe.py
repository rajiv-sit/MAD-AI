from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mad_ai.features import ResidualFeatureBuilder, TemporalFeatureBuilder
from mad_ai.utils.sample_data import make_tracked_sensor_data
from mad_ai.visualizer import CesiumGlobeViewerBuilder
from mad_ai.wmm import WMMMagneticModel


def main() -> None:
    raw = make_tracked_sensor_data()
    enriched = ResidualFeatureBuilder(WMMMagneticModel(cache_path="data/cache/wmm_cache.json")).transform(raw)
    temporal = TemporalFeatureBuilder().transform(enriched)

    viewer = CesiumGlobeViewerBuilder(
        title="MAD-AI Tracked Magnetic Globe",
        component_columns={
            "Observed Total": "observed_total_nt",
            "Baseline Total": "baseline_total_nt",
            "Residual Total": "residual_total_nt",
            "Observed Declination": "observed_declination_deg",
            "Baseline Declination": "baseline_declination_deg",
        },
        default_component="observed_total_nt",
    )
    output = viewer.build(temporal, Path("outputs/viewer/cesium_tracked_magnetic_globe.html"))
    print(f"Saved tracked Cesium globe viewer to {output}")


if __name__ == "__main__":
    main()
