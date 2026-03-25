from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mad_ai.features import ResidualFeatureBuilder, TemporalFeatureBuilder
from mad_ai.ingest import CsvSensorIngestor
from mad_ai.visualizer import CesiumGlobeViewerBuilder
from mad_ai.wmm import WMMMagneticModel


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts\\build_observed_csv_cesium_viewer.py <input_csv> [output_html]")

    input_csv = Path(sys.argv[1])
    output_html = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs/viewer/cesium_observed_magnetic_globe.html")

    raw = CsvSensorIngestor().load(input_csv)
    enriched = ResidualFeatureBuilder(WMMMagneticModel(cache_path="data/cache/wmm_cache.json")).transform(raw)
    temporal = TemporalFeatureBuilder().transform(enriched)

    viewer = CesiumGlobeViewerBuilder(
        title="MAD-AI Observed vs Baseline Magnetic Globe",
        component_columns={
            "Observed Total": "observed_total_nt",
            "Baseline Total": "baseline_total_nt",
            "Residual Total": "residual_total_nt",
            "Observed Declination": "observed_declination_deg",
            "Baseline Declination": "baseline_declination_deg",
            "Residual Declination": "residual_declination_deg",
        },
        default_component="observed_total_nt",
    )
    output = viewer.build(temporal, output_html)
    print(f"Saved observed-data Cesium globe viewer to {output}")


if __name__ == "__main__":
    main()
