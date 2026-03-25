from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mad_ai.features import ResidualFeatureBuilder, TemporalFeatureBuilder
from mad_ai.utils.sample_data import make_sample_sensor_data
from mad_ai.viz import AnomalyScatter3DVisualizer, ContourMapVisualizer, Globe3DVisualizer, HeatmapVisualizer, Surface3DVisualizer
from mad_ai.wmm import WMMMagneticModel


def main() -> None:
    raw = make_sample_sensor_data()
    enriched = ResidualFeatureBuilder(WMMMagneticModel()).transform(raw)
    temporal = TemporalFeatureBuilder().transform(enriched)
    output_dir = Path("outputs/figures/noaa_wmm")
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        HeatmapVisualizer(value_column="baseline_total_nt", title="NOAA WMM2025 Baseline Total Field Heatmap").render(
            temporal, output_dir / "noaa_baseline_heatmap_2d.png"
        ),
        HeatmapVisualizer(value_column="residual_total_nt", title="Residual vs NOAA WMM2025 Baseline", cmap="coolwarm").render(
            temporal, output_dir / "noaa_residual_heatmap_2d.png"
        ),
        ContourMapVisualizer(value_column="baseline_total_nt", title="NOAA WMM2025 Baseline Contours").render(
            temporal, output_dir / "noaa_baseline_contours_2d.png"
        ),
        Surface3DVisualizer(value_column="baseline_total_nt", title="NOAA WMM2025 Baseline Surface").render(
            temporal, output_dir / "noaa_baseline_surface_3d.png"
        ),
        Surface3DVisualizer(value_column="residual_total_nt", title="Residual Surface vs NOAA WMM2025", cmap="coolwarm").render(
            temporal, output_dir / "noaa_residual_surface_3d.png"
        ),
        Globe3DVisualizer(value_column="baseline_total_nt", title="NOAA WMM2025 Baseline Globe").render(
            temporal, output_dir / "noaa_baseline_globe_3d.png"
        ),
        Globe3DVisualizer(value_column="residual_total_nt", title="Residual Globe vs NOAA WMM2025", cmap="coolwarm").render(
            temporal, output_dir / "noaa_residual_globe_3d.png"
        ),
        AnomalyScatter3DVisualizer(residual_column="residual_total_nt", threshold=250.0, title="Anomaly Scatter vs NOAA WMM2025").render(
            temporal, output_dir / "noaa_anomaly_scatter_3d.png"
        ),
    ]

    note = output_dir / "README.txt"
    note.write_text(
        "\n".join(
            [
                "NOAA WMM2025 reference pages:",
                "https://www.ncei.noaa.gov/products/world-magnetic-model",
                "https://www.ncei.noaa.gov/products/world-magnetic-model/wmm-coefficients",
                "",
                "These figures are generated in this repo through the installed WMM-derived backend used by mad_ai.wmm.WMMMagneticModel.",
                "For direct official NOAA coefficient-file ingestion, add a downloader/import step against the WMM2025 coefficient package.",
            ]
        ),
        encoding="utf-8",
    )
    outputs.append(note)

    for output in outputs:
        print(f"Saved NOAA WMM visualization artifact to {output}")


if __name__ == "__main__":
    main()
