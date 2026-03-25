from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mad_ai.features import ResidualFeatureBuilder, TemporalFeatureBuilder
from mad_ai.utils.sample_data import make_sample_sensor_data
from mad_ai.viz import (
    AnomalyScatter3DVisualizer,
    ContourMapVisualizer,
    HeatmapVisualizer,
    Surface3DVisualizer,
)
from mad_ai.wmm import WMMMagneticModel


def main() -> None:
    raw = make_sample_sensor_data()
    enriched = ResidualFeatureBuilder(WMMMagneticModel()).transform(raw)
    temporal = TemporalFeatureBuilder().transform(enriched)
    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        HeatmapVisualizer(value_column="baseline_total_nt", title="Baseline Total Field Heatmap").render(
            temporal, output_dir / "baseline_heatmap_2d.png"
        ),
        HeatmapVisualizer(value_column="residual_total_nt", title="Residual Field Heatmap", cmap="coolwarm").render(
            temporal, output_dir / "residual_heatmap_2d.png"
        ),
        ContourMapVisualizer(value_column="baseline_total_nt", title="Baseline Total Field Contours").render(
            temporal, output_dir / "baseline_contours_2d.png"
        ),
        Surface3DVisualizer(value_column="baseline_total_nt", title="Baseline Total Field Surface").render(
            temporal, output_dir / "baseline_surface_3d.png"
        ),
        Surface3DVisualizer(value_column="residual_total_nt", title="Residual Field Surface", cmap="coolwarm").render(
            temporal, output_dir / "residual_surface_3d.png"
        ),
        AnomalyScatter3DVisualizer(residual_column="residual_total_nt", threshold=250.0).render(
            temporal, output_dir / "anomaly_scatter_3d.png"
        ),
    ]

    for output in outputs:
        print(f"Saved visualization to {output}")


if __name__ == "__main__":
    main()
