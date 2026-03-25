from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from mad_ai.viz import ContourMapVisualizer, Globe3DVisualizer, HeatmapVisualizer, Surface3DVisualizer
from mad_ai.visualizer import CesiumGlobeViewerBuilder


def main() -> None:
    data = pd.read_csv("data/processed/global_noaa/global_wmm_grid.csv")
    figure_data = data.loc[data["altitude_m"] == data["altitude_m"].min()].copy()
    output_dir = Path("outputs/figures/global_noaa_wmm")
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        HeatmapVisualizer(value_column="baseline_total_nt", title="Global NOAA WMM Total Field Heatmap").render(
            figure_data, output_dir / "global_baseline_heatmap_2d.png"
        ),
        ContourMapVisualizer(value_column="baseline_declination_deg", title="Global NOAA WMM Declination Contours", cmap="plasma").render(
            figure_data, output_dir / "global_declination_contours_2d.png"
        ),
        Surface3DVisualizer(value_column="baseline_total_nt", title="Global NOAA WMM Total Field Surface").render(
            figure_data, output_dir / "global_baseline_surface_3d.png"
        ),
        Globe3DVisualizer(value_column="baseline_total_nt", title="Global NOAA WMM Total Field Globe").render(
            figure_data, output_dir / "global_baseline_globe_3d.png"
        ),
        Globe3DVisualizer(value_column="baseline_declination_deg", title="Global NOAA WMM Declination Globe", cmap="plasma").render(
            figure_data, output_dir / "global_declination_globe_3d.png"
        ),
        CesiumGlobeViewerBuilder(title="MAD-AI Global NOAA Magnetic Globe").build(
            data,
            Path("outputs/viewer/cesium_global_magnetic_globe.html"),
        ),
    ]

    for output in outputs:
        print(f"Saved global NOAA visualization artifact to {output}")


if __name__ == "__main__":
    main()
