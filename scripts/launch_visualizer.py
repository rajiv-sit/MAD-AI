from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mad_ai.features import ResidualFeatureBuilder, TemporalFeatureBuilder
from mad_ai.utils.sample_data import make_sample_sensor_data
from mad_ai.visualizer import VisualizerApp
from mad_ai.wmm import WMMMagneticModel


def main() -> None:
    raw = make_sample_sensor_data()
    enriched = ResidualFeatureBuilder(WMMMagneticModel()).transform(raw)
    temporal = TemporalFeatureBuilder().transform(enriched)
    outputs = VisualizerApp().show(temporal, Path("outputs/figures"))
    for output in outputs:
        print(f"Saved visualizer artifact to {output}")


if __name__ == "__main__":
    main()
