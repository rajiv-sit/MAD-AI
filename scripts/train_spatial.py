from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mad_ai.datasets import SpatialGridBuilder
from mad_ai.features import ResidualFeatureBuilder
from mad_ai.models.spatial import CNNAnomalyModel
from mad_ai.utils.sample_data import make_sample_sensor_data
from mad_ai.wmm import WMMMagneticModel


def main() -> None:
    data = make_sample_sensor_data()
    enriched = ResidualFeatureBuilder(WMMMagneticModel()).transform(data)
    grid = SpatialGridBuilder().build(enriched)
    model = CNNAnomalyModel()
    model.train(np.stack([grid, grid], axis=0))
    output = Path("outputs/models/spatial_autoencoder.pt")
    model.save(output)
    print(f"Saved spatial model to {output}")


if __name__ == "__main__":
    main()
