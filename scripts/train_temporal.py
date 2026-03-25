from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mad_ai.datasets import TemporalSequenceBuilder
from mad_ai.features import ResidualFeatureBuilder, TemporalFeatureBuilder
from mad_ai.models.temporal import LSTMAnomalyModel
from mad_ai.utils.sample_data import make_sample_sensor_data
from mad_ai.wmm import WMMMagneticModel


def main() -> None:
    data = make_sample_sensor_data()
    enriched = ResidualFeatureBuilder(WMMMagneticModel()).transform(data)
    temporal = TemporalFeatureBuilder().transform(enriched)
    sequences = TemporalSequenceBuilder().build(temporal)
    model = LSTMAnomalyModel()
    model.train(sequences)
    output = Path("outputs/models/temporal_autoencoder.pt")
    model.save(output)
    print(f"Saved temporal model to {output}")


if __name__ == "__main__":
    main()
