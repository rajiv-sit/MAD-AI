from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mad_ai.utils import ArtifactStore, build_processed_artifacts
from mad_ai.utils.sample_data import make_sample_sensor_data
from mad_ai.wmm import WMMMagneticModel


def main() -> None:
    store = ArtifactStore("data/processed")
    wmm_model = WMMMagneticModel(cache_path="data/cache/wmm_cache.json")
    raw = make_sample_sensor_data()
    paths = build_processed_artifacts(raw=raw, store=store, wmm_model=wmm_model)
    for path in paths.values():
        print(f"Saved artifact to {path}")


if __name__ == "__main__":
    main()
