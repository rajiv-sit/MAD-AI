from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mad_ai.ingest import CsvSensorIngestor
from mad_ai.utils import ArtifactStore, build_processed_artifacts
from mad_ai.wmm import WMMMagneticModel


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts\\build_processed_datasets_from_csv.py <input_csv> [output_dir]")

    input_csv = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/processed")

    raw = CsvSensorIngestor().load(input_csv)
    store = ArtifactStore(output_dir)
    wmm_model = WMMMagneticModel(cache_path="data/cache/wmm_cache.json")
    paths = build_processed_artifacts(raw=raw, store=store, wmm_model=wmm_model)

    for path in paths.values():
        print(f"Saved artifact to {path}")


if __name__ == "__main__":
    main()
