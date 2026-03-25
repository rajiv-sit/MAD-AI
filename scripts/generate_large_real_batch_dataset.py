from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mad_ai.utils.sample_data import write_large_real_batch_dataset


def main() -> None:
    output_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/real_batch_large")
    outputs = write_large_real_batch_dataset(output_root)
    for split_name, paths in outputs.items():
        print(f"{split_name}: {len(paths)} files -> {output_root / split_name}")


if __name__ == "__main__":
    main()
