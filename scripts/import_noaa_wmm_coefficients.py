from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mad_ai.wmm import extract_noaa_wmm_zip, find_coefficient_file, parse_wmm_cof_metadata, save_wmm_metadata


def main() -> None:
    zip_path = Path("data/raw/noaa_wmm2025.zip")
    output_dir = Path("data/raw/noaa_wmm2025")
    extracted = extract_noaa_wmm_zip(zip_path, output_dir)
    coefficient_file = find_coefficient_file(output_dir)
    metadata = parse_wmm_cof_metadata(coefficient_file, source_zip=zip_path)
    metadata_path = save_wmm_metadata(metadata, output_dir / "wmm_metadata.json")

    print(f"Extracted {len(extracted)} NOAA files into {output_dir}")
    print(f"Coefficient file: {coefficient_file}")
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
