from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mad_ai.ingest import load_bahamas_mad_ascii


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python scripts\\import_bahamas_mad_ascii.py <input_asc> [output_csv]"
        )

    input_asc = Path(sys.argv[1])
    output_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/processed/bahamas/bahamas_mad_observed.csv")

    frame = load_bahamas_mad_ascii(input_asc)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    print(f"Saved Bahamas MAD observed CSV to {output_csv}")


if __name__ == "__main__":
    main()
