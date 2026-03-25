from __future__ import annotations

from dataclasses import asdict
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class NOAAWMMMetadata:
    model_name: str
    epoch: float
    source_zip: str
    coefficient_file: str
    line_count: int


def extract_noaa_wmm_zip(zip_path: str | Path, output_dir: str | Path) -> list[Path]:
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(output_dir)
        return [output_dir / name for name in archive.namelist()]


def find_coefficient_file(root: str | Path) -> Path:
    root = Path(root)
    matches = sorted(root.rglob("*.COF"))
    if not matches:
        matches = sorted(root.rglob("*.cof"))
    if not matches:
        raise FileNotFoundError(f"No WMM coefficient file found under {root}")
    return matches[0]


def parse_wmm_cof_metadata(cof_path: str | Path, source_zip: str | Path | None = None) -> NOAAWMMMetadata:
    cof_path = Path(cof_path)
    lines = [line.strip() for line in cof_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Coefficient file is empty: {cof_path}")

    header = lines[0].split()
    epoch = float(header[0])
    model_name = header[1] if len(header) > 1 else cof_path.stem
    return NOAAWMMMetadata(
        model_name=model_name,
        epoch=epoch,
        source_zip=str(source_zip) if source_zip is not None else "",
        coefficient_file=str(cof_path),
        line_count=len(lines),
    )


def save_wmm_metadata(metadata: NOAAWMMMetadata, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")
    return output_path
