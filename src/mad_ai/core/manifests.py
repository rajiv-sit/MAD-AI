from __future__ import annotations

from pathlib import Path
from typing import Any

from mad_ai.core.config import load_config


REQUIRED_MANIFEST_KEYS = (
    "dataset_name",
    "dataset_kind",
    "provenance",
    "split_definitions",
)


def load_dataset_manifest(path: str | Path) -> dict[str, Any]:
    manifest = load_config(path)
    missing = [key for key in REQUIRED_MANIFEST_KEYS if key not in manifest]
    if missing:
        missing_keys = ", ".join(missing)
        raise ValueError(f"Dataset manifest is missing required keys: {missing_keys}")
    return manifest
