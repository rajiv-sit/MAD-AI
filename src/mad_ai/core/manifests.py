from __future__ import annotations

from pathlib import Path
from typing import Any

from mad_ai.core.config import load_config


REQUIRED_MANIFEST_KEYS = (
    "dataset_name",
    "dataset_kind",
    "provenance",
    "split_definitions",
    "schema_mapping",
    "data_quality_issues",
)
REQUIRED_PROVENANCE_KEYS = (
    "source_type",
    "source_location",
    "source_formats",
    "time_span",
    "platforms",
    "altitude_range_m",
    "coordinate_convention",
    "units",
    "label_quality",
    "notes",
)
REQUIRED_SPLIT_KEYS = ("train", "calibration", "nominal_eval", "anomalous_eval")
REQUIRED_SCHEMA_MAPPING_KEYS = (
    "latitude_deg",
    "longitude_deg",
    "altitude_m",
    "timestamp",
    "observed_total_nt",
    "observed_declination_deg",
    "observed_inclination_deg",
    "track_id",
    "is_injected_anomaly",
)


def load_dataset_manifest(path: str | Path) -> dict[str, Any]:
    manifest = load_config(path)
    missing = [key for key in REQUIRED_MANIFEST_KEYS if key not in manifest]
    if missing:
        missing_keys = ", ".join(missing)
        raise ValueError(f"Dataset manifest is missing required keys: {missing_keys}")
    _validate_provenance(manifest["provenance"])
    _validate_split_definitions(manifest["split_definitions"])
    _validate_schema_mapping(manifest["schema_mapping"])
    _validate_data_quality_issues(manifest["data_quality_issues"])
    return manifest


def _validate_provenance(provenance: Any) -> None:
    if not isinstance(provenance, dict):
        raise ValueError("Dataset manifest provenance must be a mapping.")
    missing = [key for key in REQUIRED_PROVENANCE_KEYS if key not in provenance]
    if missing:
        missing_keys = ", ".join(missing)
        raise ValueError(f"Dataset manifest provenance is missing required keys: {missing_keys}")


def _validate_split_definitions(split_definitions: Any) -> None:
    if not isinstance(split_definitions, dict):
        raise ValueError("Dataset manifest split_definitions must be a mapping.")
    missing = [key for key in REQUIRED_SPLIT_KEYS if key not in split_definitions]
    if missing:
        missing_keys = ", ".join(missing)
        raise ValueError(f"Dataset manifest split_definitions are missing required splits: {missing_keys}")
    for split_name, definition in split_definitions.items():
        if not isinstance(definition, dict):
            raise ValueError(f"Dataset manifest split definition '{split_name}' must be a mapping.")
        if "description" not in definition or "source_path" not in definition:
            raise ValueError(
                f"Dataset manifest split definition '{split_name}' must define 'description' and 'source_path'."
            )


def _validate_schema_mapping(schema_mapping: Any) -> None:
    if not isinstance(schema_mapping, dict):
        raise ValueError("Dataset manifest schema_mapping must be a mapping.")
    missing = [key for key in REQUIRED_SCHEMA_MAPPING_KEYS if key not in schema_mapping]
    if missing:
        missing_keys = ", ".join(missing)
        raise ValueError(f"Dataset manifest schema_mapping is missing required keys: {missing_keys}")


def _validate_data_quality_issues(data_quality_issues: Any) -> None:
    if not isinstance(data_quality_issues, list) or not data_quality_issues:
        raise ValueError("Dataset manifest data_quality_issues must be a non-empty list.")
