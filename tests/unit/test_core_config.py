from __future__ import annotations

import unittest
from pathlib import Path

from mad_ai.core.config import load_config
from mad_ai.core.manifests import load_dataset_manifest
from tests.unit.helpers import workspace_temp_dir


class ConfigTestCase(unittest.TestCase):
    def test_load_json_config(self) -> None:
        with workspace_temp_dir() as tmp_dir:
            path = tmp_dir / "config.json"
            path.write_text('{"name": "mad-ai", "enabled": true}', encoding="utf-8")
            loaded = load_config(path)
            self.assertEqual(loaded["name"], "mad-ai")
            self.assertTrue(loaded["enabled"])

    def test_load_yaml_config_without_pyyaml(self) -> None:
        with workspace_temp_dir() as tmp_dir:
            path = tmp_dir / "config.yaml"
            path.write_text("root:\n  value: 3\nflag: true\n", encoding="utf-8")
            loaded = load_config(path)
            self.assertEqual(loaded["root"]["value"], 3)
            self.assertTrue(loaded["flag"])

    def test_missing_config_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_config("does-not-exist.yaml")

    def test_load_dataset_manifest_validates_required_keys(self) -> None:
        manifest = load_dataset_manifest("data/manifests/real_batch.sample.json")
        self.assertEqual(manifest["dataset_name"], "real_batch_sample")
        self.assertIn("split_definitions", manifest)
        self.assertIn("schema_mapping", manifest)
        self.assertEqual(
            manifest["split_definitions"]["nominal_eval"]["source_path"],
            "data/raw/real_batch/nominal_eval",
        )

    def test_load_dataset_manifest_rejects_missing_required_keys(self) -> None:
        with workspace_temp_dir() as tmp_dir:
            path = tmp_dir / "manifest.json"
            path.write_text('{"dataset_name":"broken"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_dataset_manifest(path)

    def test_load_dataset_manifest_rejects_missing_split_metadata(self) -> None:
        with workspace_temp_dir() as tmp_dir:
            path = tmp_dir / "manifest.json"
            path.write_text(
                """{
  "dataset_name": "broken",
  "dataset_kind": "sample",
  "provenance": {
    "source_type": "repo",
    "source_location": "data/raw/example",
    "source_formats": ["csv"],
    "time_span": {"start": "2026-03-24T00:00:00", "end": "2026-03-24T01:00:00"},
    "platforms": ["alpha"],
    "altitude_range_m": [0, 100],
    "coordinate_convention": "WGS84",
    "units": {"latitude_deg": "decimal_degrees"},
    "label_quality": "demo",
    "notes": "demo"
  },
  "split_definitions": {
    "train": {"description": "train only"}
  },
  "schema_mapping": {
    "latitude_deg": ["latitude"],
    "longitude_deg": ["longitude"],
    "altitude_m": ["altitude"],
    "timestamp": ["time"],
    "observed_total_nt": ["total_field_nt"],
    "observed_declination_deg": ["declination_deg"],
    "observed_inclination_deg": ["inclination_deg"],
    "track_id": ["platform_id"],
    "is_injected_anomaly": ["label_anomaly"]
  },
  "data_quality_issues": [{"issue": "demo"}]
}""",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_dataset_manifest(path)


if __name__ == "__main__":
    unittest.main()
