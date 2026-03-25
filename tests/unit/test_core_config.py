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

    def test_load_dataset_manifest_rejects_missing_required_keys(self) -> None:
        with workspace_temp_dir() as tmp_dir:
            path = tmp_dir / "manifest.json"
            path.write_text('{"dataset_name":"broken"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_dataset_manifest(path)


if __name__ == "__main__":
    unittest.main()
