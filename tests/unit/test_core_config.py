from __future__ import annotations

import unittest

from mad_ai.core.config import load_config
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


if __name__ == "__main__":
    unittest.main()
