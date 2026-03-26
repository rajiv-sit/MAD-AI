from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


def _load_module():
    script_path = Path("scripts/benchmark_magnetic_backends.py").resolve()
    spec = importlib.util.spec_from_file_location("benchmark_magnetic_backends", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load benchmark_magnetic_backends.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BackendBenchmarkScriptTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module()

    def test_recommend_backend_covers_expected_workflows(self) -> None:
        self.assertEqual(self.module._recommend_backend("global_grid_generation"), "wmm-required")
        self.assertIn("analytic", self.module._recommend_backend("observed_feature_preparation"))

    def test_safe_ratio_handles_zero_denominator(self) -> None:
        self.assertIsNone(self.module._safe_ratio(2.0, 0.0))
        self.assertEqual(self.module._safe_ratio(6.0, 3.0), 2.0)


if __name__ == "__main__":
    unittest.main()
