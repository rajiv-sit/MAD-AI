from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import pandas as pd


def _load_module():
    script_path = Path("scripts/evaluate_bahamas_tracking.py").resolve()
    spec = importlib.util.spec_from_file_location("evaluate_bahamas_tracking", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load evaluate_bahamas_tracking.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BahamasTrackingEvalScriptTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module()

    def test_confidence_bucket_labels(self) -> None:
        self.assertEqual(self.module._confidence_bucket(0.8), "strong")
        self.assertEqual(self.module._confidence_bucket(0.5), "moderate")
        self.assertEqual(self.module._confidence_bucket(0.2), "weak")

    def test_segment_metrics_and_confidence_summary_are_built(self) -> None:
        frame = pd.DataFrame(
            {
                "timestamp": [f"2008-02-19T15:32:{value:02d}" for value in range(10)],
                "tracking_error_m": [1000.0, 1200.0, 900.0, 1100.0, 1300.0, 4000.0, 3800.0, 4200.0, 3900.0, 4100.0],
                "innovation_nt": [0.1, 0.2, 0.1, 0.3, 0.2, 1.2, 1.0, 1.1, 1.3, 1.2],
                "confidence": [0.9, 0.8, 0.85, 0.75, 0.78, 0.2, 0.25, 0.3, 0.22, 0.18],
            }
        )

        confidence_summary = self.module._build_confidence_summary(frame)
        segment_metrics = self.module._build_segment_metrics(frame, segment_count=2)

        self.assertIn("buckets", confidence_summary)
        self.assertGreaterEqual(len(confidence_summary["buckets"]), 2)
        self.assertEqual(len(segment_metrics), 2)
        self.assertIn("confidence_bucket", segment_metrics[0])
        self.assertIn(segment_metrics[0]["confidence_bucket"], {"strong", "moderate", "weak"})


if __name__ == "__main__":
    unittest.main()
