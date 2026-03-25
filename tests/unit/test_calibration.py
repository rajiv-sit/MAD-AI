from __future__ import annotations

import json
import unittest

from mad_ai.inference import AnomalyFusionEngine, ThresholdCalibrator, summarize_scores
from tests.unit.helpers import workspace_temp_dir


class CalibrationTestCase(unittest.TestCase):
    def test_threshold_calibrator_returns_expected_summary(self) -> None:
        summary = ThresholdCalibrator(percentile=80.0).calibrate([0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertEqual(summary.percentile, 80.0)
        self.assertEqual(summary.sample_count, 5)
        self.assertGreaterEqual(summary.threshold, 0.0)

    def test_summarize_scores_returns_basic_statistics(self) -> None:
        summary = summarize_scores([1.0, 2.0, 3.0])
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["min"], 1.0)
        self.assertEqual(summary["max"], 3.0)

    def test_anomaly_fusion_engine_can_load_threshold_artifact(self) -> None:
        with workspace_temp_dir() as tmp_dir:
            path = tmp_dir / "thresholds.json"
            path.write_text(
                json.dumps(
                    {
                        "threshold": 12.5,
                        "spatial_weight": 0.25,
                        "temporal_weight": 0.75,
                    }
                ),
                encoding="utf-8",
            )
            engine = AnomalyFusionEngine.from_json(path)
            result = engine.fuse(10.0, 20.0)
            self.assertEqual(engine.threshold, 12.5)
            self.assertTrue(result.is_anomaly)


if __name__ == "__main__":
    unittest.main()
