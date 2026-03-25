from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

from mad_ai.core.config import load_config
from mad_ai.ingest import SplitBatchSensorIngestor
from mad_ai.inference import prepare_observed_features
from mad_ai.wmm import AnalyticMagneticModel


class RealBatchWorkflowTestCase(unittest.TestCase):
    def test_real_batch_config_loads_expected_splits(self) -> None:
        config = load_config("config/real_batch.yaml")
        self.assertIn("split_sources", config)
        self.assertEqual(sorted(config["split_sources"].keys()), ["anomalous_eval", "calibration", "nominal_eval", "train"])

    def test_schema_mapped_real_batch_input_can_be_prepared(self) -> None:
        config = load_config("config/real_batch.yaml")
        raw_splits = SplitBatchSensorIngestor(schema_mapping=config["schema_mapping"]).load_splits(config["split_sources"])
        prepared = prepare_observed_features(raw_splits["anomalous_eval"], AnalyticMagneticModel())
        self.assertIn("residual_total_nt", prepared.columns)
        self.assertIn("source_file", prepared.columns)

    def test_real_batch_scored_csv_contains_baseline_and_fused_columns(self) -> None:
        frame = pd.read_csv("data/processed/real_batch_scored/real_batch_scored.csv")
        self.assertIn("baseline_residual_score", frame.columns)
        self.assertIn("baseline_is_anomaly", frame.columns)
        self.assertIn("final_anomaly_score", frame.columns)

    def test_real_batch_calibration_contains_robustness_block(self) -> None:
        payload = json.loads(Path("outputs/calibration/real_batch_thresholds.json").read_text(encoding="utf-8"))
        self.assertIn("robustness", payload)
        self.assertIn("fusion_thresholds", payload["robustness"])
        self.assertIn("baseline_thresholds", payload["robustness"])


if __name__ == "__main__":
    unittest.main()
