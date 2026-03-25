from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

from mad_ai.core.config import load_config
from mad_ai.ingest import SplitBatchSensorIngestor
from mad_ai.inference import prepare_observed_features
from mad_ai.utils.sample_data import write_large_real_batch_dataset
from mad_ai.wmm import AnalyticMagneticModel
from tests.unit.helpers import workspace_temp_dir


class RealBatchWorkflowTestCase(unittest.TestCase):
    def test_real_batch_config_loads_expected_splits(self) -> None:
        config = load_config("config/real_batch.yaml")
        self.assertIn("split_sources", config)
        self.assertEqual(config["dataset_manifest"], "data/manifests/real_batch.sample.json")
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

    def test_real_batch_metrics_include_dataset_manifest_metadata(self) -> None:
        payload = json.loads(Path("outputs/evaluation/real_batch_metrics.json").read_text(encoding="utf-8"))
        self.assertIn("dataset_manifest", payload)
        self.assertEqual(payload["dataset_manifest"]["dataset_name"], "real_batch_sample")

    def test_large_real_batch_generator_writes_mixed_formats(self) -> None:
        with workspace_temp_dir() as tmp_dir:
            outputs = write_large_real_batch_dataset(
                tmp_dir / "real_batch_large",
                train_runs=4,
                calibration_runs=2,
                nominal_eval_runs=2,
                anomalous_eval_runs=2,
                rows_per_run=24,
            )
            written_suffixes = {path.suffix.lower() for paths in outputs.values() for path in paths}
            self.assertIn(".csv", written_suffixes)
            self.assertIn(".jsonl", written_suffixes)
            self.assertIn(".sqlite", written_suffixes)
            self.assertTrue(".parquet" in written_suffixes or any("_fallback.jsonl" in str(path) for paths in outputs.values() for path in paths))


if __name__ == "__main__":
    unittest.main()
