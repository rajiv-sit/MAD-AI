from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import pandas as pd

from mad_ai.wmm import AnalyticMagneticModel
from tests.unit.helpers import workspace_temp_dir


def _load_script_module(script_name: str):
    script_path = Path("scripts") / script_name
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path.resolve())
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScriptRegressionTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.score_real_batch_module = _load_script_module("score_real_batch_folder.py")
        cls.build_real_batch_viewer_module = _load_script_module("build_real_batch_cesium_viewer.py")

    def test_score_real_batch_folder_script_writes_scored_outputs(self) -> None:
        with workspace_temp_dir() as tmp_dir:
            output_csv = tmp_dir / "scored.csv"
            summary_json = tmp_dir / "summary.json"
            argv = [
                "score_real_batch_folder.py",
                "data/raw/real_batch/anomalous_eval",
                "config/real_batch.yaml",
                str(output_csv),
                str(summary_json),
            ]
            with patch.object(sys, "argv", argv):
                with patch.object(
                    self.score_real_batch_module,
                    "_make_magnetic_model",
                    return_value=AnalyticMagneticModel(),
                ):
                    self.score_real_batch_module.main()

            self.assertTrue(output_csv.exists())
            self.assertTrue(summary_json.exists())

            frame = pd.read_csv(output_csv)
            summary = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertIn("final_anomaly_score", frame.columns)
            self.assertIn("baseline_residual_score", frame.columns)
            self.assertIn("baseline_is_anomaly", frame.columns)
            self.assertEqual(summary["output_csv"], str(output_csv))
            self.assertIn("baseline_anomaly_count", summary)

    def test_score_real_batch_folder_script_requires_input_dir_argument(self) -> None:
        with patch.object(sys, "argv", ["score_real_batch_folder.py"]):
            with self.assertRaises(SystemExit):
                self.score_real_batch_module.main()

    def test_build_real_batch_cesium_viewer_script_writes_html(self) -> None:
        with workspace_temp_dir() as tmp_dir:
            output_html = tmp_dir / "real_batch_viewer.html"
            argv = [
                "build_real_batch_cesium_viewer.py",
                "config/real_batch.yaml",
                str(output_html),
            ]
            with patch.object(sys, "argv", argv):
                with patch.object(self.build_real_batch_viewer_module.subprocess, "run") as mocked_run:
                    self.build_real_batch_viewer_module.main()

            self.assertTrue(output_html.exists())
            text = output_html.read_text(encoding="utf-8")
            self.assertIn("MAD-AI Real Batch Globe", text)
            self.assertIn("Final Anomaly Score", text)
            self.assertIn("Baseline Residual Score", text)
            mocked_run.assert_called()


if __name__ == "__main__":
    unittest.main()
