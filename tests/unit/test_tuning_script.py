from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


def _load_tuning_module():
    script_path = Path("scripts/tune_real_batch_models.py").resolve()
    spec = importlib.util.spec_from_file_location("tune_real_batch_models", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load tune_real_batch_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TuningScriptTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_tuning_module()

    def test_build_search_space_includes_model_hyperparameters(self) -> None:
        search_space = self.module._build_search_space(
            tuning={
                "spatial_window_sizes": [12],
                "temporal_sequence_lengths": [8],
                "strides": [4],
                "spatial_weights": [0.6],
                "spatial_latent_channels": [24, 32],
                "temporal_hidden_sizes": [48],
                "spatial_dropouts": [0.05],
                "temporal_dropouts": [0.1],
                "spatial_learning_rates": [0.001],
                "temporal_learning_rates": [0.002],
                "spatial_batch_sizes": [16],
                "temporal_batch_sizes": [24],
                "spatial_epochs": [3],
                "temporal_epochs": [4],
            },
            training_config={},
        )

        self.assertEqual(len(search_space), 2)
        first = search_space[0]
        self.assertIn("spatial_latent_channels", first)
        self.assertIn("temporal_hidden_size", first)
        self.assertIn("spatial_dropout", first)
        self.assertIn("temporal_dropout", first)
        self.assertIn("spatial_learning_rate", first)
        self.assertIn("temporal_learning_rate", first)
        self.assertAlmostEqual(first["temporal_weight"], 0.4)

    def test_best_candidate_config_writes_reusable_training_and_inference_settings(self) -> None:
        best_config = self.module._best_candidate_config(
            config={
                "magnetic_backend": "analytic",
                "training": {"spatial_epochs": 1},
                "inference": {"calibration_percentile": 97.5},
            },
            best={
                "spatial_window_size": 18,
                "temporal_sequence_length": 12,
                "stride": 4,
                "spatial_epochs": 3,
                "temporal_epochs": 4,
                "spatial_batch_size": 16,
                "temporal_batch_size": 24,
                "spatial_learning_rate": 0.001,
                "temporal_learning_rate": 0.002,
                "spatial_latent_channels": 32,
                "temporal_hidden_size": 64,
                "spatial_dropout": 0.05,
                "temporal_dropout": 0.1,
                "spatial_weight": 0.45,
                "temporal_weight": 0.55,
                "threshold": 0.2,
                "anomalous_f1": 0.8,
                "anomalous_recall": 0.9,
                "nominal_false_positive_count": 3,
            },
        )

        self.assertEqual(best_config["training"]["spatial_window_size"], 18)
        self.assertEqual(best_config["training"]["temporal_hidden_size"], 64)
        self.assertAlmostEqual(best_config["inference"]["spatial_weight"], 0.45)
        self.assertEqual(best_config["tuning_result"]["nominal_false_positive_count"], 3)


if __name__ == "__main__":
    unittest.main()
