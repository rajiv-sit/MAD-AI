from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mad_ai.datasets import SpatialGridBuilder, TemporalSequenceBuilder
from mad_ai.features import ResidualFeatureBuilder, TemporalFeatureBuilder
from mad_ai.inference import (
    AnomalyFusionEngine,
    SpatialScorer,
    TemporalScorer,
    combine_weighted_scores,
    evaluate_threshold,
)
from mad_ai.models.spatial import CNNAnomalyModel
from mad_ai.models.temporal import LSTMAnomalyModel
from mad_ai.utils import ArtifactStore
from mad_ai.utils.sample_data import make_sample_sensor_data
from mad_ai.wmm import WMMMagneticModel


def main() -> None:
    calibration_path = Path("outputs/calibration/thresholds.json")
    engine = AnomalyFusionEngine.from_json(calibration_path) if calibration_path.exists() else AnomalyFusionEngine()
    store = ArtifactStore("outputs/evaluation")
    wmm_model = WMMMagneticModel(cache_path="data/cache/wmm_cache.json")

    nominal_raw = make_sample_sensor_data(anomaly_magnitude=0.0)
    anomalous_raw = make_sample_sensor_data(anomaly_magnitude=900.0)

    nominal_temporal = _prepare_temporal_features(nominal_raw, wmm_model)
    anomalous_temporal = _prepare_temporal_features(anomalous_raw, wmm_model)

    nominal_grid = SpatialGridBuilder().build(nominal_temporal)
    anomalous_grid = SpatialGridBuilder().build(anomalous_temporal)
    nominal_sequences = TemporalSequenceBuilder().build(nominal_temporal)
    anomalous_sequences = TemporalSequenceBuilder().build(anomalous_temporal)

    spatial_model = CNNAnomalyModel()
    spatial_model.train(np.stack([nominal_grid, nominal_grid, nominal_grid], axis=0))
    temporal_model = LSTMAnomalyModel()
    temporal_model.train(nominal_sequences)

    spatial_nominal_scores = np.asarray(spatial_model.score(np.stack([nominal_grid, nominal_grid], axis=0)), dtype=float)
    spatial_anomalous_scores = np.asarray(spatial_model.score(np.stack([anomalous_grid, anomalous_grid], axis=0)), dtype=float)
    temporal_nominal_scores = np.asarray(temporal_model.score(nominal_sequences), dtype=float)
    temporal_anomalous_scores = np.asarray(temporal_model.score(anomalous_sequences), dtype=float)

    nominal_scores = combine_weighted_scores(
        spatial_nominal_scores.mean(),
        temporal_nominal_scores,
        spatial_weight=engine.spatial_weight,
        temporal_weight=engine.temporal_weight,
    )
    anomalous_scores = combine_weighted_scores(
        spatial_anomalous_scores.mean(),
        temporal_anomalous_scores,
        spatial_weight=engine.spatial_weight,
        temporal_weight=engine.temporal_weight,
    )

    scores = np.concatenate([nominal_scores, anomalous_scores])
    labels = np.concatenate([np.zeros_like(nominal_scores, dtype=int), np.ones_like(anomalous_scores, dtype=int)])
    metrics = evaluate_threshold(scores, labels, engine.threshold)

    rows = []
    for score in nominal_scores:
        rows.append({"label": 0, "score": float(score)})
    for score in anomalous_scores:
        rows.append({"label": 1, "score": float(score)})

    score_path = store.save_dataframe("score_comparison", pd.DataFrame(rows))
    metrics_path = store.save_json(
        "metrics",
        {
            "threshold": metrics.threshold,
            "true_positives": metrics.true_positives,
            "true_negatives": metrics.true_negatives,
            "false_positives": metrics.false_positives,
            "false_negatives": metrics.false_negatives,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "accuracy": metrics.accuracy,
            "f1_score": metrics.f1_score,
        },
    )

    print(f"Saved evaluation scores to {score_path}")
    print(f"Saved evaluation metrics to {metrics_path}")


def _prepare_temporal_features(raw: pd.DataFrame, wmm_model: WMMMagneticModel) -> pd.DataFrame:
    enriched = ResidualFeatureBuilder(wmm_model).transform(raw)
    return TemporalFeatureBuilder().transform(enriched)


if __name__ == "__main__":
    main()
