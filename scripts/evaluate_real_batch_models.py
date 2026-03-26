from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mad_ai.core.config import load_config
from mad_ai.core.manifests import load_dataset_manifest
from mad_ai.ingest import SplitBatchSensorIngestor
from mad_ai.inference import ThresholdCalibrator
from mad_ai.inference import prepare_observed_features, score_observed_features, train_observed_models
from mad_ai.inference import evaluate_threshold
from mad_ai.models.spatial import CNNAnomalyModel
from mad_ai.models.temporal import LSTMAnomalyModel
from mad_ai.wmm import AnalyticMagneticModel, WMMMagneticModel


def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config/real_batch.yaml")
    config = load_config(config_path)
    dataset_manifest_path = config.get("dataset_manifest")
    schema_mapping = config.get("schema_mapping", {})
    split_sources = config.get("split_sources", {})
    training_config = config.get("training", {})
    inference_config = config.get("inference", {})
    source_options = config.get("source_options", {})
    if not split_sources:
        raise ValueError("real batch config must define split_sources.")
    spatial_window_size = int(training_config.get("spatial_window_size", 12))
    temporal_sequence_length = int(training_config.get("temporal_sequence_length", 6))
    stride = int(training_config.get("stride", 3))
    spatial_epochs = int(training_config.get("spatial_epochs", 8))
    temporal_epochs = int(training_config.get("temporal_epochs", 10))
    spatial_batch_size = int(training_config.get("spatial_batch_size", 16))
    temporal_batch_size = int(training_config.get("temporal_batch_size", 32))
    spatial_weight = float(inference_config.get("spatial_weight", 0.5))
    temporal_weight = float(inference_config.get("temporal_weight", 0.5))
    calibration_percentile = float(inference_config.get("calibration_percentile", 97.5))
    robustness_percentiles = [float(value) for value in inference_config.get("robustness_percentiles", [90.0, 95.0, 97.5, 99.0])]
    dataset_manifest = load_dataset_manifest(dataset_manifest_path) if dataset_manifest_path else None

    ingestor = SplitBatchSensorIngestor(
        schema_mapping=schema_mapping,
        sqlite_table_name=str(source_options.get("sqlite_table_name", "sensor_readings")),
    )
    raw_splits = ingestor.load_splits(split_sources)
    magnetic_model = _make_magnetic_model(config)
    prepared = {name: prepare_observed_features(frame, magnetic_model) for name, frame in raw_splits.items()}

    spatial_model, temporal_model, training_summary = train_observed_models(
        prepared["train"],
        spatial_model=CNNAnomalyModel(epochs=spatial_epochs, batch_size=spatial_batch_size, latent_channels=24),
        temporal_model=LSTMAnomalyModel(epochs=temporal_epochs, batch_size=temporal_batch_size, hidden_size=48),
        spatial_window_size=spatial_window_size,
        temporal_sequence_length=temporal_sequence_length,
        stride=stride,
    )

    models_dir = Path("outputs/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    spatial_model_path = models_dir / "real_batch_spatial.pt"
    temporal_model_path = models_dir / "real_batch_temporal.pt"
    spatial_model.save(spatial_model_path)
    temporal_model.save(temporal_model_path)

    nominal_scored, nominal_summary = score_observed_features(
        prepared["nominal_eval"],
        spatial_model=spatial_model,
        temporal_model=temporal_model,
        calibration_features=prepared["calibration"],
        calibration_percentile=calibration_percentile,
        spatial_window_size=spatial_window_size,
        temporal_sequence_length=temporal_sequence_length,
        stride=stride,
        spatial_weight=spatial_weight,
        temporal_weight=temporal_weight,
    )
    threshold = float(nominal_summary["threshold"])
    anomalous_scored, anomalous_summary = score_observed_features(
        prepared["anomalous_eval"],
        spatial_model=spatial_model,
        temporal_model=temporal_model,
        threshold=threshold,
        spatial_window_size=spatial_window_size,
        temporal_sequence_length=temporal_sequence_length,
        stride=stride,
        spatial_weight=spatial_weight,
        temporal_weight=temporal_weight,
    )

    baseline_threshold = ThresholdCalibrator(percentile=calibration_percentile).calibrate(
        prepared["calibration"]["residual_total_nt"].abs().to_numpy(dtype=float)
    ).threshold
    _add_baseline_residual_scoring(nominal_scored, baseline_threshold)
    _add_baseline_residual_scoring(anomalous_scored, baseline_threshold)
    baseline_robustness = {
        f"{percentile:.1f}": ThresholdCalibrator(percentile=percentile).calibrate(
            prepared["calibration"]["residual_total_nt"].abs().to_numpy(dtype=float)
        ).threshold
        for percentile in robustness_percentiles
    }
    fusion_robustness = {
        f"{percentile:.1f}": ThresholdCalibrator(percentile=percentile).calibrate(
            nominal_scored["final_anomaly_score"].to_numpy(dtype=float)
        ).threshold
        for percentile in robustness_percentiles
    }
    false_positive_analysis = _build_false_positive_analysis(
        nominal_scored=nominal_scored,
        baseline_thresholds=baseline_robustness,
        fusion_thresholds=fusion_robustness,
    )
    baseline_fused_comparison = _build_baseline_fused_comparison(
        nominal_scored=nominal_scored,
        anomalous_scored=anomalous_scored,
        baseline_threshold=baseline_threshold,
        fused_threshold=threshold,
    )

    artifact_stem = _artifact_stem(dataset_manifest)
    calibration_path = Path("outputs/calibration") / f"{artifact_stem}_thresholds.json"
    calibration_payload = {
        "threshold": threshold,
        "spatial_weight": spatial_weight,
        "temporal_weight": temporal_weight,
        "baseline_threshold": baseline_threshold,
        "training_summary": training_summary,
        "calibration": nominal_summary.get("calibration", {}),
        "robustness": {
            "fusion_thresholds": fusion_robustness,
            "baseline_thresholds": baseline_robustness,
        },
        "false_positive_analysis": false_positive_analysis,
        "baseline_fused_comparison": baseline_fused_comparison,
        "training": {
            "spatial_window_size": spatial_window_size,
            "temporal_sequence_length": temporal_sequence_length,
            "stride": stride,
            "spatial_epochs": spatial_epochs,
            "temporal_epochs": temporal_epochs,
            "spatial_batch_size": spatial_batch_size,
            "temporal_batch_size": temporal_batch_size,
        },
        "magnetic_backend": str(config.get("magnetic_backend", "wmm")).lower(),
        "config_path": str(config_path),
        "dataset_manifest_path": str(dataset_manifest_path) if dataset_manifest_path else None,
        "dataset_manifest": dataset_manifest,
    }
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path.write_text(json.dumps(calibration_payload, indent=2), encoding="utf-8")

    output_dir = Path("outputs/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / f"{artifact_stem}_score_comparison.csv"
    metrics_path = output_dir / f"{artifact_stem}_metrics.json"
    histogram_path = output_dir / f"{artifact_stem}_score_histogram.png"
    robustness_path = output_dir / f"{artifact_stem}_calibration_robustness.json"
    false_positive_analysis_path = output_dir / f"{artifact_stem}_false_positive_analysis.json"
    false_positive_rows_path = output_dir / f"{artifact_stem}_false_positive_rows.csv"
    comparison_summary_path = output_dir / f"{artifact_stem}_baseline_fused_comparison.json"

    comparison = pd.concat(
        [
            _tag_frame(nominal_scored, "nominal_eval"),
            _tag_frame(anomalous_scored, "anomalous_eval"),
        ],
        ignore_index=True,
    )
    comparison.to_csv(comparison_path, index=False)
    metrics_path.write_text(
        json.dumps(
            {
                "threshold": threshold,
                "baseline_threshold": baseline_threshold,
                "training_summary": training_summary,
                "nominal_eval": nominal_summary,
                "anomalous_eval": anomalous_summary,
                "false_positive_analysis": false_positive_analysis,
                "baseline_fused_comparison": baseline_fused_comparison,
                "dataset_manifest_path": str(dataset_manifest_path) if dataset_manifest_path else None,
                "dataset_manifest": dataset_manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    robustness_path.write_text(
        json.dumps(
            {
                "calibration_percentile": calibration_percentile,
                "fusion_thresholds": fusion_robustness,
                "baseline_thresholds": baseline_robustness,
                "false_positive_analysis": false_positive_analysis,
                "baseline_fused_comparison": baseline_fused_comparison,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    false_positive_analysis_path.write_text(json.dumps(false_positive_analysis, indent=2), encoding="utf-8")
    _build_false_positive_rows(nominal_scored, threshold, baseline_threshold).to_csv(false_positive_rows_path, index=False)
    comparison_summary_path.write_text(json.dumps(baseline_fused_comparison, indent=2), encoding="utf-8")
    _save_histogram(nominal_scored["final_anomaly_score"], anomalous_scored["final_anomaly_score"], threshold, histogram_path)

    print(f"Saved real batch spatial model to {spatial_model_path}")
    print(f"Saved real batch temporal model to {temporal_model_path}")
    print(f"Saved real batch calibration to {calibration_path}")
    print(f"Saved real batch comparison to {comparison_path}")
    print(f"Saved real batch metrics to {metrics_path}")
    print(f"Saved real batch robustness report to {robustness_path}")
    print(f"Saved real batch false-positive analysis to {false_positive_analysis_path}")
    print(f"Saved real batch false-positive rows to {false_positive_rows_path}")
    print(f"Saved real batch baseline-vs-fused comparison to {comparison_summary_path}")
    print(f"Saved real batch histogram to {histogram_path}")


def _make_magnetic_model(config: dict) -> AnalyticMagneticModel | WMMMagneticModel:
    backend = str(config.get("magnetic_backend", "wmm")).strip().lower()
    if backend == "analytic":
        return AnalyticMagneticModel()
    return WMMMagneticModel(cache_path="data/cache/wmm_cache.json")


def _tag_frame(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    tagged = frame.copy()
    tagged["split"] = split_name
    return tagged


def _artifact_stem(dataset_manifest: dict | None) -> str:
    dataset_name = str((dataset_manifest or {}).get("dataset_name", "real_batch")).strip().lower()
    if dataset_name == "real_batch_sample":
        return "real_batch"
    sanitized = re.sub(r"[^a-z0-9]+", "_", dataset_name).strip("_")
    return sanitized or "real_batch"


def _build_false_positive_analysis(
    nominal_scored: pd.DataFrame,
    baseline_thresholds: dict[str, float],
    fusion_thresholds: dict[str, float],
) -> dict[str, object]:
    total_rows = int(len(nominal_scored))
    baseline_scores = nominal_scored["residual_total_nt"].abs().to_numpy(dtype=float)
    fusion_scores = nominal_scored["final_anomaly_score"].to_numpy(dtype=float)
    sweeps = []
    for percentile in sorted(fusion_thresholds.keys(), key=float):
        fusion_threshold = float(fusion_thresholds[percentile])
        baseline_threshold = float(baseline_thresholds[percentile])
        fusion_positive_mask = fusion_scores >= fusion_threshold
        baseline_positive_mask = baseline_scores >= baseline_threshold
        sweeps.append(
            {
                "percentile": float(percentile),
                "fusion_threshold": fusion_threshold,
                "fusion_false_positive_count": int(fusion_positive_mask.sum()),
                "fusion_false_positive_rate": _safe_rate(int(fusion_positive_mask.sum()), total_rows),
                "baseline_threshold": baseline_threshold,
                "baseline_false_positive_count": int(baseline_positive_mask.sum()),
                "baseline_false_positive_rate": _safe_rate(int(baseline_positive_mask.sum()), total_rows),
            }
        )

    worst_tracks = _group_false_positive_breakdown(
        nominal_scored,
        threshold=float(fusion_thresholds[max(fusion_thresholds.keys(), key=float)]),
        baseline_threshold=float(baseline_thresholds[max(baseline_thresholds.keys(), key=float)]),
        group_column="track_id",
    )
    worst_sources = _group_false_positive_breakdown(
        nominal_scored,
        threshold=float(fusion_thresholds[max(fusion_thresholds.keys(), key=float)]),
        baseline_threshold=float(baseline_thresholds[max(baseline_thresholds.keys(), key=float)]),
        group_column="source_file",
    )
    return {
        "nominal_row_count": total_rows,
        "sweeps": sweeps,
        "review_threshold_percentile": max((float(key) for key in fusion_thresholds.keys()), default=0.0),
        "track_breakdown": worst_tracks,
        "source_file_breakdown": worst_sources,
    }


def _build_baseline_fused_comparison(
    nominal_scored: pd.DataFrame,
    anomalous_scored: pd.DataFrame,
    baseline_threshold: float,
    fused_threshold: float,
) -> dict[str, object]:
    nominal_baseline = _summarize_scored_split(
        nominal_scored,
        score_column="residual_total_nt",
        threshold=float(baseline_threshold),
        use_absolute_score=True,
    )
    nominal_fused = _summarize_scored_split(
        nominal_scored,
        score_column="final_anomaly_score",
        threshold=float(fused_threshold),
        use_absolute_score=False,
    )
    anomalous_baseline = _summarize_scored_split(
        anomalous_scored,
        score_column="residual_total_nt",
        threshold=float(baseline_threshold),
        use_absolute_score=True,
    )
    anomalous_fused = _summarize_scored_split(
        anomalous_scored,
        score_column="final_anomaly_score",
        threshold=float(fused_threshold),
        use_absolute_score=False,
    )
    return {
        "baseline_threshold": float(baseline_threshold),
        "fused_threshold": float(fused_threshold),
        "nominal_eval": {
            "baseline": nominal_baseline,
            "fused": nominal_fused,
            "delta_anomaly_count": int(nominal_fused["anomaly_count"] - nominal_baseline["anomaly_count"]),
            "delta_anomaly_rate": float(nominal_fused["anomaly_rate"] - nominal_baseline["anomaly_rate"]),
        },
        "anomalous_eval": {
            "baseline": anomalous_baseline,
            "fused": anomalous_fused,
            "delta_anomaly_count": int(anomalous_fused["anomaly_count"] - anomalous_baseline["anomaly_count"]),
            "delta_anomaly_rate": float(anomalous_fused["anomaly_rate"] - anomalous_baseline["anomaly_rate"]),
        },
    }


def _summarize_scored_split(
    frame: pd.DataFrame,
    score_column: str,
    threshold: float,
    use_absolute_score: bool,
) -> dict[str, object]:
    raw_scores = frame[score_column].to_numpy(dtype=float)
    scores = abs(raw_scores) if use_absolute_score else raw_scores
    anomaly_count = int((scores >= threshold).sum())
    row_count = int(len(frame))
    summary: dict[str, object] = {
        "rows": row_count,
        "score_column": score_column,
        "threshold": float(threshold),
        "anomaly_count": anomaly_count,
        "anomaly_rate": _safe_rate(anomaly_count, row_count),
        "mean_score": float(scores.mean()) if row_count else 0.0,
        "max_score": float(scores.max()) if row_count else 0.0,
    }
    if "is_injected_anomaly" in frame.columns:
        metrics = evaluate_threshold(
            scores=scores,
            labels=frame["is_injected_anomaly"].astype(int).to_numpy(dtype=int),
            threshold=float(threshold),
        )
        summary["metrics"] = {
            "threshold": metrics.threshold,
            "true_positives": metrics.true_positives,
            "true_negatives": metrics.true_negatives,
            "false_positives": metrics.false_positives,
            "false_negatives": metrics.false_negatives,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "accuracy": metrics.accuracy,
            "f1_score": metrics.f1_score,
        }
    return summary


def _group_false_positive_breakdown(
    nominal_scored: pd.DataFrame,
    threshold: float,
    baseline_threshold: float,
    group_column: str,
) -> list[dict[str, object]]:
    if group_column not in nominal_scored.columns:
        return []
    rows = []
    for group_value, group in nominal_scored.groupby(group_column, dropna=False, sort=True):
        fusion_count = int((group["final_anomaly_score"].to_numpy(dtype=float) >= threshold).sum())
        baseline_count = int((group["residual_total_nt"].abs().to_numpy(dtype=float) >= baseline_threshold).sum())
        row_count = int(len(group))
        rows.append(
            {
                group_column: "" if pd.isna(group_value) else str(group_value),
                "row_count": row_count,
                "fusion_false_positive_count": fusion_count,
                "fusion_false_positive_rate": _safe_rate(fusion_count, row_count),
                "baseline_false_positive_count": baseline_count,
                "baseline_false_positive_rate": _safe_rate(baseline_count, row_count),
            }
        )
    rows.sort(key=lambda item: (item["fusion_false_positive_count"], item["baseline_false_positive_count"], item["row_count"]), reverse=True)
    return rows


def _build_false_positive_rows(
    nominal_scored: pd.DataFrame,
    threshold: float,
    baseline_threshold: float,
) -> pd.DataFrame:
    rows = nominal_scored.copy()
    rows["fusion_is_false_positive"] = rows["final_anomaly_score"].to_numpy(dtype=float) >= float(threshold)
    rows["baseline_is_false_positive"] = rows["residual_total_nt"].abs().to_numpy(dtype=float) >= float(baseline_threshold)
    flagged = rows[rows["fusion_is_false_positive"] | rows["baseline_is_false_positive"]].copy()
    preferred_columns = [
        "track_id",
        "timestamp",
        "latitude_deg",
        "longitude_deg",
        "altitude_m",
        "source_file",
        "baseline_residual_score",
        "baseline_is_false_positive",
        "final_anomaly_score",
        "fusion_is_false_positive",
        "is_injected_anomaly",
    ]
    ordered = [column for column in preferred_columns if column in flagged.columns]
    remainder = [column for column in flagged.columns if column not in ordered]
    return flagged[ordered + remainder]


def _safe_rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(count / total)


def _save_histogram(nominal_scores, anomalous_scores, threshold: float, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(nominal_scores, bins=16, alpha=0.65, label="Nominal", color="#336699")
    ax.hist(anomalous_scores, bins=16, alpha=0.65, label="Anomalous", color="#cc5533")
    ax.axvline(threshold, color="#111111", linestyle="--", linewidth=1.5, label="Threshold")
    ax.set_title("Real Batch Final Anomaly Score Distribution")
    ax.set_xlabel("Final anomaly score")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _add_baseline_residual_scoring(scored: pd.DataFrame, baseline_threshold: float) -> None:
    scored["baseline_residual_score"] = scored["residual_total_nt"].abs() / max(float(baseline_threshold), 1e-6)
    scored["baseline_is_anomaly"] = scored["baseline_residual_score"] >= 1.0


if __name__ == "__main__":
    main()
