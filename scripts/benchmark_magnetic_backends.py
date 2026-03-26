from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mad_ai.core.config import load_config
from mad_ai.ingest import SplitBatchSensorIngestor
from mad_ai.inference import prepare_observed_features
from mad_ai.utils.sample_data import make_multi_altitude_global_wmm_grid, make_sample_sensor_data
from mad_ai.wmm import AnalyticMagneticModel, WMMMagneticModel


def main() -> None:
    output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/benchmarks/magnetic_backend_benchmark.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = build_benchmark_report()
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved magnetic backend benchmark report to {output_path}")


def build_benchmark_report() -> dict[str, object]:
    observed_raw = make_sample_sensor_data(rows=128, anomaly_span=0)
    real_batch_large_raw = _load_real_batch_nominal_eval("config/real_batch_large.yaml")
    benchmark_timestamp = datetime(2026, 3, 24, 0, 0, 0)

    workflows = [
        {
            "name": "observed_feature_preparation",
            "payload_rows": int(len(observed_raw)),
            "runner": lambda model: prepare_observed_features(observed_raw, model),
        },
        {
            "name": "real_batch_large_nominal_eval_preparation",
            "payload_rows": int(len(real_batch_large_raw)),
            "runner": lambda model: prepare_observed_features(real_batch_large_raw, model),
        },
        {
            "name": "global_grid_generation",
            "payload_rows": 0,
            "runner": lambda model: make_multi_altitude_global_wmm_grid(
                altitudes_m=[0.0, 1000.0],
                timestamp=benchmark_timestamp,
                lat_step_deg=20.0,
                lon_step_deg=20.0,
                magnetic_model=model,
            ),
        },
    ]

    workflow_reports = []
    for workflow in workflows:
        analytic_metrics = _benchmark_backend(
            workflow_name=str(workflow["name"]),
            backend_name="analytic",
            model=AnalyticMagneticModel(),
            runner=workflow["runner"],
        )
        wmm_metrics = _benchmark_backend(
            workflow_name=str(workflow["name"]),
            backend_name="wmm",
            model=WMMMagneticModel(cache_path=f"outputs/benchmarks/{workflow['name']}_wmm_cache.json"),
            runner=workflow["runner"],
        )
        workflow_reports.append(
            {
                "workflow": workflow["name"],
                "payload_rows": workflow["payload_rows"],
                "analytic": analytic_metrics,
                "wmm": wmm_metrics,
                "speed_ratio_wmm_over_analytic": _safe_ratio(
                    float(wmm_metrics["elapsed_seconds"]),
                    float(analytic_metrics["elapsed_seconds"]),
                ),
                "recommended_backend": _recommend_backend(str(workflow["name"])),
            }
        )

    return {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "wmm_reference_guidance": "Use WMM for physically meaningful baseline analysis, reported evidence, and reviewer-facing interpretation.",
            "analytic_guidance": "Use analytic for fast local checks, tuning loops, tests, and scaling experiments where exact magnetic fidelity is not the point.",
        },
        "workflows": workflow_reports,
    }


def _load_real_batch_nominal_eval(config_path: str | Path):
    config = load_config(config_path)
    ingestor = SplitBatchSensorIngestor(
        schema_mapping=config.get("schema_mapping", {}),
        sqlite_table_name=str(config.get("source_options", {}).get("sqlite_table_name", "sensor_readings")),
    )
    splits = ingestor.load_splits(config["split_sources"])
    return splits["nominal_eval"]


def _benchmark_backend(
    workflow_name: str,
    backend_name: str,
    model,
    runner,
) -> dict[str, object]:
    started = time.perf_counter()
    result = runner(model)
    elapsed = time.perf_counter() - started
    source_counts = {}
    if hasattr(result, "columns") and "source" in result.columns:
        for key, value in result["source"].astype(str).value_counts().items():
            source_counts[str(key)] = int(value)
    return {
        "backend": backend_name,
        "elapsed_seconds": elapsed,
        "result_rows": int(len(result)) if hasattr(result, "__len__") else None,
        "source_counts": source_counts,
        "workflow": workflow_name,
    }


def _recommend_backend(workflow_name: str) -> str:
    if workflow_name == "global_grid_generation":
        return "wmm-required"
    if workflow_name == "real_batch_large_nominal_eval_preparation":
        return "wmm-for-evidence analytic-for-fast-iteration"
    return "analytic-acceptable-for-local-checks wmm-required-for-reported-results"


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return float(numerator / denominator)


if __name__ == "__main__":
    main()
