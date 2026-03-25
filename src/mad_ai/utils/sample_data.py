from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from mad_ai.wmm import WMMMagneticModel


def make_sample_sensor_data(rows: int = 128, anomaly_magnitude: float = 900.0, anomaly_span: int = 6) -> pd.DataFrame:
    latitudes = np.linspace(43.2, 44.8, rows)
    longitudes = np.linspace(-80.8, -78.4, rows)
    timestamps = [datetime(2026, 3, 24) + timedelta(minutes=5 * idx) for idx in range(rows)]
    baseline = 52000 + 500 * np.sin(np.linspace(0, 4 * np.pi, rows))
    anomaly = np.zeros(rows)
    span = max(0, min(anomaly_span, rows))
    anomaly[rows // 2 : rows // 2 + span] = anomaly_magnitude

    return pd.DataFrame(
        {
            "latitude_deg": latitudes,
            "longitude_deg": longitudes,
            "altitude_m": np.zeros(rows),
            "timestamp": timestamps,
            "observed_total_nt": baseline + anomaly,
            "observed_declination_deg": -8.0 + 0.2 * np.cos(np.linspace(0, 2 * np.pi, rows)),
            "observed_inclination_deg": 58.0 + 0.5 * np.sin(np.linspace(0, 3 * np.pi, rows)),
            "is_injected_anomaly": anomaly > 0.0,
        }
    )


def make_observed_residual_datasets(
    train_runs: int = 8,
    calibration_runs: int = 4,
    nominal_eval_runs: int = 4,
    anomalous_eval_runs: int = 4,
    rows_per_run: int = 96,
) -> dict[str, pd.DataFrame]:
    """Build deterministic observed datasets for training, calibration, and evaluation."""

    def _make_run(run_index: int, anomaly_magnitude: float, anomaly_span: int, split_name: str) -> pd.DataFrame:
        frame = make_sample_sensor_data(
            rows=rows_per_run,
            anomaly_magnitude=anomaly_magnitude,
            anomaly_span=anomaly_span,
        ).copy()
        frame["run_id"] = f"{split_name}_run_{run_index + 1}"
        frame["dataset_split"] = split_name
        frame["latitude_deg"] = frame["latitude_deg"] + (0.18 * run_index)
        frame["longitude_deg"] = frame["longitude_deg"] - (0.22 * run_index)
        frame["altitude_m"] = float((run_index % 4) * 250.0)
        frame["observed_total_nt"] = frame["observed_total_nt"] + (35.0 * run_index)
        frame["observed_declination_deg"] = frame["observed_declination_deg"] + (0.05 * run_index)
        frame["observed_inclination_deg"] = frame["observed_inclination_deg"] - (0.04 * run_index)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"]) + pd.to_timedelta(run_index * rows_per_run * 5, unit="m")
        return frame

    train_frames = [_make_run(idx, anomaly_magnitude=0.0, anomaly_span=0, split_name="train") for idx in range(train_runs)]
    calibration_frames = [
        _make_run(idx, anomaly_magnitude=0.0, anomaly_span=0, split_name="calibration") for idx in range(calibration_runs)
    ]
    nominal_eval_frames = [
        _make_run(idx, anomaly_magnitude=0.0, anomaly_span=0, split_name="nominal_eval") for idx in range(nominal_eval_runs)
    ]
    anomalous_eval_frames = [
        _make_run(
            idx,
            anomaly_magnitude=float([350.0, 650.0, 900.0, 1200.0][idx % 4]),
            anomaly_span=int([4, 6, 8, 10][idx % 4]),
            split_name="anomalous_eval",
        )
        for idx in range(anomalous_eval_runs)
    ]

    return {
        "train": pd.concat(train_frames, ignore_index=True),
        "calibration": pd.concat(calibration_frames, ignore_index=True),
        "nominal_eval": pd.concat(nominal_eval_frames, ignore_index=True),
        "anomalous_eval": pd.concat(anomalous_eval_frames, ignore_index=True),
    }


def make_large_real_batch_datasets(
    train_runs: int = 10,
    calibration_runs: int = 4,
    nominal_eval_runs: int = 4,
    anomalous_eval_runs: int = 4,
    rows_per_run: int = 72,
) -> dict[str, pd.DataFrame]:
    datasets = make_observed_residual_datasets(
        train_runs=train_runs,
        calibration_runs=calibration_runs,
        nominal_eval_runs=nominal_eval_runs,
        anomalous_eval_runs=anomalous_eval_runs,
        rows_per_run=rows_per_run,
    )
    for split_name, frame in datasets.items():
        datasets[split_name] = _rename_to_real_sensor_schema(frame)
    return datasets


def write_large_real_batch_dataset(
    output_root: str | Path,
    train_runs: int = 10,
    calibration_runs: int = 4,
    nominal_eval_runs: int = 4,
    anomalous_eval_runs: int = 4,
    rows_per_run: int = 72,
) -> dict[str, list[Path]]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    datasets = make_large_real_batch_datasets(
        train_runs=train_runs,
        calibration_runs=calibration_runs,
        nominal_eval_runs=nominal_eval_runs,
        anomalous_eval_runs=anomalous_eval_runs,
        rows_per_run=rows_per_run,
    )
    split_outputs: dict[str, list[Path]] = {}
    for split_name, frame in datasets.items():
        split_dir = output_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        split_outputs[split_name] = []
        for run_id, run_frame in frame.groupby("platform_id", sort=False):
            run_frame = run_frame.reset_index(drop=True)
            file_index = len(split_outputs[split_name])
            if file_index % 4 == 0:
                path = split_dir / f"{run_id}.csv"
                run_frame.to_csv(path, index=False)
            elif file_index % 4 == 1:
                path = split_dir / f"{run_id}.jsonl"
                run_frame.to_json(path, orient="records", lines=True, date_format="iso")
            elif file_index % 4 == 2:
                path = _write_parquet_or_fallback(run_frame, split_dir, run_id)
            else:
                path = split_dir / f"{run_id}.sqlite"
                _write_sqlite_frame(run_frame, path)
            split_outputs[split_name].append(path)
    return split_outputs


def make_global_wmm_grid(
    lat_step_deg: float = 10.0,
    lon_step_deg: float = 10.0,
    altitude_m: float = 0.0,
    timestamp: datetime | None = None,
    magnetic_model: WMMMagneticModel | None = None,
) -> pd.DataFrame:
    model = magnetic_model or WMMMagneticModel()
    latitudes = np.arange(-90.0, 90.0 + 1e-9, lat_step_deg)
    longitudes = np.arange(-180.0, 180.0 + 1e-9, lon_step_deg)

    coordinate_rows = [(float(lat), float(lon)) for lat in latitudes for lon in longitudes]
    query_rows = [(lat, lon, float(altitude_m), timestamp) for lat, lon in coordinate_rows]
    if hasattr(model, "get_fields"):
        fields = model.get_fields(query_rows)
    else:
        fields = [model.get_field(lat, lon, altitude_m, timestamp) for lat, lon in coordinate_rows]

    rows: list[dict[str, float | datetime | str]] = []
    for (lat, lon), field in zip(coordinate_rows, fields):
        rows.append(
            {
                "latitude_deg": float(lat),
                "longitude_deg": float(lon),
                "altitude_m": float(altitude_m),
                "timestamp": timestamp,
                "baseline_total_nt": float(field["total_intensity_nt"]),
                "baseline_declination_deg": float(field["declination_deg"]),
                "baseline_inclination_deg": float(field["inclination_deg"]),
                "horizontal_intensity_nt": float(field["horizontal_intensity_nt"]),
                "north_nt": float(field["north_nt"]),
                "east_nt": float(field["east_nt"]),
                "down_nt": float(field["down_nt"]),
                "source": str(field["source"]),
                "residual_total_nt": 0.0,
            }
        )

    return pd.DataFrame(rows)


def make_time_series_global_wmm_grid(
    timestamps: list[datetime],
    lat_step_deg: float = 20.0,
    lon_step_deg: float = 20.0,
    altitude_m: float = 0.0,
    magnetic_model: WMMMagneticModel | None = None,
) -> pd.DataFrame:
    frames = [
        make_global_wmm_grid(
            lat_step_deg=lat_step_deg,
            lon_step_deg=lon_step_deg,
            altitude_m=altitude_m,
            timestamp=timestamp,
            magnetic_model=magnetic_model,
        )
        for timestamp in timestamps
    ]
    return pd.concat(frames, ignore_index=True)


def make_multi_altitude_global_wmm_grid(
    altitudes_m: list[float],
    timestamp: datetime | None = None,
    lat_step_deg: float = 20.0,
    lon_step_deg: float = 20.0,
    magnetic_model: WMMMagneticModel | None = None,
) -> pd.DataFrame:
    frames = [
        make_global_wmm_grid(
            lat_step_deg=lat_step_deg,
            lon_step_deg=lon_step_deg,
            altitude_m=altitude,
            timestamp=timestamp,
            magnetic_model=magnetic_model,
        )
        for altitude in altitudes_m
    ]
    return pd.concat(frames, ignore_index=True)


def make_tracked_sensor_data(
    track_count: int = 3,
    samples_per_track: int = 12,
    anomaly_magnitude: float = 750.0,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    start_time = datetime(2026, 3, 24, 0, 0, 0)

    for track_index in range(track_count):
        latitudes = np.linspace(35.0 + 4.0 * track_index, 42.0 + 4.0 * track_index, samples_per_track)
        longitudes = np.linspace(-125.0 + 6.0 * track_index, -95.0 + 5.0 * track_index, samples_per_track)
        altitudes = np.linspace(250.0 * track_index, 250.0 * track_index + 1200.0, samples_per_track)
        timestamps = [start_time + timedelta(minutes=10 * idx) for idx in range(samples_per_track)]
        phase = np.linspace(0.0, 2.0 * np.pi, samples_per_track)
        baseline = 50000.0 + 1100.0 * np.sin(phase + track_index * 0.35)
        anomaly = np.zeros(samples_per_track)
        anomaly[samples_per_track // 2] = anomaly_magnitude

        rows.append(
            pd.DataFrame(
                {
                    "track_id": f"track_{track_index + 1}",
                    "latitude_deg": latitudes,
                    "longitude_deg": longitudes,
                    "altitude_m": altitudes,
                    "timestamp": timestamps,
                    "observed_total_nt": baseline + anomaly,
                    "observed_declination_deg": -12.0 + track_index + 0.35 * np.cos(phase),
                    "observed_inclination_deg": 50.0 + 1.5 * track_index + 0.45 * np.sin(phase),
                }
            )
        )

    return pd.concat(rows, ignore_index=True)


def _rename_to_real_sensor_schema(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.rename(
        columns={
            "latitude_deg": "latitude",
            "longitude_deg": "longitude",
            "altitude_m": "altitude",
            "timestamp": "time",
            "observed_total_nt": "total_field_nt",
            "observed_declination_deg": "declination_deg",
            "observed_inclination_deg": "inclination_deg",
            "track_id": "platform_id",
            "run_id": "platform_id",
            "is_injected_anomaly": "label_anomaly",
        }
    ).copy()
    if "platform_id" not in renamed.columns:
        renamed["platform_id"] = [f"platform_{idx + 1}" for idx in range(len(renamed))]
    return renamed


def _write_sqlite_frame(frame: pd.DataFrame, path: Path) -> None:
    with sqlite3.connect(path) as connection:
        frame.to_sql("sensor_readings", connection, if_exists="replace", index=False)


def _write_parquet_or_fallback(frame: pd.DataFrame, split_dir: Path, run_id: str) -> Path:
    parquet_path = split_dir / f"{run_id}.parquet"
    try:
        frame.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception:
        fallback_path = split_dir / f"{run_id}_fallback.jsonl"
        frame.to_json(fallback_path, orient="records", lines=True, date_format="iso")
        return fallback_path
