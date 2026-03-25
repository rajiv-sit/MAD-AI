from __future__ import annotations

from datetime import datetime, timedelta

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
        }
    )


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

    rows: list[dict[str, float | datetime | str]] = []
    for lat in latitudes:
        for lon in longitudes:
            field = model.get_field(float(lat), float(lon), altitude_m, timestamp)
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
