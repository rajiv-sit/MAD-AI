from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


BAHAMAS_ASC_COLUMNS = [
    "source_time_seconds",
    "observed_total_nt",
    "latitude_deg",
    "longitude_deg",
    "aircraft_altitude_ft",
    "aircraft_heading_deg",
    "vessel_latitude_deg",
    "vessel_longitude_deg",
    "vessel_altitude_ft",
    "vessel_heading_deg",
]

FEET_TO_METERS = 0.3048


def load_bahamas_mad_ascii(source: str | Path, track_id: str = "bahamas_flight") -> pd.DataFrame:
    path = Path(source)
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=BAHAMAS_ASC_COLUMNS,
        dtype=float,
        engine="python",
    )
    frame["altitude_m"] = frame["aircraft_altitude_ft"] * FEET_TO_METERS
    frame["vessel_altitude_m"] = frame["vessel_altitude_ft"] * FEET_TO_METERS
    frame["timestamp"] = _build_timestamps(frame["source_time_seconds"], path.name)
    frame["track_id"] = track_id
    frame["source_file"] = path.name
    frame["aircraft_speed_mps"] = _compute_track_speed_mps(frame["latitude_deg"], frame["longitude_deg"], frame["timestamp"])
    frame["aircraft_speed_knots"] = frame["aircraft_speed_mps"] * 1.9438444924406
    frame["vessel_speed_mps"] = _compute_track_speed_mps(
        frame["vessel_latitude_deg"],
        frame["vessel_longitude_deg"],
        frame["timestamp"],
    )
    frame["vessel_speed_knots"] = frame["vessel_speed_mps"] * 1.9438444924406
    frame["range_to_vessel_m"] = _compute_pairwise_range_m(
        frame["latitude_deg"],
        frame["longitude_deg"],
        frame["vessel_latitude_deg"],
        frame["vessel_longitude_deg"],
    )
    return frame


def _build_timestamps(source_time_seconds: pd.Series, source_name: str) -> pd.Series:
    base_date = _infer_base_date_from_source_name(source_name)
    if base_date is None:
        return pd.to_datetime(source_time_seconds, unit="s", origin="unix")
    base_timestamp = pd.Timestamp(base_date)
    offsets = pd.to_timedelta(source_time_seconds.astype(float), unit="s")
    return base_timestamp + offsets


def _infer_base_date_from_source_name(source_name: str) -> str | None:
    normalized_name = source_name.lower()
    if normalized_name == "mad_data.asc":
        return "2008-02-19"
    match = re.search(r"([A-Za-z]{3})(\d{1,2})[_-](\d{4})", source_name)
    if not match:
        if "time_tf_acnav_shipnav" in normalized_name:
            return "2008-02-19"
        return None
    month_text, day_text, year_text = match.groups()
    month_number = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }[month_text.lower()]
    return f"{int(year_text):04d}-{month_number:02d}-{int(day_text):02d}"


def _compute_track_speed_mps(latitude_deg: pd.Series, longitude_deg: pd.Series, timestamp: pd.Series) -> pd.Series:
    lat = latitude_deg.astype(float).to_numpy()
    lon = longitude_deg.astype(float).to_numpy()
    time_seconds = pd.to_datetime(timestamp).astype("int64").to_numpy(dtype=np.float64) / 1_000_000_000.0
    speed = np.zeros(len(lat), dtype=np.float64)
    if len(lat) <= 1:
        return pd.Series(speed)

    earth_radius_m = 6_371_000.0
    lat1 = np.radians(lat[:-1])
    lat2 = np.radians(lat[1:])
    dlat = lat2 - lat1
    dlon = np.radians(lon[1:] - lon[:-1])
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1e-12, 1.0 - a)))
    distance_m = earth_radius_m * c
    delta_t = np.diff(time_seconds)
    delta_t = np.where(delta_t <= 0.0, 1.0, delta_t)
    speed[1:] = distance_m / delta_t
    speed[0] = speed[1]
    return pd.Series(speed)


def _compute_pairwise_range_m(
    latitude_a_deg: pd.Series,
    longitude_a_deg: pd.Series,
    latitude_b_deg: pd.Series,
    longitude_b_deg: pd.Series,
) -> pd.Series:
    lat1 = np.radians(latitude_a_deg.astype(float).to_numpy())
    lon1 = np.radians(longitude_a_deg.astype(float).to_numpy())
    lat2 = np.radians(latitude_b_deg.astype(float).to_numpy())
    lon2 = np.radians(longitude_b_deg.astype(float).to_numpy())
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1e-12, 1.0 - a)))
    return pd.Series(6_371_000.0 * c)
