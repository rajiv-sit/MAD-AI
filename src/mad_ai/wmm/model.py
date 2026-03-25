from __future__ import annotations

import json
import math
from functools import lru_cache
from datetime import datetime
from datetime import date
from pathlib import Path

from mad_ai.core.base import BaseMagneticModel
from mad_ai.core.types import MagneticField


class AnalyticMagneticModel(BaseMagneticModel):
    """
    Deterministic fallback magnetic model.

    This is not a replacement for NOAA WMM. It provides stable synthetic values so
    the rest of the pipeline can run before a WMM backend is integrated.
    """

    def get_field(self, lat: float, lon: float, alt: float, timestamp: datetime | None = None) -> dict[str, float | str]:
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        seasonal = 0.0 if timestamp is None else 150.0 * math.sin(timestamp.timetuple().tm_yday / 365.25)

        total = 47000.0 + 8000.0 * math.sin(lat_rad) + 1200.0 * math.cos(lon_rad) - 0.002 * alt + seasonal
        declination = 12.0 * math.sin(lon_rad) * math.cos(lat_rad)
        inclination = 65.0 * math.sin(lat_rad)
        horizontal = total * math.cos(math.radians(inclination))
        north = horizontal * math.cos(math.radians(declination))
        east = horizontal * math.sin(math.radians(declination))
        down = total * math.sin(math.radians(inclination))
        field = MagneticField(
            declination_deg=declination,
            inclination_deg=inclination,
            total_intensity_nt=total,
            horizontal_intensity_nt=horizontal,
            north_nt=north,
            east_nt=east,
            down_nt=down,
            source="analytic-fallback",
        )
        return {
            "declination_deg": field.declination_deg,
            "inclination_deg": field.inclination_deg,
            "total_intensity_nt": field.total_intensity_nt,
            "horizontal_intensity_nt": field.horizontal_intensity_nt,
            "north_nt": field.north_nt,
            "east_nt": field.east_nt,
            "down_nt": field.down_nt,
            "source": field.source,
        }


class WMMMagneticModel(BaseMagneticModel):
    """
    WMM wrapper with a deterministic fallback.

    Replace `_try_backend_query` with a concrete NOAA or geomagmodels integration once
    the library choice is finalized in the environment.
    """

    def __init__(self, cache_path: str | Path | None = "data/cache/wmm_cache.json") -> None:
        self._fallback = AnalyticMagneticModel()
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self._disk_cache: dict[str, dict[str, float | str]] | None = None
        self._cache_dirty = False

    def get_field(self, lat: float, lon: float, alt: float, timestamp: datetime | None = None) -> dict[str, float | str]:
        query_date = _coerce_date(timestamp)
        cache_key = _make_cache_key(float(lat), float(lon), float(alt), query_date)

        cached = self._get_disk_cached(cache_key)
        if cached is not None:
            return cached

        data = self._cached_backend_query(float(lat), float(lon), float(alt), query_date.isoformat() if query_date else "")
        if data is not None:
            self._set_disk_cached(cache_key, data)
            return data

        fallback = self._fallback.get_field(lat, lon, alt, timestamp)
        self._set_disk_cached(cache_key, fallback)
        return fallback

    def get_fields(
        self,
        queries: list[tuple[float, float, float, datetime | date | None]],
    ) -> list[dict[str, float | str]]:
        results: list[dict[str, float | str]] = []
        for lat, lon, alt, timestamp in queries:
            query_date = _coerce_date(timestamp)
            cache_key = _make_cache_key(float(lat), float(lon), float(alt), query_date)

            cached = self._get_disk_cached(cache_key)
            if cached is not None:
                results.append(cached)
                continue

            backend = self._cached_backend_query(float(lat), float(lon), float(alt), query_date.isoformat() if query_date else "")
            if backend is not None:
                self._set_disk_cached(cache_key, backend, persist=False)
                results.append(backend)
                continue

            fallback = self._fallback.get_field(lat, lon, alt, timestamp if isinstance(timestamp, datetime) else None)
            self._set_disk_cached(cache_key, fallback, persist=False)
            results.append(fallback)

        self._flush_disk_cache()
        return results

    @staticmethod
    @lru_cache(maxsize=4096)
    def _cached_backend_query(lat: float, lon: float, alt: float, date_key: str) -> dict[str, float | str] | None:
        query_date = date.fromisoformat(date_key) if date_key else None
        return WMMMagneticModel._try_backend_query(lat, lon, alt, query_date)

    def _try_backend_query(
        lat: float,
        lon: float,
        alt: float,
        timestamp: date | None = None,
    ) -> dict[str, float | str] | None:
        geomag_result = _query_geomag(lat, lon, alt, timestamp)
        if geomag_result is not None:
            return geomag_result

        wmm2020_result = _query_wmm2020(lat, lon, alt, timestamp)
        if wmm2020_result is not None:
            return wmm2020_result
        return None

    def _get_disk_cached(self, cache_key: str) -> dict[str, float | str] | None:
        if self.cache_path is None:
            return None
        cache = self._load_disk_cache()
        return cache.get(cache_key)

    def _set_disk_cached(self, cache_key: str, value: dict[str, float | str], persist: bool = True) -> None:
        if self.cache_path is None:
            return
        cache = self._load_disk_cache()
        cache[cache_key] = value
        self._cache_dirty = True
        if persist:
            self._flush_disk_cache()

    def _flush_disk_cache(self) -> None:
        if self.cache_path is None or not self._cache_dirty:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._load_disk_cache(), indent=2), encoding="utf-8")
        self._cache_dirty = False

    def _load_disk_cache(self) -> dict[str, dict[str, float | str]]:
        if self._disk_cache is not None:
            return self._disk_cache
        if self.cache_path is None or not self.cache_path.exists():
            self._disk_cache = {}
            return self._disk_cache
        self._disk_cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        return self._disk_cache


def _query_geomag(lat: float, lon: float, alt: float, timestamp: date | None) -> dict[str, float | str] | None:
    try:
        from geomag import geomag
    except ImportError:
        return None

    model = _get_geomag_model(geomag)
    altitude_ft = alt * 3.280839895
    result = model.GeoMag(lat, lon, h=altitude_ft, time=timestamp or date.today())
    return {
        "declination_deg": float(result.dec),
        "inclination_deg": float(result.dip),
        "total_intensity_nt": float(result.ti),
        "horizontal_intensity_nt": float(result.bh),
        "north_nt": float(result.bx),
        "east_nt": float(result.by),
        "down_nt": float(result.bz),
        "source": "geomag",
    }


def _query_wmm2020(lat: float, lon: float, alt: float, timestamp: date | None) -> dict[str, float | str] | None:
    try:
        import wmm2020
    except Exception:
        return None

    try:
        result = wmm2020.wmm_point(glat=lat, glon=lon, alt_km=alt / 1000.0, yeardec=_decimal_year(timestamp))
    except Exception:
        return None

    keys = {key.lower(): value for key, value in dict(result).items()}
    total = float(keys.get("f", keys.get("total", 0.0)))
    horizontal = float(keys.get("h", keys.get("horizontal", 0.0)))
    north = float(keys.get("x", keys.get("north", 0.0)))
    east = float(keys.get("y", keys.get("east", 0.0)))
    down = float(keys.get("z", keys.get("down", 0.0)))
    declination = float(keys.get("d", keys.get("declination", 0.0)))
    inclination = float(keys.get("i", keys.get("inclination", 0.0)))
    return {
        "declination_deg": declination,
        "inclination_deg": inclination,
        "total_intensity_nt": total,
        "horizontal_intensity_nt": horizontal,
        "north_nt": north,
        "east_nt": east,
        "down_nt": down,
        "source": "wmm2020",
    }


@lru_cache(maxsize=1)
def _get_geomag_model(geomag_module: object):
    return geomag_module.GeoMag()


def _coerce_date(timestamp: datetime | date | None) -> date | None:
    if timestamp is None:
        return None
    if isinstance(timestamp, datetime):
        return timestamp.date()
    return timestamp


def _make_cache_key(lat: float, lon: float, alt: float, timestamp: date | None) -> str:
    date_key = timestamp.isoformat() if timestamp is not None else ""
    return f"{lat:.6f}|{lon:.6f}|{alt:.3f}|{date_key}"


def _decimal_year(timestamp: date | None) -> float:
    if timestamp is None:
        timestamp = date.today()
    start = date(timestamp.year, 1, 1)
    next_start = date(timestamp.year + 1, 1, 1)
    elapsed = (timestamp - start).days
    span = (next_start - start).days
    return timestamp.year + (elapsed / span)
