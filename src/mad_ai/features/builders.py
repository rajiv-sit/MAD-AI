from __future__ import annotations

import pandas as pd

from mad_ai.core.base import BaseFeatureBuilder, BaseMagneticModel


class ResidualFeatureBuilder(BaseFeatureBuilder):
    def __init__(self, magnetic_model: BaseMagneticModel) -> None:
        self.magnetic_model = magnetic_model

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for row in data.itertuples(index=False):
            baseline = self.magnetic_model.get_field(
                lat=float(row.latitude_deg),
                lon=float(row.longitude_deg),
                alt=float(row.altitude_m),
                timestamp=getattr(row, "timestamp", None),
            )
            result = dict(row._asdict())
            result["baseline_total_nt"] = baseline["total_intensity_nt"]
            result["baseline_declination_deg"] = baseline["declination_deg"]
            result["baseline_inclination_deg"] = baseline["inclination_deg"]

            observed_total = result.get("observed_total_nt", baseline["total_intensity_nt"])
            observed_declination = result.get("observed_declination_deg", baseline["declination_deg"])
            observed_inclination = result.get("observed_inclination_deg", baseline["inclination_deg"])

            result["residual_total_nt"] = float(observed_total) - float(baseline["total_intensity_nt"])
            result["residual_declination_deg"] = float(observed_declination) - float(baseline["declination_deg"])
            result["residual_inclination_deg"] = float(observed_inclination) - float(baseline["inclination_deg"])
            rows.append(result)
        return pd.DataFrame(rows)


class TemporalFeatureBuilder(BaseFeatureBuilder):
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        ordered = data.sort_values("timestamp").copy() if "timestamp" in data.columns else data.copy()
        if "residual_total_nt" in ordered.columns:
            ordered["delta_residual_total_nt"] = ordered["residual_total_nt"].diff().fillna(0.0)
            ordered["rolling_residual_total_nt"] = ordered["residual_total_nt"].rolling(window=3, min_periods=1).mean()
        return ordered
