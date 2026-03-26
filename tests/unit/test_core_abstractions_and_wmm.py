from __future__ import annotations

import unittest
from datetime import date, datetime
from unittest.mock import patch

from mad_ai.core.base import (
    BaseAnomalyModel,
    BaseDataIngestor,
    BaseDatasetBuilder,
    BaseFeatureBuilder,
    BaseMagneticForwardModel,
    BaseMagneticModel,
    BaseTracker,
    BaseViewModel,
    BaseVisualizer,
)
from mad_ai.wmm.model import WMMMagneticModel, _coerce_date, _decimal_year, _make_cache_key
from tests.unit.helpers import workspace_temp_dir


class CoreAbstractionsAndWMMTestCase(unittest.TestCase):
    def test_abstract_base_classes_cannot_be_instantiated(self) -> None:
        abstract_types = [
            BaseMagneticModel,
            BaseDataIngestor,
            BaseFeatureBuilder,
            BaseDatasetBuilder,
            BaseAnomalyModel,
            BaseVisualizer,
            BaseViewModel,
            BaseMagneticForwardModel,
            BaseTracker,
        ]
        for abstract_type in abstract_types:
            with self.assertRaises(TypeError):
                abstract_type()

    def test_wmm_model_uses_fallback_when_backend_query_returns_none(self) -> None:
        with workspace_temp_dir() as tmp_dir:
            cache_path = tmp_dir / "wmm_cache.json"
            with patch.object(WMMMagneticModel, "_cached_backend_query", return_value=None):
                model = WMMMagneticModel(cache_path=cache_path)
                field = model.get_field(43.7, -79.4, 150.0, datetime(2026, 3, 24, 12, 0, 0))

            self.assertEqual(field["source"], "analytic-fallback")
            self.assertTrue(cache_path.exists())

    def test_wmm_model_prefers_backend_result_when_available(self) -> None:
        backend_result = {
            "declination_deg": 1.0,
            "inclination_deg": 2.0,
            "total_intensity_nt": 3.0,
            "horizontal_intensity_nt": 4.0,
            "north_nt": 5.0,
            "east_nt": 6.0,
            "down_nt": 7.0,
            "source": "geomag",
        }
        with workspace_temp_dir() as tmp_dir:
            with patch.object(WMMMagneticModel, "_cached_backend_query", return_value=backend_result):
                model = WMMMagneticModel(cache_path=tmp_dir / "wmm_cache.json")
                field = model.get_field(10.0, 20.0, 30.0, datetime(2026, 3, 24))
        self.assertEqual(field, backend_result)

    def test_wmm_model_batch_query_uses_datetime_fallback_for_datetimes_and_date_for_dates(self) -> None:
        with workspace_temp_dir() as tmp_dir:
            with patch.object(WMMMagneticModel, "_cached_backend_query", return_value=None):
                model = WMMMagneticModel(cache_path=tmp_dir / "wmm_cache.json")
                fields = model.get_fields(
                    [
                        (43.7, -79.4, 0.0, datetime(2026, 3, 24, 12, 0, 0)),
                        (43.8, -79.5, 100.0, date(2026, 3, 24)),
                    ]
                )

        self.assertEqual(len(fields), 2)
        self.assertEqual(fields[0]["source"], "analytic-fallback")
        self.assertEqual(fields[1]["source"], "analytic-fallback")

    def test_wmm_helper_functions_cover_date_and_cache_key_branches(self) -> None:
        moment = datetime(2026, 3, 24, 8, 30, 0)
        self.assertEqual(_coerce_date(moment), date(2026, 3, 24))
        self.assertEqual(_coerce_date(date(2026, 3, 24)), date(2026, 3, 24))
        self.assertIsNone(_coerce_date(None))
        self.assertEqual(_make_cache_key(1.2345678, 2.3456789, 100.1234, date(2026, 3, 24)), "1.234568|2.345679|100.123|2026-03-24")
        self.assertGreater(_decimal_year(date(2024, 7, 2)), 2024.4)


if __name__ == "__main__":
    unittest.main()
