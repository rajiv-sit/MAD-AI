from __future__ import annotations

import json
import unittest
from datetime import datetime

import numpy as np

from mad_ai.utils import ArtifactStore
from mad_ai.utils.sample_data import make_sample_sensor_data
from mad_ai.wmm import WMMMagneticModel
from tests.unit.helpers import workspace_temp_dir


class PersistenceAndCacheTestCase(unittest.TestCase):
    def test_artifact_store_saves_and_loads_dataframe_array_and_json(self) -> None:
        with workspace_temp_dir() as tmp_dir:
            store = ArtifactStore(tmp_dir)
            frame = make_sample_sensor_data(4)
            array = np.arange(12, dtype=float).reshape(3, 4)

            frame_path = store.save_dataframe("samples", frame)
            array_path = store.save_array("grid", array)
            json_path = store.save_json("meta", {"rows": 4})

            self.assertTrue(frame_path.exists())
            self.assertTrue(array_path.exists())
            self.assertTrue(json_path.exists())

            loaded_frame = store.load_dataframe("samples", parse_dates=["timestamp"])
            loaded_array = store.load_array("grid")
            self.assertEqual(len(loaded_frame), 4)
            self.assertEqual(loaded_array.shape, (3, 4))

    def test_wmm_model_persists_disk_cache(self) -> None:
        with workspace_temp_dir() as tmp_dir:
            cache_path = tmp_dir / "wmm_cache.json"
            model = WMMMagneticModel(cache_path=cache_path)
            field = model.get_field(43.7, -79.4, 0.0, datetime(2026, 3, 24))

            self.assertTrue(cache_path.exists())
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            self.assertEqual(len(cached), 1)
            cached_value = next(iter(cached.values()))
            self.assertEqual(cached_value["source"], field["source"])

            reloaded = WMMMagneticModel(cache_path=cache_path)
            field_again = reloaded.get_field(43.7, -79.4, 0.0, datetime(2026, 3, 24))
            self.assertEqual(field_again["total_intensity_nt"], field["total_intensity_nt"])


if __name__ == "__main__":
    unittest.main()
