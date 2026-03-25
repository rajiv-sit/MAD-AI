from __future__ import annotations

import unittest

from mad_ai.features import ResidualFeatureBuilder, TemporalFeatureBuilder
from mad_ai.utils.sample_data import make_sample_sensor_data, make_tracked_sensor_data
from mad_ai.visualizer import CesiumGlobeViewerBuilder
from mad_ai.wmm import AnalyticMagneticModel
from tests.unit.helpers import workspace_temp_dir


class CesiumViewerTestCase(unittest.TestCase):
    def test_cesium_viewer_builder_writes_html(self) -> None:
        raw = make_sample_sensor_data(8)
        enriched = ResidualFeatureBuilder(AnalyticMagneticModel()).transform(raw)
        temporal = TemporalFeatureBuilder().transform(enriched)

        with workspace_temp_dir() as tmp_dir:
            output = CesiumGlobeViewerBuilder().build(temporal, tmp_dir / "viewer.html")
            text = output.read_text(encoding="utf-8")
            texture = tmp_dir / "world_texture.svg"
            magnetic_overlay = tmp_dir / "magnetic_overlay_total_field_0_0m.png"
            self.assertTrue(output.exists())
            self.assertTrue(texture.exists())
            self.assertTrue(magnetic_overlay.exists())
            self.assertIn("Cesium.Viewer", text)
            self.assertIn("sampleData", text)
            self.assertIn("componentSelect", text)
            self.assertIn("Displayed Component", text)
            self.assertIn("timeSlider", text)
            self.assertIn("altitudeSlider", text)
            self.assertIn("altitudeLabel", text)
            self.assertIn("selectionInfo", text)
            self.assertIn("Click the earth surface to inspect magnetic details.", text)
            self.assertIn("colorLegend", text)
            self.assertIn("colorLegendMin", text)
            self.assertIn("colorLegendMax", text)
            self.assertIn("Play", text)
            self.assertIn("worldGeoJson", text)
            self.assertIn("worldTexturePath", text)
            self.assertIn("magneticOverlayAssets", text)
            self.assertIn("GeoJsonDataSource.load", text)
            self.assertIn("OpenStreetMapImageryProvider", text)
            self.assertIn("world_texture.svg", text)
            self.assertIn("refreshMagneticOverlay", text)
            self.assertIn("magnetic_overlay_total_field_0_0m.png", text)
            self.assertIn("magneticOverlayLayer.alpha = 0.52", text)
            self.assertIn("viewer.clock.shouldAnimate = true", text)
            self.assertIn("viewer.clock.multiplier = 600", text)
            self.assertIn("spinToggleBtn", text)
            self.assertIn("viewer.clock.onTick.addEventListener(spinCamera)", text)
            self.assertIn("findNearestActivePoint", text)
            self.assertIn("updateSurfaceSelection", text)
            self.assertIn("ScreenSpaceEventHandler", text)
            self.assertIn("pickEllipsoid", text)
            self.assertIn("refreshColorLegend", text)

    def test_cesium_viewer_builder_supports_custom_observed_components(self) -> None:
        raw = make_sample_sensor_data(8)
        enriched = ResidualFeatureBuilder(AnalyticMagneticModel()).transform(raw)
        temporal = TemporalFeatureBuilder().transform(enriched)
        temporal["final_anomaly_score"] = 0.42
        temporal["is_anomaly"] = True

        with workspace_temp_dir() as tmp_dir:
            output = CesiumGlobeViewerBuilder(
                component_columns={
                    "Observed Total": "observed_total_nt",
                    "Baseline Total": "baseline_total_nt",
                    "Residual Total": "residual_total_nt",
                    "Final Anomaly Score": "final_anomaly_score",
                },
                default_component="final_anomaly_score",
                anomaly_flag_column="is_anomaly",
                anomaly_score_column="final_anomaly_score",
            ).build(temporal, tmp_dir / "observed_viewer.html")
            text = output.read_text(encoding="utf-8")
            self.assertIn("Observed Total", text)
            self.assertIn("Baseline Total", text)
            self.assertIn("Residual Total", text)
            self.assertIn("Final Anomaly Score", text)
            self.assertIn("anomalyScore", text)
            self.assertIn("colorLegendTitle", text)

    def test_cesium_viewer_builder_renders_track_paths_when_track_ids_exist(self) -> None:
        raw = make_tracked_sensor_data(track_count=2, samples_per_track=5)
        enriched = ResidualFeatureBuilder(AnalyticMagneticModel()).transform(raw)
        temporal = TemporalFeatureBuilder().transform(enriched)

        with workspace_temp_dir() as tmp_dir:
            output = CesiumGlobeViewerBuilder().build(temporal, tmp_dir / "tracked_viewer.html")
            text = output.read_text(encoding="utf-8")
            self.assertIn("trackId", text)
            self.assertIn("buildTracks", text)
            self.assertIn("getHistoricalTrackPoints", text)
            self.assertIn("polyline", text)
            self.assertIn("Surface selection", text)


if __name__ == "__main__":
    unittest.main()
