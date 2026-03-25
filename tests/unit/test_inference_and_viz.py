from __future__ import annotations

import unittest

from mad_ai.inference import AnomalyFusionEngine
from mad_ai.viz import (
    AnomalyReportVisualizer,
    AnomalyScatter3DVisualizer,
    ContourMapVisualizer,
    Globe3DVisualizer,
    HeatmapVisualizer,
    Surface3DVisualizer,
)
from mad_ai.visualizer import AnomalyReviewViewModel, MapViewModel, TimelineViewModel, VisualizerApp
from mad_ai.utils.sample_data import make_sample_sensor_data
from mad_ai.features import ResidualFeatureBuilder, TemporalFeatureBuilder
from mad_ai.wmm import AnalyticMagneticModel
from tests.unit.helpers import workspace_temp_dir


class InferenceAndVizTestCase(unittest.TestCase):
    def test_anomaly_fusion_engine_flags_threshold_crossing(self) -> None:
        engine = AnomalyFusionEngine(spatial_weight=0.25, temporal_weight=0.75, threshold=0.5)
        result = engine.fuse(0.2, 0.8)
        self.assertTrue(result.is_anomaly)
        self.assertAlmostEqual(result.final_score, 0.65)

    def test_heatmap_visualizer_writes_output(self) -> None:
        raw = make_sample_sensor_data(16)
        enriched = ResidualFeatureBuilder(AnalyticMagneticModel()).transform(raw)
        with workspace_temp_dir() as tmp_dir:
            output = HeatmapVisualizer().render(enriched, tmp_dir / "heatmap.png")
            self.assertTrue(output.exists())

    def test_contour_and_3d_visualizers_write_outputs(self) -> None:
        raw = make_sample_sensor_data(24)
        enriched = ResidualFeatureBuilder(AnalyticMagneticModel()).transform(raw)
        temporal = TemporalFeatureBuilder().transform(enriched)
        with workspace_temp_dir() as tmp_dir:
            contour = ContourMapVisualizer().render(temporal, tmp_dir / "contour.png")
            surface = Surface3DVisualizer().render(temporal, tmp_dir / "surface.png")
            globe = Globe3DVisualizer().render(temporal, tmp_dir / "globe.png")
            scatter = AnomalyScatter3DVisualizer().render(temporal, tmp_dir / "scatter.png")
            self.assertTrue(contour.exists())
            self.assertTrue(surface.exists())
            self.assertTrue(globe.exists())
            self.assertTrue(scatter.exists())

    def test_anomaly_report_visualizer_writes_report(self) -> None:
        with workspace_temp_dir() as tmp_dir:
            output = AnomalyReportVisualizer().render(
                [{"spatial_score": 0.1, "temporal_score": 0.2, "final_score": 0.3, "is_anomaly": False}],
                tmp_dir / "report.txt",
            )
            self.assertIn("spatial=0.1000", output.read_text(encoding="utf-8"))

    def test_view_models_and_visualizer_app_produce_outputs(self) -> None:
        raw = make_sample_sensor_data(16)
        enriched = ResidualFeatureBuilder(AnalyticMagneticModel()).transform(raw)
        temporal = TemporalFeatureBuilder().transform(enriched)

        map_state = MapViewModel().build_view_state(temporal)
        timeline_state = TimelineViewModel().build_view_state(temporal)
        review_state = AnomalyReviewViewModel(threshold=100.0).build_view_state(temporal)
        self.assertEqual(map_state.value_column, "baseline_total_nt")
        self.assertEqual(timeline_state.y_column, "residual_total_nt")
        self.assertIn("is_anomaly_event", review_state.data.columns)
        self.assertGreater(len(review_state.events), 0)

        with workspace_temp_dir() as tmp_dir:
            outputs = VisualizerApp().show(temporal, tmp_dir)
            self.assertEqual(len(outputs), 12)
            for output in outputs:
                self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
