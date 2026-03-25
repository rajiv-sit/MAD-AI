from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from mad_ai.viz import AnomalyScatter3DVisualizer, ContourMapVisualizer, Globe3DVisualizer, HeatmapVisualizer, Surface3DVisualizer
from mad_ai.visualizer.viewmodels import AnomalyReviewViewModel, MapViewModel, TimelineViewModel


class VisualizerApp:
    """
    Minimal visualizer foundation.

    This starts with matplotlib-based review windows and can later be upgraded to a
    PySide6 desktop shell without changing the view-model contracts.
    """

    def __init__(
        self,
        map_view_model: MapViewModel | None = None,
        timeline_view_model: TimelineViewModel | None = None,
        anomaly_review_view_model: AnomalyReviewViewModel | None = None,
        include_advanced_visualizations: bool = True,
    ) -> None:
        self.map_view_model = map_view_model or MapViewModel()
        self.timeline_view_model = timeline_view_model or TimelineViewModel()
        self.anomaly_review_view_model = anomaly_review_view_model or AnomalyReviewViewModel()
        self.include_advanced_visualizations = include_advanced_visualizations

    def show(self, data: pd.DataFrame, output_dir: str | Path | None = None) -> list[Path]:
        outputs: list[Path] = []
        target_dir = Path(output_dir) if output_dir is not None else None
        if target_dir is not None:
            target_dir.mkdir(parents=True, exist_ok=True)

        map_state = self.map_view_model.build_view_state(data)
        map_path = self._render_map(
            data=map_state.data,
            value_column=map_state.value_column,
            title=map_state.title,
            output_path=target_dir / "visualizer_map.png" if target_dir is not None else None,
        )
        if map_path is not None:
            outputs.append(map_path)

        review_state = self.anomaly_review_view_model.build_view_state(data)
        residual_map_path = self._render_map(
            data=review_state.data,
            value_column=review_state.residual_column,
            title="Residual Magnetic Map",
            output_path=target_dir / "visualizer_residual_map.png" if target_dir is not None else None,
            cmap="coolwarm",
        )
        if residual_map_path is not None:
            outputs.append(residual_map_path)

        if "timestamp" in data.columns and self.timeline_view_model.y_column in data.columns:
            timeline_state = self.timeline_view_model.build_view_state(data.sort_values("timestamp"))
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(timeline_state.data[timeline_state.x_column], timeline_state.data[timeline_state.y_column], linewidth=1.5)
            if not review_state.events.empty and timeline_state.x_column in review_state.events.columns:
                ax2.scatter(
                    review_state.events[timeline_state.x_column],
                    review_state.events[review_state.residual_column],
                    color="red",
                    s=18,
                    label="Anomaly event",
                )
                ax2.legend()
            ax2.axhline(review_state.threshold, color="orange", linestyle="--", linewidth=1.0)
            ax2.axhline(-review_state.threshold, color="orange", linestyle="--", linewidth=1.0)
            ax2.set_title(timeline_state.title)
            ax2.set_xlabel(timeline_state.x_column)
            ax2.set_ylabel(timeline_state.y_column)
            fig2.tight_layout()
            if target_dir is not None:
                timeline_path = target_dir / "visualizer_timeline.png"
                fig2.savefig(timeline_path, dpi=120)
                outputs.append(timeline_path)
            plt.close(fig2)

        if target_dir is not None:
            events_path = target_dir / "visualizer_anomaly_events.csv"
            review_state.events.to_csv(events_path, index=False)
            outputs.append(events_path)

            if self.include_advanced_visualizations:
                outputs.extend(self._render_advanced_visualizations(review_state.data, target_dir))

        return outputs

    def _render_map(
        self,
        data: pd.DataFrame,
        value_column: str,
        title: str,
        output_path: Path | None,
        cmap: str = "viridis",
    ) -> Path | None:
        pivot = data.pivot_table(
            index="latitude_deg",
            columns="longitude_deg",
            values=value_column,
            aggfunc="mean",
        )
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        im = ax1.imshow(pivot.sort_index(ascending=False).to_numpy(), cmap=cmap, aspect="auto")
        ax1.set_title(title)
        fig1.colorbar(im, ax=ax1, label=value_column)
        fig1.tight_layout()
        saved_path: Path | None = None
        if output_path is not None:
            fig1.savefig(output_path, dpi=120)
            saved_path = output_path
        plt.close(fig1)
        return saved_path

    def _render_advanced_visualizations(self, data: pd.DataFrame, target_dir: Path) -> list[Path]:
        visualizers = [
            HeatmapVisualizer(value_column="baseline_total_nt", title="Baseline Total Field Heatmap"),
            HeatmapVisualizer(value_column="residual_total_nt", title="Residual Field Heatmap", cmap="coolwarm"),
            ContourMapVisualizer(value_column="baseline_total_nt", title="Baseline Total Field Contours"),
            Surface3DVisualizer(value_column="baseline_total_nt", title="Baseline Total Field Surface"),
            Surface3DVisualizer(value_column="residual_total_nt", title="Residual Field Surface", cmap="coolwarm"),
            Globe3DVisualizer(value_column="baseline_total_nt", title="Baseline Magnetic Globe"),
            Globe3DVisualizer(value_column="residual_total_nt", title="Residual Magnetic Globe", cmap="coolwarm"),
            AnomalyScatter3DVisualizer(
                residual_column=self.anomaly_review_view_model.residual_column,
                threshold=self.anomaly_review_view_model.threshold,
            ),
        ]
        names = [
            "baseline_heatmap_2d.png",
            "residual_heatmap_2d.png",
            "baseline_contours_2d.png",
            "baseline_surface_3d.png",
            "residual_surface_3d.png",
            "baseline_globe_3d.png",
            "residual_globe_3d.png",
            "anomaly_scatter_3d.png",
        ]
        outputs: list[Path] = []
        for visualizer, name in zip(visualizers, names, strict=True):
            outputs.append(visualizer.render(data, target_dir / name))
        return outputs
