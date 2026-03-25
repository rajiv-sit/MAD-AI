from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from mad_ai.core.base import BaseVisualizer


def _pivot_frame(data: pd.DataFrame, value_column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pivot = data.pivot_table(
        index="latitude_deg",
        columns="longitude_deg",
        values=value_column,
        aggfunc="mean",
    ).sort_index(ascending=False)
    latitudes = pivot.index.to_numpy(dtype=float)
    longitudes = pivot.columns.to_numpy(dtype=float)
    values = pivot.to_numpy(dtype=float)
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    return lat_grid, lon_grid, values


def _lat_lon_to_xyz(lat_deg, lon_deg, radius):
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return x, y, z


def _iter_geojson_rings(geojson_path: str | Path):
    path = Path(geojson_path)
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    for feature in payload.get("features", []):
        geometry = feature.get("geometry") or {}
        geom_type = geometry.get("type")
        coords = geometry.get("coordinates", [])
        if geom_type == "Polygon":
            for ring in coords:
                yield ring
        elif geom_type == "MultiPolygon":
            for polygon in coords:
                for ring in polygon:
                    yield ring


class HeatmapVisualizer(BaseVisualizer):
    def __init__(self, value_column: str = "baseline_total_nt", title: str = "Magnetic Heatmap", cmap: str = "viridis") -> None:
        self.value_column = value_column
        self.title = title
        self.cmap = cmap

    def render(self, data, output_path: str | Path) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(data, pd.DataFrame):
            lat_grid, lon_grid, values = _pivot_frame(data, self.value_column)
            extent = [lon_grid.min(), lon_grid.max(), lat_grid.min(), lat_grid.max()]
        else:
            array = np.asarray(data, dtype=float)
            values = array[:, :, 0] if array.ndim == 3 else array
            extent = None

        fig, ax = plt.subplots(figsize=(8, 6))
        image = ax.imshow(values, cmap=self.cmap, aspect="auto", extent=extent, origin="lower" if extent else None)
        ax.set_title(self.title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(image, ax=ax, label=self.value_column)
        fig.tight_layout()
        fig.savefig(output, dpi=140)
        plt.close(fig)
        return output


class ContourMapVisualizer(BaseVisualizer):
    def __init__(self, value_column: str = "baseline_total_nt", title: str = "Contour Map", cmap: str = "viridis", levels: int = 12) -> None:
        self.value_column = value_column
        self.title = title
        self.cmap = cmap
        self.levels = levels

    def render(self, data, output_path: str | Path) -> Path:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("ContourMapVisualizer requires a DataFrame input.")
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        lat_grid, lon_grid, values = _pivot_frame(data, self.value_column)
        fig, ax = plt.subplots(figsize=(8, 6))
        contour = ax.contourf(lon_grid, lat_grid, values, levels=self.levels, cmap=self.cmap)
        ax.contour(lon_grid, lat_grid, values, levels=self.levels, colors="black", linewidths=0.4, alpha=0.45)
        ax.set_title(self.title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(contour, ax=ax, label=self.value_column)
        fig.tight_layout()
        fig.savefig(output, dpi=140)
        plt.close(fig)
        return output


class Surface3DVisualizer(BaseVisualizer):
    def __init__(self, value_column: str = "baseline_total_nt", title: str = "3D Surface", cmap: str = "viridis") -> None:
        self.value_column = value_column
        self.title = title
        self.cmap = cmap

    def render(self, data, output_path: str | Path) -> Path:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Surface3DVisualizer requires a DataFrame input.")
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        lat_grid, lon_grid, values = _pivot_frame(data, self.value_column)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        surface = ax.plot_surface(lon_grid, lat_grid, values, cmap=self.cmap, linewidth=0, antialiased=True)
        ax.set_title(self.title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_zlabel(self.value_column)
        fig.colorbar(surface, ax=ax, shrink=0.6, pad=0.1)
        fig.tight_layout()
        fig.savefig(output, dpi=140)
        plt.close(fig)
        return output


class Globe3DVisualizer(BaseVisualizer):
    def __init__(
        self,
        value_column: str = "baseline_total_nt",
        title: str = "3D Magnetic Globe",
        cmap: str = "viridis",
        radius: float = 1.0,
        point_size: float = 18.0,
        geojson_path: str | Path | None = "data/raw/world_countries_110m.geojson",
    ) -> None:
        self.value_column = value_column
        self.title = title
        self.cmap = cmap
        self.radius = radius
        self.point_size = point_size
        self.geojson_path = Path(geojson_path) if geojson_path is not None else None

    def render(self, data, output_path: str | Path) -> Path:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Globe3DVisualizer requires a DataFrame input.")
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Globe shell
        u = np.linspace(0, 2 * np.pi, 72)
        v = np.linspace(-np.pi / 2, np.pi / 2, 36)
        uu, vv = np.meshgrid(u, v)
        xs = self.radius * np.cos(vv) * np.cos(uu)
        ys = self.radius * np.cos(vv) * np.sin(uu)
        zs = self.radius * np.sin(vv)
        ax.plot_surface(xs, ys, zs, color="#d9e7f5", alpha=0.15, linewidth=0, shade=False)

        # Graticules
        for lat in range(-60, 90, 30):
            lon = np.linspace(-180, 180, 240)
            gx, gy, gz = _lat_lon_to_xyz(lat, lon, self.radius * 1.001)
            ax.plot(gx, gy, gz, color="gray", linewidth=0.35, alpha=0.35)
        for lon in range(-150, 180, 30):
            lat = np.linspace(-90, 90, 240)
            gx, gy, gz = _lat_lon_to_xyz(lat, lon, self.radius * 1.001)
            ax.plot(gx, gy, gz, color="gray", linewidth=0.35, alpha=0.35)

        if self.geojson_path is not None:
            for ring in _iter_geojson_rings(self.geojson_path):
                if not ring:
                    continue
                ring_array = np.asarray(ring, dtype=float)
                lon = ring_array[:, 0]
                lat = ring_array[:, 1]
                gx, gy, gz = _lat_lon_to_xyz(lat, lon, self.radius * 1.003)
                ax.plot(gx, gy, gz, color="#3e4a59", linewidth=0.45, alpha=0.55)

        values = data[self.value_column].to_numpy(dtype=float)
        span = np.ptp(values)
        x, y, z = _lat_lon_to_xyz(data["latitude_deg"].to_numpy(), data["longitude_deg"].to_numpy(), self.radius * 1.02)
        scatter = ax.scatter(x, y, z, c=values, cmap=self.cmap, s=self.point_size, depthshade=True)

        ax.set_title(self.title)
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.05, label=self.value_column)
        fig.tight_layout()
        fig.savefig(output, dpi=160)
        plt.close(fig)
        return output


class AnomalyScatter3DVisualizer(BaseVisualizer):
    def __init__(self, residual_column: str = "residual_total_nt", threshold: float = 250.0, title: str = "3D Anomaly Scatter") -> None:
        self.residual_column = residual_column
        self.threshold = threshold
        self.title = title

    def render(self, data, output_path: str | Path) -> Path:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("AnomalyScatter3DVisualizer requires a DataFrame input.")
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        review = data.copy()
        review["is_anomaly_event"] = review[self.residual_column].abs() >= self.threshold

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        normal = review.loc[~review["is_anomaly_event"]]
        anomalous = review.loc[review["is_anomaly_event"]]

        if not normal.empty:
            ax.scatter(
                normal["longitude_deg"],
                normal["latitude_deg"],
                normal[self.residual_column],
                c="steelblue",
                s=16,
                alpha=0.6,
                label="Normal",
            )
        if not anomalous.empty:
            ax.scatter(
                anomalous["longitude_deg"],
                anomalous["latitude_deg"],
                anomalous[self.residual_column],
                c="crimson",
                s=28,
                alpha=0.9,
                label="Anomaly",
            )

        ax.set_title(self.title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_zlabel(self.residual_column)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output, dpi=140)
        plt.close(fig)
        return output


class AnomalyReportVisualizer(BaseVisualizer):
    def render(self, data, output_path: str | Path) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for item in data:
            rows.append(
                f"spatial={item['spatial_score']:.4f}, temporal={item['temporal_score']:.4f}, "
                f"final={item['final_score']:.4f}, anomaly={item['is_anomaly']}"
            )
        output.write_text("\n".join(rows), encoding="utf-8")
        return output


class RealtimeAnomalyDashboardVisualizer(BaseVisualizer):
    def __init__(
        self,
        title: str = "Bahamas Realtime Magnetic Anomaly Dashboard",
        observed_column: str = "observed_total_nt",
        baseline_column: str = "baseline_total_nt",
        residual_column: str = "residual_total_nt",
        anomaly_score_column: str = "final_anomaly_score",
        threshold_column: str | None = None,
        range_column: str = "range_to_vessel_m",
    ) -> None:
        self.title = title
        self.observed_column = observed_column
        self.baseline_column = baseline_column
        self.residual_column = residual_column
        self.anomaly_score_column = anomaly_score_column
        self.threshold_column = threshold_column
        self.range_column = range_column

    def render(self, data, output_path: str | Path) -> Path:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("RealtimeAnomalyDashboardVisualizer requires a DataFrame input.")

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        frame = data.sort_values("timestamp").copy()
        time_axis = pd.to_datetime(frame["timestamp"]) if "timestamp" in frame.columns else np.arange(len(frame))
        threshold_value = None
        if self.threshold_column and self.threshold_column in frame.columns:
            threshold_value = float(frame[self.threshold_column].iloc[0])

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        axes[0].plot(time_axis, frame[self.observed_column], color="#ffd166", linewidth=1.1, label="Observed")
        axes[0].plot(time_axis, frame[self.baseline_column], color="#4fc3f7", linewidth=1.1, label="NOAA/WMM Baseline")
        axes[0].set_ylabel("Field (nT)")
        axes[0].set_title(self.title)
        axes[0].legend(loc="upper right")
        axes[0].grid(alpha=0.25)

        axes[1].plot(time_axis, frame[self.residual_column], color="#ff7b72", linewidth=1.0, label="Residual")
        score_axis = axes[1].twinx()
        score_axis.plot(time_axis, frame[self.anomaly_score_column], color="#7ee787", linewidth=1.0, label="Final Anomaly Score")
        if threshold_value is not None:
            score_axis.axhline(threshold_value, color="#d2a8ff", linestyle="--", linewidth=1.0, label="Threshold")
        axes[1].set_ylabel("Residual (nT)")
        score_axis.set_ylabel("Score")
        axes[1].grid(alpha=0.25)
        residual_lines, residual_labels = axes[1].get_legend_handles_labels()
        score_lines, score_labels = score_axis.get_legend_handles_labels()
        axes[1].legend(residual_lines + score_lines, residual_labels + score_labels, loc="upper right")

        if self.range_column in frame.columns:
            axes[2].plot(time_axis, frame[self.range_column], color="#58a6ff", linewidth=1.0, label="Range To Vessel")
            axes[2].invert_yaxis()
            axes[2].set_ylabel("Range (m)")
            axes[2].legend(loc="upper right")
        else:
            axes[2].text(0.5, 0.5, "Range-to-vessel data not available", transform=axes[2].transAxes, ha="center", va="center")
        axes[2].set_xlabel("Time")
        axes[2].grid(alpha=0.25)

        fig.tight_layout()
        fig.savefig(output, dpi=160)
        plt.close(fig)
        return output
