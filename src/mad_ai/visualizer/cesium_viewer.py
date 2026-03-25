from __future__ import annotations

import json
from pathlib import Path

from matplotlib import colormaps
import numpy as np
import pandas as pd


class CesiumGlobeViewerBuilder:
    def __init__(
        self,
        title: str = "MAD-AI Magnetic Globe",
        residual_column: str = "residual_total_nt",
        baseline_column: str = "baseline_total_nt",
        anomaly_threshold: float = 250.0,
        component_columns: dict[str, str] | None = None,
        default_component: str = "baseline_total_nt",
        anomaly_flag_column: str = "is_anomaly",
        anomaly_score_column: str = "final_anomaly_score",
        track_id_column: str = "track_id",
        world_geojson_path: str | Path | None = "data/raw/world_countries_110m.geojson",
    ) -> None:
        self.title = title
        self.residual_column = residual_column
        self.baseline_column = baseline_column
        self.anomaly_threshold = anomaly_threshold
        self.component_columns = component_columns or {
            "Total Field": "baseline_total_nt",
            "Declination": "baseline_declination_deg",
            "Inclination": "baseline_inclination_deg",
            "Residual": "residual_total_nt",
        }
        self.default_component = default_component
        self.anomaly_flag_column = anomaly_flag_column
        self.anomaly_score_column = anomaly_score_column
        self.track_id_column = track_id_column
        self.world_geojson_path = Path(world_geojson_path) if world_geojson_path is not None else None

    def build(self, data: pd.DataFrame, output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._to_payload(data)
        world_texture_filename = self._write_world_texture_asset(output_path)
        overlay_assets = self._write_magnetic_overlay_assets(data, output_path)
        output_path.write_text(self._render_html(payload, world_texture_filename, overlay_assets), encoding="utf-8")
        return output_path

    def _to_payload(self, data: pd.DataFrame) -> list[dict[str, float | str | bool]]:
        rows = []
        for row in data.itertuples(index=False):
            residual = float(getattr(row, self.residual_column, 0.0))
            components: dict[str, float] = {}
            for label, column in self.component_columns.items():
                if hasattr(row, column):
                    components[label] = float(getattr(row, column))
            if hasattr(row, self.anomaly_score_column):
                components["Anomaly Score"] = float(getattr(row, self.anomaly_score_column))
            rows.append(
                {
                    "lat": float(row.latitude_deg),
                    "lon": float(row.longitude_deg),
                    "alt": float(getattr(row, "altitude_m", 0.0)),
                    "baseline": float(getattr(row, self.baseline_column, 0.0)),
                    "residual": residual,
                    "components": components,
                    "timestamp": str(getattr(row, "timestamp", "")),
                    "trackId": str(getattr(row, self.track_id_column, "")),
                    "anomalyScore": float(getattr(row, self.anomaly_score_column, 0.0))
                    if hasattr(row, self.anomaly_score_column)
                    else 0.0,
                    "isAnomaly": bool(getattr(row, self.anomaly_flag_column))
                    if hasattr(row, self.anomaly_flag_column)
                    else abs(residual) >= self.anomaly_threshold,
                }
            )
        return rows

    def _render_html(
        self,
        payload: list[dict[str, float | str | bool]],
        world_texture_filename: str | None,
        overlay_assets: dict[str, dict[str, str]],
    ) -> str:
        data_json = json.dumps(payload)
        world_geojson = self._load_world_geojson()
        threshold = self.anomaly_threshold
        title = self.title
        component_options = list(self.component_columns.keys())
        component_json = json.dumps(component_options)
        world_geojson_json = json.dumps(world_geojson)
        world_texture_json = json.dumps(world_texture_filename)
        overlay_assets_json = json.dumps(overlay_assets)
        default_label = next(
            (label for label, column in self.component_columns.items() if column == self.default_component),
            component_options[0],
        )
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{title}</title>
  <link href="https://cesium.com/downloads/cesiumjs/releases/1.118/Build/Cesium/Widgets/widgets.css" rel="stylesheet">
  <script src="https://cesium.com/downloads/cesiumjs/releases/1.118/Build/Cesium/Cesium.js"></script>
  <style>
    html, body, #cesiumContainer {{ width: 100%; height: 100%; margin: 0; overflow: hidden; background: #07111f; }}
    #panel {{
      position: absolute; top: 12px; left: 12px; z-index: 10; width: 320px;
      background: rgba(8, 16, 28, 0.84); color: #dfe9f6; padding: 14px 16px; border-radius: 12px;
      font-family: Segoe UI, sans-serif; backdrop-filter: blur(8px); box-shadow: 0 8px 28px rgba(0,0,0,0.28);
    }}
    #panel h1 {{ margin: 0 0 8px 0; font-size: 18px; }}
    #panel p {{ margin: 0 0 6px 0; font-size: 13px; line-height: 1.4; }}
    .legend {{ display: flex; gap: 10px; margin-top: 10px; font-size: 12px; align-items: center; flex-wrap: wrap; }}
    .swatch {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; }}
    select {{
      margin-top: 8px; width: 100%; padding: 8px 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.12);
      background: rgba(255,255,255,0.08); color: #eef5ff;
    }}
    input[type="range"] {{ width: 100%; margin-top: 8px; }}
    .time-row {{ margin-top: 10px; font-size: 12px; }}
    button {{
      margin-top: 8px; margin-right: 6px; padding: 6px 10px; border-radius: 8px; border: 0;
      background: #1f7ae0; color: white; cursor: pointer;
    }}
    #colorLegend {{
      margin-top: 12px;
      padding-top: 10px;
      border-top: 1px solid rgba(255,255,255,0.12);
    }}
    #colorLegendTitle {{
      font-size: 12px;
      margin-bottom: 6px;
    }}
    #colorLegendBar {{
      height: 14px;
      border-radius: 999px;
      background: linear-gradient(90deg, #30123b 0%, #4666dd 20%, #35b779 50%, #fde725 75%, #a50026 100%);
      border: 1px solid rgba(255,255,255,0.16);
    }}
    #colorLegendScale {{
      display: flex;
      justify-content: space-between;
      font-size: 11px;
      margin-top: 6px;
      color: #c6d5e6;
    }}
  </style>
</head>
<body>
  <div id="panel">
    <h1>{title}</h1>
    <p>Interactive Cesium globe adapted from the ConstellationVizFrontend approach.</p>
    <p>Magnetic values are draped across the full earth surface. Click the globe to inspect the nearest magnetic sample at the active time and altitude.</p>
    <label for="componentSelect">Displayed Component</label>
    <select id="componentSelect"></select>
    <div class="time-row">
      <label for="timeSlider">Time Slice</label>
      <input id="timeSlider" type="range" min="0" max="0" value="0" step="1" />
      <div id="timeLabel">Single timestamp</div>
      <button id="playPauseBtn" type="button">Play</button>
    </div>
    <div class="time-row">
      <label for="altitudeSlider">Altitude Layer</label>
      <input id="altitudeSlider" type="range" min="0" max="0" value="0" step="1" />
      <div id="altitudeLabel">0.0 m</div>
    </div>
    <div class="time-row">
      <button id="spinToggleBtn" type="button">Pause Earth Spin</button>
    </div>
    <div class="time-row">
      <div id="selectionInfo">Click the earth surface to inspect magnetic details.</div>
    </div>
    <div id="colorLegend">
      <div id="colorLegendTitle">Overlay Scale</div>
      <div id="colorLegendBar"></div>
      <div id="colorLegendScale">
        <span id="colorLegendMin">min</span>
        <span id="colorLegendMid">mid</span>
        <span id="colorLegendMax">max</span>
      </div>
    </div>
    <div class="legend">
      <span><span class="swatch" style="background:#4fc3f7"></span> Magnetic overlay</span>
      <span><span class="swatch" style="background:#ffd166"></span> Track path</span>
      <span><span class="swatch" style="background:#ffffff"></span> Surface selection</span>
    </div>
  </div>
  <div id="cesiumContainer"></div>
  <script>
    const sampleData = {data_json};
    const worldGeoJson = {world_geojson_json};
    const worldTexturePath = {world_texture_json};
    const magneticOverlayAssets = {overlay_assets_json};
    const componentOptions = {component_json};
    let currentComponent = "{default_label}";
    const groupedByTime = new Map();
    sampleData.forEach((point) => {{
      const key = point.timestamp || "Single timestamp";
      if (!groupedByTime.has(key)) groupedByTime.set(key, []);
      groupedByTime.get(key).push(point);
    }});
    const parseTimeKey = (value) => {{
      const parsed = Date.parse(value);
      return Number.isNaN(parsed) ? 0 : parsed;
    }};
    const timeKeys = Array.from(groupedByTime.keys()).sort((left, right) => parseTimeKey(left) - parseTimeKey(right));
    const altitudeKeys = Array.from(new Set(sampleData.map((point) => Number(point.alt).toFixed(1))));
    let currentTimeIndex = 0;
    let currentAltitudeKey = altitudeKeys[0] || "0.0";
    let playTimer = null;
    let magneticOverlayLayer = null;
    Cesium.Ion.defaultAccessToken = "";
    const viewer = new Cesium.Viewer("cesiumContainer", {{
      animation: false,
      timeline: false,
      baseLayerPicker: false,
      geocoder: false,
      imageryProvider: false,
      sceneMode: Cesium.SceneMode.SCENE3D
    }});

    const globe = viewer.scene.globe;
    globe.baseColor = Cesium.Color.fromCssColorString("#0f2d40");
    globe.enableLighting = true;
    globe.dynamicAtmosphereLighting = true;
    globe.dynamicAtmosphereLightingFromSun = true;
    globe.showGroundAtmosphere = true;
    viewer.scene.skyAtmosphere.show = true;
    viewer.scene.sun.show = true;
    viewer.scene.moon.show = true;
    viewer.scene.light = new Cesium.SunLight();
    viewer.shadows = true;
    viewer.terrainShadows = Cesium.ShadowMode.RECEIVE_ONLY;
    viewer.clock.clockRange = Cesium.ClockRange.UNBOUNDED;
    viewer.clock.multiplier = 600;
    viewer.clock.shouldAnimate = true;
    viewer.clock.currentTime = Cesium.JulianDate.fromDate(new Date());
    let globeSpinEnabled = true;
    let lastSpinTime = viewer.clock.currentTime;

    try {{
      const layers = viewer.scene.globe.imageryLayers;
      layers.removeAll();
      layers.addImageryProvider(new Cesium.OpenStreetMapImageryProvider({{
        url: "https://a.tile.openstreetmap.org/"
      }}));
    }} catch (err) {{
      console.warn("OpenStreetMap imagery failed", err);
      if (worldTexturePath) {{
        try {{
          viewer.scene.globe.imageryLayers.addImageryProvider(new Cesium.SingleTileImageryProvider({{
            url: worldTexturePath,
            rectangle: Cesium.Rectangle.fromDegrees(-180.0, -90.0, 180.0, 90.0)
          }}));
        }} catch (fallbackErr) {{
          console.warn("Local texture fallback failed", fallbackErr);
        }}
      }}
    }}

    function refreshMagneticOverlay() {{
      const layers = viewer.scene.globe.imageryLayers;
      if (magneticOverlayLayer) {{
        layers.remove(magneticOverlayLayer, true);
        magneticOverlayLayer = null;
      }}
      const overlayPath = magneticOverlayAssets[currentComponent]?.[currentAltitudeKey];
      if (!overlayPath) {{
        return;
      }}
      magneticOverlayLayer = layers.addImageryProvider(new Cesium.SingleTileImageryProvider({{
        url: overlayPath,
        rectangle: Cesium.Rectangle.fromDegrees(-180.0, -90.0, 180.0, 90.0)
      }}));
      magneticOverlayLayer.alpha = 0.52;
      refreshColorLegend();
    }}

    if (worldGeoJson) {{
      Cesium.GeoJsonDataSource.load(worldGeoJson, {{
        stroke: Cesium.Color.fromCssColorString("#dfe9f6"),
        fill: Cesium.Color.fromCssColorString("rgba(0, 0, 0, 0.0)"),
        strokeWidth: 0.8,
        clampToGround: false
      }}).then((dataSource) => {{
        viewer.dataSources.add(dataSource);
        const entities = dataSource.entities.values;
        entities.forEach((entity) => {{
          if (entity.polygon) {{
            entity.polygon = undefined;
          }}
          if (entity.polyline) {{
            entity.polyline.material = Cesium.Color.fromCssColorString("rgba(223, 233, 246, 0.35)");
            entity.polyline.width = 0.7;
            entity.polyline.clampToGround = false;
          }}
        }});
      }}).catch((err) => {{
        console.warn("World GeoJSON overlay failed", err);
      }});
    }}

    function getComponentValue(point) {{
      return point.components[currentComponent] ?? point.baseline;
    }}

    function refreshColorLegend() {{
      const activePoints = getActivePoints();
      const values = activePoints
        .map((point) => Number(getComponentValue(point)))
        .filter((value) => Number.isFinite(value));
      const titleEl = document.getElementById("colorLegendTitle");
      const minEl = document.getElementById("colorLegendMin");
      const midEl = document.getElementById("colorLegendMid");
      const maxEl = document.getElementById("colorLegendMax");
      titleEl.textContent = `${{currentComponent}} Scale`;
      if (values.length === 0) {{
        minEl.textContent = "n/a";
        midEl.textContent = "n/a";
        maxEl.textContent = "n/a";
        return;
      }}
      const minValue = Math.min(...values);
      const maxValue = Math.max(...values);
      const midValue = (minValue + maxValue) / 2.0;
      minEl.textContent = minValue.toFixed(2);
      midEl.textContent = midValue.toFixed(2);
      maxEl.textContent = maxValue.toFixed(2);
    }}

    function spinCamera(clock) {{
      if (!globeSpinEnabled) {{
        lastSpinTime = clock.currentTime;
        return;
      }}
      const elapsedSeconds = Cesium.JulianDate.secondsDifference(clock.currentTime, lastSpinTime);
      lastSpinTime = clock.currentTime;
      const spinRateRadiansPerSecond = (2.0 * Math.PI) / 86400.0;
      viewer.scene.camera.rotate(Cesium.Cartesian3.UNIT_Z, -spinRateRadiansPerSecond * elapsedSeconds);
    }}

    function getActivePoints() {{
      const activeTimePoints = groupedByTime.get(timeKeys[currentTimeIndex]) || [];
      return activeTimePoints.filter((point) => Number(point.alt).toFixed(1) === currentAltitudeKey);
    }}

    function descriptionForPoint(point) {{
      const value = getComponentValue(point);
      const componentRows = Object.entries(point.components).map(([name, componentValue]) =>
        `<p><b>${{name}}:</b> ${{Number(componentValue).toFixed(2)}}</p>`
      ).join("");
      return `
          <h3>Magnetic Sample</h3>
          <p><b>Latitude:</b> ${{point.lat.toFixed(3)}}</p>
          <p><b>Longitude:</b> ${{point.lon.toFixed(3)}}</p>
          <p><b>Altitude:</b> ${{point.alt.toFixed(1)}} m</p>
          <p><b>Current Component:</b> ${{currentComponent}} = ${{Number(value).toFixed(2)}}</p>
          <p><b>Baseline:</b> ${{point.baseline.toFixed(2)}} nT</p>
          <p><b>Residual:</b> ${{point.residual.toFixed(2)}} nT</p>
          <p><b>Anomaly Score:</b> ${{Number(point.anomalyScore).toFixed(4)}}</p>
          <p><b>Anomaly:</b> ${{point.isAnomaly}}</p>
          <p><b>Timestamp:</b> ${{point.timestamp}}</p>
          <p><b>Track:</b> ${{point.trackId || "Untracked"}}</p>
          ${{componentRows}}
      `;
    }}
    let trackEntities = [];
    let selectionEntity = null;

    function getHistoricalTrackPoints() {{
      const visibleTimeKeys = new Set(timeKeys.slice(0, currentTimeIndex + 1));
      return sampleData.filter((point) =>
        point.trackId &&
        visibleTimeKeys.has(point.timestamp || "Single timestamp") &&
        Number(point.alt).toFixed(1) === currentAltitudeKey
      );
    }}

    function clearDynamicEntities() {{
      trackEntities.forEach((entity) => viewer.entities.remove(entity));
      trackEntities = [];
      if (selectionEntity) {{
        viewer.entities.remove(selectionEntity);
        selectionEntity = null;
      }}
    }}

    function parseTimeValue(point) {{
      return parseTimeKey(point.timestamp || "");
    }}

    function buildTracks() {{
      const historicalPoints = getHistoricalTrackPoints();
      const groupedByTrack = new Map();
      historicalPoints.forEach((point) => {{
        if (!point.trackId) return;
        if (!groupedByTrack.has(point.trackId)) groupedByTrack.set(point.trackId, []);
        groupedByTrack.get(point.trackId).push(point);
      }});

      trackEntities = Array.from(groupedByTrack.entries())
        .filter(([, points]) => points.length >= 2)
        .map(([trackId, points], index) => {{
          const orderedPoints = points.slice().sort((left, right) => parseTimeValue(left) - parseTimeValue(right));
          const positions = orderedPoints.map((point) => Cesium.Cartesian3.fromDegrees(point.lon, point.lat, point.alt));
          return viewer.entities.add({{
            id: `track_${{index}}_${{trackId}}`,
            name: `Track ${{trackId}}`,
            polyline: {{
              positions,
              width: 3,
              material: Cesium.Color.fromCssColorString("#ffd166"),
              clampToGround: false
            }},
            description: `<h3>Track ${{trackId}}</h3><p><b>Visible History Samples:</b> ${{orderedPoints.length}}</p>`
          }});
        }});
    }}

    function refreshScene() {{
      clearDynamicEntities();
      buildTracks();
      document.getElementById("timeLabel").textContent = timeKeys[currentTimeIndex];
      refreshColorLegend();
    }}

    function describePointForPanel(point, clickedLat, clickedLon) {{
      const currentValue = getComponentValue(point);
      const componentRows = Object.entries(point.components)
        .map(([name, componentValue]) => `<div><b>${{name}}:</b> ${{Number(componentValue).toFixed(2)}}</div>`)
        .join("");
      return `
        <div><b>Clicked Latitude:</b> ${{clickedLat.toFixed(3)}}</div>
        <div><b>Clicked Longitude:</b> ${{clickedLon.toFixed(3)}}</div>
        <div><b>Nearest Sample Latitude:</b> ${{point.lat.toFixed(3)}}</div>
        <div><b>Nearest Sample Longitude:</b> ${{point.lon.toFixed(3)}}</div>
        <div><b>Altitude:</b> ${{point.alt.toFixed(1)}} m</div>
        <div><b>Current Component:</b> ${{currentComponent}} = ${{Number(currentValue).toFixed(2)}}</div>
        <div><b>Baseline:</b> ${{point.baseline.toFixed(2)}} nT</div>
        <div><b>Residual:</b> ${{point.residual.toFixed(2)}} nT</div>
        <div><b>Anomaly Score:</b> ${{Number(point.anomalyScore).toFixed(4)}}</div>
        <div><b>Anomaly:</b> ${{point.isAnomaly}}</div>
        <div><b>Timestamp:</b> ${{point.timestamp}}</div>
        ${{componentRows}}
      `;
    }}

    function angularDistanceDegrees(latA, lonA, latB, lonB) {{
      const toRadians = (degrees) => degrees * Math.PI / 180.0;
      const lat1 = toRadians(latA);
      const lat2 = toRadians(latB);
      const dLat = lat2 - lat1;
      const dLon = toRadians(lonB - lonA);
      const sinLat = Math.sin(dLat / 2.0);
      const sinLon = Math.sin(dLon / 2.0);
      const haversine = sinLat * sinLat + Math.cos(lat1) * Math.cos(lat2) * sinLon * sinLon;
      return 2.0 * Math.atan2(Math.sqrt(haversine), Math.sqrt(Math.max(0.0, 1.0 - haversine)));
    }}

    function findNearestActivePoint(lat, lon) {{
      const activePoints = getActivePoints();
      if (activePoints.length === 0) {{
        return null;
      }}
      let bestPoint = activePoints[0];
      let bestDistance = angularDistanceDegrees(lat, lon, bestPoint.lat, bestPoint.lon);
      activePoints.slice(1).forEach((point) => {{
        const distance = angularDistanceDegrees(lat, lon, point.lat, point.lon);
        if (distance < bestDistance) {{
          bestDistance = distance;
          bestPoint = point;
        }}
      }});
      return bestPoint;
    }}

    function updateSurfaceSelection(cartographic) {{
      const clickedLat = Cesium.Math.toDegrees(cartographic.latitude);
      const clickedLon = Cesium.Math.toDegrees(cartographic.longitude);
      const nearestPoint = findNearestActivePoint(clickedLat, clickedLon);
      const selectionInfo = document.getElementById("selectionInfo");
      if (!nearestPoint) {{
        selectionInfo.textContent = "No magnetic sample is available for the current time and altitude.";
        return;
      }}
      selectionInfo.innerHTML = describePointForPanel(nearestPoint, clickedLat, clickedLon);
      if (selectionEntity) {{
        viewer.entities.remove(selectionEntity);
      }}
      selectionEntity = viewer.entities.add({{
        position: Cesium.Cartesian3.fromRadians(cartographic.longitude, cartographic.latitude, nearestPoint.alt),
        point: {{
          pixelSize: 11,
          color: Cesium.Color.WHITE,
          outlineColor: Cesium.Color.BLACK,
          outlineWidth: 2
        }},
        description: descriptionForPoint(nearestPoint)
      }});
      viewer.selectedEntity = selectionEntity;
    }}

    const componentSelect = document.getElementById("componentSelect");
    componentOptions.forEach((label) => {{
      const option = document.createElement("option");
      option.value = label;
      option.textContent = label;
      option.selected = label === currentComponent;
      componentSelect.appendChild(option);
    }});
    componentSelect.addEventListener("change", (event) => {{
      currentComponent = event.target.value;
      refreshMagneticOverlay();
      refreshScene();
    }});

    const timeSlider = document.getElementById("timeSlider");
    timeSlider.max = String(Math.max(0, timeKeys.length - 1));
    timeSlider.value = "0";
    timeSlider.addEventListener("input", (event) => {{
      currentTimeIndex = Number(event.target.value);
      refreshScene();
    }});

    const altitudeSlider = document.getElementById("altitudeSlider");
    const altitudeLabel = document.getElementById("altitudeLabel");
    altitudeSlider.max = String(Math.max(0, altitudeKeys.length - 1));
    altitudeSlider.value = "0";
    altitudeLabel.textContent = `${{currentAltitudeKey}} m`;
    altitudeSlider.addEventListener("input", (event) => {{
      currentAltitudeKey = altitudeKeys[Number(event.target.value)] || altitudeKeys[0] || "0.0";
      altitudeLabel.textContent = `${{currentAltitudeKey}} m`;
      refreshMagneticOverlay();
      refreshScene();
    }});

    const playPauseBtn = document.getElementById("playPauseBtn");
    playPauseBtn.addEventListener("click", () => {{
      if (playTimer) {{
        clearInterval(playTimer);
        playTimer = null;
        playPauseBtn.textContent = "Play";
        return;
      }}
      playTimer = setInterval(() => {{
        currentTimeIndex = (currentTimeIndex + 1) % timeKeys.length;
        timeSlider.value = String(currentTimeIndex);
        refreshScene();
      }}, 900);
      playPauseBtn.textContent = "Pause";
    }});

    const spinToggleBtn = document.getElementById("spinToggleBtn");
    spinToggleBtn.addEventListener("click", () => {{
      globeSpinEnabled = !globeSpinEnabled;
      spinToggleBtn.textContent = globeSpinEnabled ? "Pause Earth Spin" : "Resume Earth Spin";
      lastSpinTime = viewer.clock.currentTime;
    }});

    viewer.clock.onTick.addEventListener(spinCamera);
    const clickHandler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
    clickHandler.setInputAction((movement) => {{
      const ellipsoid = viewer.scene.globe.ellipsoid;
      const cartesian = viewer.camera.pickEllipsoid(movement.position, ellipsoid);
      if (!cartesian) {{
        return;
      }}
      const cartographic = Cesium.Cartographic.fromCartesian(cartesian);
      updateSurfaceSelection(cartographic);
    }}, Cesium.ScreenSpaceEventType.LEFT_CLICK);

    refreshMagneticOverlay();
    refreshScene();

    if (sampleData.length > 0) {{
      viewer.flyTo(viewer.entities, {{
        duration: 1.8
      }});
    }}
  </script>
</body>
</html>
"""

    def _load_world_geojson(self) -> dict | None:
        if self.world_geojson_path is None or not self.world_geojson_path.exists():
            return None
        return json.loads(self.world_geojson_path.read_text(encoding="utf-8"))

    def _write_world_texture_asset(self, output_path: Path) -> str | None:
        world_geojson = self._load_world_geojson()
        if not world_geojson:
            return None

        texture_path = output_path.with_name("world_texture.svg")
        texture_path.write_text(self._build_world_texture_svg(world_geojson), encoding="utf-8")
        return texture_path.name

    def _write_magnetic_overlay_assets(self, data: pd.DataFrame, output_path: Path) -> dict[str, dict[str, str]]:
        assets: dict[str, dict[str, str]] = {}
        if "latitude_deg" not in data.columns or "longitude_deg" not in data.columns:
            return assets

        altitude_groups = {
            altitude_key: group.copy()
            for altitude_key, group in data.groupby(data["altitude_m"].map(lambda value: f"{float(value):.1f}"))
        }

        for label, column in self.component_columns.items():
            if column not in data.columns:
                continue
            assets[label] = {}
            for altitude_key, altitude_frame in altitude_groups.items():
                overlay = self._build_overlay_image(altitude_frame, column)
                if overlay is None:
                    continue
                asset_name = f"magnetic_overlay_{self._slugify(label)}_{self._slugify(altitude_key)}m.png"
                asset_path = output_path.with_name(asset_name)
                self._save_overlay_image(overlay, asset_path)
                assets[label][altitude_key] = asset_path.name
        return assets

    def _build_overlay_image(self, data: pd.DataFrame, value_column: str) -> np.ndarray | None:
        frame = data[["latitude_deg", "longitude_deg", value_column]].dropna().copy()
        if frame.empty:
            return None

        grouped = frame.groupby(["latitude_deg", "longitude_deg"], as_index=False)[value_column].mean()
        pivot = grouped.pivot(index="latitude_deg", columns="longitude_deg", values=value_column).sort_index().sort_index(axis=1)
        pivot = pivot.interpolate(axis=0, limit_direction="both").interpolate(axis=1, limit_direction="both")
        pivot = pivot.ffill(axis=0).bfill(axis=0).ffill(axis=1).bfill(axis=1)
        values = pivot.to_numpy(dtype=float)
        if values.ndim != 2 or values.size == 0:
            return None
        if np.isnan(values).all():
            return None

        values = np.flipud(values)
        lat_target = 720
        lon_target = 1440
        row_positions = np.linspace(0, values.shape[0] - 1, lat_target)
        col_positions = np.linspace(0, values.shape[1] - 1, lon_target)
        resampled = self._bilinear_resample(values, row_positions, col_positions)

        vmin = float(np.nanpercentile(resampled, 2.0))
        vmax = float(np.nanpercentile(resampled, 98.0))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0
        normalized = np.clip((resampled - vmin) / (vmax - vmin), 0.0, 1.0)
        rgba = colormaps["turbo"](normalized)
        rgba[..., 3] = 0.68
        return rgba

    def _bilinear_resample(self, values: np.ndarray, row_positions: np.ndarray, col_positions: np.ndarray) -> np.ndarray:
        row_floor = np.floor(row_positions).astype(int)
        row_ceil = np.clip(row_floor + 1, 0, values.shape[0] - 1)
        col_floor = np.floor(col_positions).astype(int)
        col_ceil = np.clip(col_floor + 1, 0, values.shape[1] - 1)

        row_weight = row_positions - row_floor
        col_weight = col_positions - col_floor

        top_left = values[row_floor[:, None], col_floor[None, :]]
        top_right = values[row_floor[:, None], col_ceil[None, :]]
        bottom_left = values[row_ceil[:, None], col_floor[None, :]]
        bottom_right = values[row_ceil[:, None], col_ceil[None, :]]

        top = top_left * (1.0 - col_weight)[None, :] + top_right * col_weight[None, :]
        bottom = bottom_left * (1.0 - col_weight)[None, :] + bottom_right * col_weight[None, :]
        return top * (1.0 - row_weight)[:, None] + bottom * row_weight[:, None]

    def _save_overlay_image(self, image: np.ndarray, asset_path: Path) -> None:
        from matplotlib import image as mpimg

        mpimg.imsave(asset_path, image)

    def _slugify(self, value: str) -> str:
        return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")

    def _build_world_texture_svg(self, world_geojson: dict) -> str:
        width = 2048
        height = 1024
        paths: list[str] = []
        for feature in world_geojson.get("features", []):
            geometry = feature.get("geometry") or {}
            geometry_type = geometry.get("type")
            coordinates = geometry.get("coordinates", [])
            if geometry_type == "Polygon":
                paths.extend(self._polygon_paths(coordinates, width, height))
            elif geometry_type == "MultiPolygon":
                for polygon in coordinates:
                    paths.extend(self._polygon_paths(polygon, width, height))

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#0f2d40"/>',
        ]
        for path in paths:
            svg_parts.append(f'<path d="{path}" fill="#527a5a" stroke="#dfe9f6" stroke-width="0.8" fill-rule="evenodd"/>')
        svg_parts.append("</svg>")
        return "".join(svg_parts)

    def _polygon_paths(self, rings: list, width: int, height: int) -> list[str]:
        paths: list[str] = []
        for ring in rings:
            if len(ring) < 2:
                continue
            commands: list[str] = []
            for index, coordinate in enumerate(ring):
                if len(coordinate) < 2:
                    continue
                lon, lat = float(coordinate[0]), float(coordinate[1])
                x = (lon + 180.0) / 360.0 * width
                y = (90.0 - lat) / 180.0 * height
                commands.append(f'{"M" if index == 0 else "L"} {x:.2f} {y:.2f}')
            if commands:
                commands.append("Z")
                paths.append(" ".join(commands))
        return paths
