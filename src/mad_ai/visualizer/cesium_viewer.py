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
        return self.build_with_overlay_data(data, output_path, overlay_data=data)

    def build_with_overlay_data(
        self,
        data: pd.DataFrame,
        output_path: str | Path,
        overlay_data: pd.DataFrame,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._to_payload(data)
        world_texture_filename = self._write_world_texture_asset(output_path)
        overlay_assets = self._write_magnetic_overlay_assets(overlay_data, output_path)
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
                    "displayMode": str(getattr(row, "display_mode", "noaa_plus_anomaly")),
                    "aircraftHeadingDeg": float(getattr(row, "aircraft_heading_deg", 0.0)) if hasattr(row, "aircraft_heading_deg") else None,
                    "aircraftSpeedMps": float(getattr(row, "aircraft_speed_mps", 0.0)) if hasattr(row, "aircraft_speed_mps") else None,
                    "aircraftSpeedKnots": float(getattr(row, "aircraft_speed_knots", 0.0)) if hasattr(row, "aircraft_speed_knots") else None,
                    "vesselLat": float(getattr(row, "vessel_latitude_deg", 0.0)) if hasattr(row, "vessel_latitude_deg") else None,
                    "vesselLon": float(getattr(row, "vessel_longitude_deg", 0.0)) if hasattr(row, "vessel_longitude_deg") else None,
                    "vesselHeadingDeg": float(getattr(row, "vessel_heading_deg", 0.0)) if hasattr(row, "vessel_heading_deg") else None,
                    "vesselSpeedMps": float(getattr(row, "vessel_speed_mps", 0.0)) if hasattr(row, "vessel_speed_mps") else None,
                    "vesselSpeedKnots": float(getattr(row, "vessel_speed_knots", 0.0)) if hasattr(row, "vessel_speed_knots") else None,
                    "rangeToVesselM": float(getattr(row, "range_to_vessel_m", 0.0)) if hasattr(row, "range_to_vessel_m") else None,
                    "estimatedVesselLat": float(getattr(row, "estimated_vessel_latitude_deg", 0.0))
                    if hasattr(row, "estimated_vessel_latitude_deg") and pd.notna(getattr(row, "estimated_vessel_latitude_deg"))
                    else None,
                    "estimatedVesselLon": float(getattr(row, "estimated_vessel_longitude_deg", 0.0))
                    if hasattr(row, "estimated_vessel_longitude_deg") and pd.notna(getattr(row, "estimated_vessel_longitude_deg"))
                    else None,
                    "trackingErrorM": float(getattr(row, "tracking_error_m", 0.0))
                    if hasattr(row, "tracking_error_m") and pd.notna(getattr(row, "tracking_error_m"))
                    else None,
                    "trackingConfidence": float(getattr(row, "confidence", 0.0))
                    if hasattr(row, "confidence") and pd.notna(getattr(row, "confidence"))
                    else None,
                    "anomalyScore": float(getattr(row, self.anomaly_score_column, 0.0))
                    if hasattr(row, self.anomaly_score_column)
                    else 0.0,
                    "anomalyConfidence": float(getattr(row, self.anomaly_score_column, 0.0))
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
        secondary_label = next((label for label in component_options if label != default_label), default_label)
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
    #slider {{
      position: absolute;
      top: 0;
      bottom: 0;
      left: 50%;
      width: 3px;
      display: none;
      z-index: 9;
      background: rgba(255,255,255,0.72);
      box-shadow: 0 0 0 1px rgba(0,0,0,0.4);
      cursor: ew-resize;
    }}
    #panel {{
      position: absolute; top: 12px; left: 12px; z-index: 10; width: 368px;
      background: rgba(8, 16, 28, 0.84); color: #dfe9f6; padding: 14px 16px; border-radius: 12px;
      font-family: Segoe UI, sans-serif; backdrop-filter: blur(8px); box-shadow: 0 8px 28px rgba(0,0,0,0.28);
      max-height: calc(100vh - 24px);
      overflow-y: auto;
    }}
    #panel h1 {{ margin: 0 0 8px 0; font-size: 18px; }}
    #panel p {{ margin: 0 0 6px 0; font-size: 13px; line-height: 1.4; }}
    .legend {{ display: flex; gap: 10px; margin-top: 10px; font-size: 12px; align-items: center; flex-wrap: wrap; }}
    .swatch {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; }}
    select {{
      margin-top: 8px; width: 100%; padding: 8px 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.12);
      background: rgba(255,255,255,0.08); color: #eef5ff;
    }}
    input[type="number"], input[type="text"] {{
      margin-top: 8px; width: 100%; padding: 8px 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.12);
      background: rgba(255,255,255,0.08); color: #eef5ff; box-sizing: border-box;
    }}
    input[type="range"] {{ width: 100%; margin-top: 8px; }}
    .time-row {{ margin-top: 10px; font-size: 12px; }}
    .control-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
    .checkbox-row {{ display: flex; align-items: center; gap: 8px; margin-top: 8px; font-size: 12px; }}
    .subtle {{ color: #c6d5e6; font-size: 11px; margin-top: 4px; }}
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
  <div id="slider"></div>
  <div id="panel">
    <h1>{title}</h1>
    <p>Interactive Cesium globe adapted from the ConstellationVizFrontend approach.</p>
    <p>Magnetic values are draped across the full earth surface. Click the globe to inspect the nearest magnetic sample at the active time and altitude.</p>
    <label for="basemapSelect">Basemap</label>
    <select id="basemapSelect">
      <option value="openstreetmap" selected>OpenStreetMap</option>
      <option value="carto_positron">Carto Positron</option>
      <option value="esri_world_imagery">Esri World Imagery</option>
      <option value="local_texture">Local Fallback</option>
    </select>
    <label for="componentSelect">Displayed Component</label>
    <select id="componentSelect"></select>
    <div class="checkbox-row">
      <input id="compareToggle" type="checkbox" />
      <label for="compareToggle">Enable comparison swipe</label>
    </div>
    <label for="secondaryComponentSelect">Secondary Component</label>
    <select id="secondaryComponentSelect"></select>
    <div class="subtle">Comparison mode shows the primary overlay on the left and the secondary overlay on the right.</div>
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
      <label for="viewModeSelect">View Mode</label>
      <select id="viewModeSelect">
        <option value="noaa_only">NOAA Only</option>
        <option value="noaa_plus_anomaly" selected>NOAA + Anomaly</option>
      </select>
      <div class="subtle">NOAA/WMM baseline is always the surface overlay. The anomaly mode adds the scored flight track on top.</div>
    </div>
    <div class="time-row">
      <label for="scoreThresholdSlider">Score Threshold Filter</label>
      <input id="scoreThresholdSlider" type="range" min="0" max="1" value="0" step="0.001" />
      <div id="scoreThresholdLabel">0.000</div>
      <div class="checkbox-row">
        <input id="anomalyOnlyCheckbox" type="checkbox" />
        <label for="anomalyOnlyCheckbox">Anomaly-only filter</label>
      </div>
      <div class="checkbox-row">
        <input id="showTracksCheckbox" type="checkbox" checked />
        <label for="showTracksCheckbox">Show tracks</label>
      </div>
    </div>
    <div class="time-row">
      <label>Search Latitude / Longitude</label>
      <div class="control-grid">
        <input id="latInput" type="number" step="0.001" placeholder="Latitude" />
        <input id="lonInput" type="number" step="0.001" placeholder="Longitude" />
      </div>
      <button id="goToLatLonBtn" type="button">Go To Location</button>
      <button id="jumpHotspotBtn" type="button">Jump To Hotspot</button>
    </div>
    <div class="time-row">
      <button id="spinToggleBtn" type="button">Pause Earth Rotation</button>
    </div>
    <div class="time-row">
      <div id="clockInfo">Clock sync: loading...</div>
    </div>
    <div class="time-row">
      <button id="exportSelectedBtn" type="button">Export Selected Anomalies CSV</button>
      <button id="exportReportBtn" type="button">Export Review Bundle</button>
      <button id="exportScreenshotBtn" type="button">Export Screenshot</button>
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
      <span><span class="swatch" style="background:#ffd166"></span> Aircraft track</span>
      <span><span class="swatch" style="background:#d2a8ff"></span> Anomaly window</span>
      <span><span class="swatch" style="background:#f778ba"></span> Peak anomaly</span>
      <span><span class="swatch" style="background:#ffa657"></span> Closest approach</span>
      <span><span class="swatch" style="background:#ff7b72"></span> Vessel track</span>
      <span><span class="swatch" style="background:#7ee787"></span> Estimated magnetic track</span>
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
    let secondaryComponent = "{secondary_label}";
    let compareEnabled = false;
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
    const anomalyScores = sampleData.map((point) => Number(point.anomalyScore || 0.0)).filter((value) => Number.isFinite(value));
    const maxAnomalyScore = anomalyScores.length > 0 ? Math.max(...anomalyScores) : 1.0;
    let currentTimeIndex = 0;
    let currentAltitudeKey = altitudeKeys[0] || "0.0";
    let currentViewMode = "noaa_plus_anomaly";
    let playTimer = null;
    let primaryMagneticOverlayLayer = null;
    let secondaryMagneticOverlayLayer = null;
    let baseImageryLayer = null;
    let currentBasemap = "openstreetmap";
    let hotspotIndex = -1;
    function findNearestTimeIndex(timestamp) {{
      if (!timestamp || timeKeys.length === 0) {{
        return 0;
      }}
      const target = parseTimeKey(timestamp);
      let bestIndex = 0;
      let bestDistance = Math.abs(parseTimeKey(timeKeys[0]) - target);
      for (let index = 1; index < timeKeys.length; index += 1) {{
        const distance = Math.abs(parseTimeKey(timeKeys[index]) - target);
        if (distance < bestDistance) {{
          bestDistance = distance;
          bestIndex = index;
        }}
      }}
      return bestIndex;
    }}
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
    viewer.clock.multiplier = 1;
    viewer.clock.shouldAnimate = true;
    viewer.clock.currentTime = Cesium.JulianDate.fromDate(new Date());
    let timeSyncEnabled = true;

    function removeOverlay(layer) {{
      if (layer) {{
        viewer.scene.globe.imageryLayers.remove(layer, true);
      }}
      return null;
    }}

    function createBasemapProvider(basemap) {{
      if (basemap === "esri_world_imagery") {{
        return new Cesium.UrlTemplateImageryProvider({{
          url: "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}",
          credit: "Esri, Maxar, Earthstar Geographics, and the GIS User Community"
        }});
      }}
      if (basemap === "carto_positron") {{
        return new Cesium.UrlTemplateImageryProvider({{
          url: "https://a.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}.png",
          credit: "CARTO, OpenStreetMap contributors"
        }});
      }}
      if (basemap === "local_texture" && worldTexturePath) {{
        return new Cesium.SingleTileImageryProvider({{
          url: worldTexturePath,
          rectangle: Cesium.Rectangle.fromDegrees(-180.0, -90.0, 180.0, 90.0)
        }});
      }}
      return new Cesium.OpenStreetMapImageryProvider({{
        url: "https://a.tile.openstreetmap.org/"
      }});
    }}

    function loadBasemap(basemap) {{
      const layers = viewer.scene.globe.imageryLayers;
      if (baseImageryLayer) {{
        layers.remove(baseImageryLayer, true);
        baseImageryLayer = null;
      }}
      try {{
        baseImageryLayer = layers.addImageryProvider(createBasemapProvider(basemap), 0);
        currentBasemap = basemap;
      }} catch (err) {{
        console.warn(`Basemap load failed for ${{basemap}}`, err);
        if (basemap !== "local_texture" && worldTexturePath) {{
          try {{
            baseImageryLayer = layers.addImageryProvider(createBasemapProvider("local_texture"), 0);
            currentBasemap = "local_texture";
            const basemapSelect = document.getElementById("basemapSelect");
            if (basemapSelect) {{
              basemapSelect.value = "local_texture";
            }}
            return;
          }} catch (fallbackErr) {{
            console.warn("Local texture fallback failed", fallbackErr);
          }}
        }}
      }}
    }}

    function refreshMagneticOverlay() {{
      const layers = viewer.scene.globe.imageryLayers;
      primaryMagneticOverlayLayer = removeOverlay(primaryMagneticOverlayLayer);
      secondaryMagneticOverlayLayer = removeOverlay(secondaryMagneticOverlayLayer);
      const overlayPath = magneticOverlayAssets[currentComponent]?.[currentAltitudeKey];
      if (overlayPath) {{
        primaryMagneticOverlayLayer = layers.addImageryProvider(new Cesium.SingleTileImageryProvider({{
          url: overlayPath,
          rectangle: Cesium.Rectangle.fromDegrees(-180.0, -90.0, 180.0, 90.0)
        }}));
        primaryMagneticOverlayLayer.alpha = 0.52;
        primaryMagneticOverlayLayer.splitDirection = compareEnabled ? Cesium.SplitDirection.LEFT : Cesium.SplitDirection.NONE;
      }}
      if (compareEnabled) {{
        const secondaryPath = magneticOverlayAssets[secondaryComponent]?.[currentAltitudeKey];
        if (secondaryPath) {{
          secondaryMagneticOverlayLayer = layers.addImageryProvider(new Cesium.SingleTileImageryProvider({{
            url: secondaryPath,
            rectangle: Cesium.Rectangle.fromDegrees(-180.0, -90.0, 180.0, 90.0)
          }}));
          secondaryMagneticOverlayLayer.alpha = 0.52;
          secondaryMagneticOverlayLayer.splitDirection = Cesium.SplitDirection.RIGHT;
        }}
      }}
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

    function getSecondaryComponentValue(point) {{
      return point.components[secondaryComponent] ?? point.baseline;
    }}

    function refreshColorLegend() {{
      const activePoints = getFilteredPoints();
      const values = activePoints
        .map((point) => Number(getComponentValue(point)))
        .filter((value) => Number.isFinite(value));
      const titleEl = document.getElementById("colorLegendTitle");
      const minEl = document.getElementById("colorLegendMin");
      const midEl = document.getElementById("colorLegendMid");
      const maxEl = document.getElementById("colorLegendMax");
      titleEl.textContent = compareEnabled
        ? `${{currentComponent}} vs ${{secondaryComponent}} Scale`
        : `${{currentComponent}} Scale`;
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

    function formatClockTime(date) {{
      try {{
        return new Intl.DateTimeFormat(undefined, {{
          year: "numeric",
          month: "2-digit",
          day: "2-digit",
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
          hour12: false,
          timeZoneName: "short"
        }}).format(date);
      }} catch (error) {{
        return date.toISOString();
      }}
    }}

    function currentUtcOffsetLabel(date) {{
      const totalMinutes = -date.getTimezoneOffset();
      const sign = totalMinutes >= 0 ? "+" : "-";
      const absMinutes = Math.abs(totalMinutes);
      const hours = String(Math.floor(absMinutes / 60)).padStart(2, "0");
      const minutes = String(absMinutes % 60).padStart(2, "0");
      return `UTC${{sign}}${{hours}}:${{minutes}}`;
    }}

    function refreshClockInfo() {{
      const clockInfo = document.getElementById("clockInfo");
      if (!clockInfo) {{
        return;
      }}
      const systemNow = new Date();
      const cesiumNow = Cesium.JulianDate.toDate(viewer.clock.currentTime);
      clockInfo.innerHTML = `
        <div><b>Clock Sync:</b> ${{timeSyncEnabled ? "live system clock" : "paused"}}</div>
        <div><b>System Time:</b> ${{formatClockTime(systemNow)}}</div>
        <div><b>Cesium Time:</b> ${{formatClockTime(cesiumNow)}}</div>
        <div><b>Local Offset:</b> ${{currentUtcOffsetLabel(systemNow)}}</div>
      `;
    }}

    function getActivePoints() {{
      const activeTimePoints = groupedByTime.get(timeKeys[currentTimeIndex]) || [];
      return activeTimePoints.filter((point) => Number(point.alt).toFixed(1) === currentAltitudeKey);
    }}

    function getFilteredPoints() {{
      const threshold = Number(document.getElementById("scoreThresholdSlider").value || 0.0);
      const anomalyOnly = document.getElementById("anomalyOnlyCheckbox").checked;
      return getActivePoints().filter((point) => {{
        const matchesViewMode = currentViewMode !== "noaa_only" && (point.displayMode || "noaa_plus_anomaly") === currentViewMode;
        const passesThreshold = Number(point.anomalyScore || 0.0) >= threshold;
        const passesAnomaly = !anomalyOnly || Boolean(point.isAnomaly);
        return matchesViewMode && passesThreshold && passesAnomaly;
      }});
    }}

    function descriptionForPoint(point) {{
      const value = getComponentValue(point);
      const secondaryValue = getSecondaryComponentValue(point);
      const componentRows = Object.entries(point.components).map(([name, componentValue]) =>
        `<p><b>${{name}}:</b> ${{Number(componentValue).toFixed(2)}}</p>`
      ).join("");
      return `
          <h3>Magnetic Sample</h3>
          <p><b>Latitude:</b> ${{point.lat.toFixed(3)}}</p>
          <p><b>Longitude:</b> ${{point.lon.toFixed(3)}}</p>
          <p><b>Altitude:</b> ${{point.alt.toFixed(1)}} m</p>
          <p><b>Current Component:</b> ${{currentComponent}} = ${{Number(value).toFixed(2)}}</p>
          <p><b>Comparison Component:</b> ${{secondaryComponent}} = ${{Number(secondaryValue).toFixed(2)}}</p>
          <p><b>Baseline:</b> ${{point.baseline.toFixed(2)}} nT</p>
          <p><b>Residual:</b> ${{point.residual.toFixed(2)}} nT</p>
          <p><b>Anomaly Score:</b> ${{Number(point.anomalyScore).toFixed(4)}}</p>
          <p><b>Anomaly Confidence:</b> ${{Number(point.anomalyConfidence || 0.0).toFixed(4)}}</p>
          <p><b>Anomaly:</b> ${{point.isAnomaly}}</p>
          <p><b>Timestamp:</b> ${{point.timestamp}}</p>
          <p><b>Track ID:</b> ${{point.trackId || "Untracked"}}</p>
          <p><b>Aircraft Heading:</b> ${{point.aircraftHeadingDeg == null ? "n/a" : Number(point.aircraftHeadingDeg).toFixed(2) + " deg"}}</p>
          <p><b>Aircraft Speed:</b> ${{point.aircraftSpeedMps == null ? "n/a" : Number(point.aircraftSpeedMps).toFixed(2) + " m/s (" + Number(point.aircraftSpeedKnots).toFixed(2) + " kn)"}}</p>
          <p><b>Vessel Position:</b> ${{point.vesselLat == null ? "n/a" : Number(point.vesselLat).toFixed(3) + ", " + Number(point.vesselLon).toFixed(3)}}</p>
          <p><b>Estimated Vessel Position:</b> ${{point.estimatedVesselLat == null ? "n/a" : Number(point.estimatedVesselLat).toFixed(3) + ", " + Number(point.estimatedVesselLon).toFixed(3)}}</p>
          <p><b>Vessel Heading:</b> ${{point.vesselHeadingDeg == null ? "n/a" : Number(point.vesselHeadingDeg).toFixed(2) + " deg"}}</p>
          <p><b>Vessel Speed:</b> ${{point.vesselSpeedMps == null ? "n/a" : Number(point.vesselSpeedMps).toFixed(2) + " m/s (" + Number(point.vesselSpeedKnots).toFixed(2) + " kn)"}}</p>
          <p><b>Range To Vessel:</b> ${{point.rangeToVesselM == null ? "n/a" : (Number(point.rangeToVesselM) / 1000.0).toFixed(3) + " km"}}</p>
          <p><b>Tracking Error:</b> ${{point.trackingErrorM == null ? "n/a" : (Number(point.trackingErrorM) / 1000.0).toFixed(3) + " km"}}</p>
          <p><b>Tracking Confidence:</b> ${{point.trackingConfidence == null ? "n/a" : Number(point.trackingConfidence).toFixed(4)}}</p>
          ${{componentRows}}
      `;
    }}
    let trackEntities = [];
    let vesselTrackEntities = [];
    let estimatedVesselTrackEntities = [];
    let anomalyWindowEntities = [];
    let anomalyPeakEntities = [];
    let closestApproachEntities = [];
    let trackLabelEntities = [];
    let currentAircraftEntities = [];
    let currentVesselEntities = [];
    let currentEstimatedVesselEntities = [];
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
      vesselTrackEntities.forEach((entity) => viewer.entities.remove(entity));
      vesselTrackEntities = [];
      estimatedVesselTrackEntities.forEach((entity) => viewer.entities.remove(entity));
      estimatedVesselTrackEntities = [];
      anomalyWindowEntities.forEach((entity) => viewer.entities.remove(entity));
      anomalyWindowEntities = [];
      anomalyPeakEntities.forEach((entity) => viewer.entities.remove(entity));
      anomalyPeakEntities = [];
      closestApproachEntities.forEach((entity) => viewer.entities.remove(entity));
      closestApproachEntities = [];
      trackLabelEntities.forEach((entity) => viewer.entities.remove(entity));
      trackLabelEntities = [];
      currentAircraftEntities.forEach((entity) => viewer.entities.remove(entity));
      currentAircraftEntities = [];
      currentVesselEntities.forEach((entity) => viewer.entities.remove(entity));
      currentVesselEntities = [];
      currentEstimatedVesselEntities.forEach((entity) => viewer.entities.remove(entity));
      currentEstimatedVesselEntities = [];
      if (selectionEntity) {{
        viewer.entities.remove(selectionEntity);
        selectionEntity = null;
      }}
    }}

    function parseTimeValue(point) {{
      return parseTimeKey(point.timestamp || "");
    }}

    function colorForAnomalyScore(score) {{
      const normalized = Math.min(Math.max(Number(score || 0.0) / Math.max(maxAnomalyScore, 0.001), 0.0), 1.0);
      return Cesium.Color.fromHsl(0.66 * (1.0 - normalized), 0.88, 0.56, 0.96);
    }}

    function normalizeAnomalyConfidence(score) {{
      return Math.min(Math.max(Number(score || 0.0) / Math.max(maxAnomalyScore, 0.001), 0.0), 1.0);
    }}

    function sequenceGroups(points) {{
      if (points.length === 0) {{
        return [];
      }}
      const ordered = points.slice().sort((left, right) => parseTimeValue(left) - parseTimeValue(right));
      const groups = [];
      let currentGroup = [ordered[0]];
      for (let index = 1; index < ordered.length; index += 1) {{
        const previous = ordered[index - 1];
        const current = ordered[index];
        const timeGapMs = Math.abs(parseTimeValue(current) - parseTimeValue(previous));
        const altitudeChanged = Number(current.alt).toFixed(1) !== Number(previous.alt).toFixed(1);
        if (timeGapMs > 120000 || altitudeChanged) {{
          groups.push(currentGroup);
          currentGroup = [current];
          continue;
        }}
        currentGroup.push(current);
      }}
      groups.push(currentGroup);
      return groups;
    }}

    function buildCurrentMarkers() {{
      const activePoints = getFilteredPoints();
      activePoints.forEach((point, index) => {{
        currentAircraftEntities.push(
          viewer.entities.add({{
            id: `current_aircraft_${{index}}`,
            position: Cesium.Cartesian3.fromDegrees(point.lon, point.lat, point.alt),
            point: {{
              pixelSize: 12,
              color: colorForAnomalyScore(point.anomalyScore),
              outlineColor: Cesium.Color.WHITE,
              outlineWidth: 2
            }},
            label: {{
              text: `Aircraft Now\\nScore ${{Number(point.anomalyScore || 0.0).toFixed(3)}}`,
              font: "13px Segoe UI",
              fillColor: Cesium.Color.WHITE,
              outlineColor: Cesium.Color.BLACK,
              outlineWidth: 2,
              style: Cesium.LabelStyle.FILL_AND_OUTLINE,
              pixelOffset: new Cesium.Cartesian2(0, -28),
              showBackground: true,
              backgroundColor: Cesium.Color.fromCssColorString("rgba(8,16,28,0.78)")
            }},
            description: descriptionForPoint(point)
          }})
        );
        if (point.vesselLat != null && point.vesselLon != null) {{
          currentVesselEntities.push(
            viewer.entities.add({{
              id: `current_vessel_${{index}}`,
              position: Cesium.Cartesian3.fromDegrees(point.vesselLon, point.vesselLat, 0.0),
              point: {{
                pixelSize: 11,
                color: Cesium.Color.fromCssColorString("#ff7b72"),
                outlineColor: Cesium.Color.WHITE,
                outlineWidth: 2
              }},
              label: {{
                text: `Vessel Now\\nRange ${{point.rangeToVesselM == null ? "n/a" : (Number(point.rangeToVesselM) / 1000.0).toFixed(2) + " km"}}`,
                font: "13px Segoe UI",
                fillColor: Cesium.Color.WHITE,
                outlineColor: Cesium.Color.BLACK,
                outlineWidth: 2,
                style: Cesium.LabelStyle.FILL_AND_OUTLINE,
                pixelOffset: new Cesium.Cartesian2(0, -28),
                showBackground: true,
                backgroundColor: Cesium.Color.fromCssColorString("rgba(8,16,28,0.78)")
              }}
            }})
          );
        }}
        if (point.estimatedVesselLat != null && point.estimatedVesselLon != null) {{
          currentEstimatedVesselEntities.push(
            viewer.entities.add({{
              id: `current_estimated_vessel_${{index}}`,
              position: Cesium.Cartesian3.fromDegrees(point.estimatedVesselLon, point.estimatedVesselLat, 0.0),
              point: {{
                pixelSize: 10,
                color: Cesium.Color.fromCssColorString("#7ee787"),
                outlineColor: Cesium.Color.WHITE,
                outlineWidth: 2
              }},
              label: {{
                text: `Estimated Vessel Now\\nError ${{point.trackingErrorM == null ? "n/a" : (Number(point.trackingErrorM) / 1000.0).toFixed(2) + " km"}}`,
                font: "13px Segoe UI",
                fillColor: Cesium.Color.WHITE,
                outlineColor: Cesium.Color.BLACK,
                outlineWidth: 2,
                style: Cesium.LabelStyle.FILL_AND_OUTLINE,
                pixelOffset: new Cesium.Cartesian2(0, -28),
                showBackground: true,
                backgroundColor: Cesium.Color.fromCssColorString("rgba(8,16,28,0.78)")
              }}
            }})
          );
        }}
      }});
    }}

    function buildTracks() {{
      if (!document.getElementById("showTracksCheckbox").checked) {{
        return;
      }}
      const threshold = Number(document.getElementById("scoreThresholdSlider").value || 0.0);
      const anomalyOnly = document.getElementById("anomalyOnlyCheckbox").checked;
      const historicalPoints = getHistoricalTrackPoints();
      const filteredHistoricalPoints = historicalPoints.filter((point) => {{
        const passesThreshold = Number(point.anomalyScore || 0.0) >= threshold;
        const passesAnomaly = !anomalyOnly || Boolean(point.isAnomaly);
        return passesThreshold && passesAnomaly;
      }});
      const groupedByTrack = new Map();
      filteredHistoricalPoints.forEach((point) => {{
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
            name: `Aircraft Track ${{trackId}}`,
            polyline: {{
              positions,
              width: 4,
              material: Cesium.Color.fromCssColorString("#ffd166"),
              clampToGround: false
            }},
            description: `<h3>Aircraft Track ${{trackId}}</h3><p><b>Visible History Samples:</b> ${{orderedPoints.length}}</p><p><b>Latest Speed:</b> ${{orderedPoints[orderedPoints.length - 1].aircraftSpeedMps == null ? "n/a" : Number(orderedPoints[orderedPoints.length - 1].aircraftSpeedMps).toFixed(2) + " m/s (" + Number(orderedPoints[orderedPoints.length - 1].aircraftSpeedKnots).toFixed(2) + " kn)"}}</p>`
          }});
        }});

      const groupedVesselTrack = new Map();
      historicalPoints.forEach((point) => {{
        if (point.vesselLat == null || point.vesselLon == null || !point.trackId) {{
          return;
        }}
        const vesselTrackId = `${{point.trackId}}_vessel`;
        if (!groupedVesselTrack.has(vesselTrackId)) groupedVesselTrack.set(vesselTrackId, []);
        groupedVesselTrack.get(vesselTrackId).push(point);
      }});

      vesselTrackEntities = Array.from(groupedVesselTrack.entries())
        .filter(([, points]) => points.length >= 2)
        .map(([trackId, points], index) => {{
          const orderedPoints = points.slice().sort((left, right) => parseTimeValue(left) - parseTimeValue(right));
          const positions = orderedPoints.map((point) => Cesium.Cartesian3.fromDegrees(point.vesselLon, point.vesselLat, 0.0));
          return viewer.entities.add({{
            id: `vessel_track_${{index}}_${{trackId}}`,
            name: `Vessel Track ${{trackId}}`,
            polyline: {{
              positions,
              width: 4,
              material: Cesium.Color.fromCssColorString("#ff7b72"),
              clampToGround: false
            }},
            description: `<h3>Vessel Track ${{trackId}}</h3><p><b>Visible History Samples:</b> ${{orderedPoints.length}}</p><p><b>Latest Heading:</b> ${{orderedPoints[orderedPoints.length - 1].vesselHeadingDeg == null ? "n/a" : Number(orderedPoints[orderedPoints.length - 1].vesselHeadingDeg).toFixed(2) + " deg"}}</p><p><b>Latest Speed:</b> ${{orderedPoints[orderedPoints.length - 1].vesselSpeedMps == null ? "n/a" : Number(orderedPoints[orderedPoints.length - 1].vesselSpeedMps).toFixed(2) + " m/s (" + Number(orderedPoints[orderedPoints.length - 1].vesselSpeedKnots).toFixed(2) + " kn)"}}</p>`
          }});
        }});

      const groupedEstimatedVesselTrack = new Map();
      historicalPoints.forEach((point) => {{
        if (point.estimatedVesselLat == null || point.estimatedVesselLon == null || !point.trackId) {{
          return;
        }}
        const estimatedTrackId = `${{point.trackId}}_estimated`;
        if (!groupedEstimatedVesselTrack.has(estimatedTrackId)) groupedEstimatedVesselTrack.set(estimatedTrackId, []);
        groupedEstimatedVesselTrack.get(estimatedTrackId).push(point);
      }});

      estimatedVesselTrackEntities = Array.from(groupedEstimatedVesselTrack.entries())
        .filter(([, points]) => points.length >= 2)
        .map(([trackId, points], index) => {{
          const orderedPoints = points.slice().sort((left, right) => parseTimeValue(left) - parseTimeValue(right));
          const positions = orderedPoints.map((point) => Cesium.Cartesian3.fromDegrees(point.estimatedVesselLon, point.estimatedVesselLat, 0.0));
          return viewer.entities.add({{
            id: `estimated_vessel_track_${{index}}_${{trackId}}`,
            name: `Estimated Magnetic Track ${{trackId}}`,
            polyline: {{
              positions,
              width: 3,
              material: Cesium.Color.fromCssColorString("#7ee787"),
              clampToGround: false
            }},
            description: `<h3>Estimated Magnetic Track ${{trackId}}</h3><p><b>Visible History Samples:</b> ${{orderedPoints.length}}</p><p><b>Latest Tracking Error:</b> ${{orderedPoints[orderedPoints.length - 1].trackingErrorM == null ? "n/a" : (Number(orderedPoints[orderedPoints.length - 1].trackingErrorM) / 1000.0).toFixed(2) + " km"}}</p><p><b>Latest Confidence:</b> ${{orderedPoints[orderedPoints.length - 1].trackingConfidence == null ? "n/a" : Number(orderedPoints[orderedPoints.length - 1].trackingConfidence).toFixed(4)}}</p>`
          }});
        }});

      trackLabelEntities = [];
      Array.from(groupedByTrack.entries()).forEach(([trackId, points], index) => {{
        const orderedPoints = points.slice().sort((left, right) => parseTimeValue(left) - parseTimeValue(right));
        const latest = orderedPoints[orderedPoints.length - 1];
        if (!latest) {{
          return;
        }}
        trackLabelEntities.push(
          viewer.entities.add({{
            id: `aircraft_label_${{index}}_${{trackId}}`,
            position: Cesium.Cartesian3.fromDegrees(latest.lon, latest.lat, latest.alt + 150.0),
            label: {{
              text: `Aircraft: ${{trackId}}`,
              font: "14px Segoe UI",
              fillColor: Cesium.Color.fromCssColorString("#ffd166"),
              outlineColor: Cesium.Color.BLACK,
              outlineWidth: 2,
              style: Cesium.LabelStyle.FILL_AND_OUTLINE,
              pixelOffset: new Cesium.Cartesian2(0, -18),
              showBackground: true,
              backgroundColor: Cesium.Color.fromCssColorString("rgba(8,16,28,0.78)")
            }}
          }})
        );
      }});
      Array.from(groupedVesselTrack.entries()).forEach(([trackId, points], index) => {{
        const orderedPoints = points.slice().sort((left, right) => parseTimeValue(left) - parseTimeValue(right));
        const latest = orderedPoints[orderedPoints.length - 1];
        if (!latest || latest.vesselLat == null || latest.vesselLon == null) {{
          return;
        }}
        trackLabelEntities.push(
          viewer.entities.add({{
            id: `vessel_label_${{index}}_${{trackId}}`,
            position: Cesium.Cartesian3.fromDegrees(latest.vesselLon, latest.vesselLat, 50.0),
            label: {{
              text: `Vessel: ${{trackId.replace("_vessel", "")}}`,
              font: "14px Segoe UI",
              fillColor: Cesium.Color.fromCssColorString("#ff7b72"),
              outlineColor: Cesium.Color.BLACK,
              outlineWidth: 2,
              style: Cesium.LabelStyle.FILL_AND_OUTLINE,
              pixelOffset: new Cesium.Cartesian2(0, -18),
              showBackground: true,
              backgroundColor: Cesium.Color.fromCssColorString("rgba(8,16,28,0.78)")
            }}
          }})
        );
      }});
      Array.from(groupedEstimatedVesselTrack.entries()).forEach(([trackId, points], index) => {{
        const orderedPoints = points.slice().sort((left, right) => parseTimeValue(left) - parseTimeValue(right));
        const latest = orderedPoints[orderedPoints.length - 1];
        if (!latest || latest.estimatedVesselLat == null || latest.estimatedVesselLon == null) {{
          return;
        }}
        trackLabelEntities.push(
          viewer.entities.add({{
            id: `estimated_vessel_label_${{index}}_${{trackId}}`,
            position: Cesium.Cartesian3.fromDegrees(latest.estimatedVesselLon, latest.estimatedVesselLat, 50.0),
            label: {{
              text: `Estimated: ${{trackId.replace("_estimated", "")}}`,
              font: "14px Segoe UI",
              fillColor: Cesium.Color.fromCssColorString("#7ee787"),
              outlineColor: Cesium.Color.BLACK,
              outlineWidth: 2,
              style: Cesium.LabelStyle.FILL_AND_OUTLINE,
              pixelOffset: new Cesium.Cartesian2(0, -18),
              showBackground: true,
              backgroundColor: Cesium.Color.fromCssColorString("rgba(8,16,28,0.78)")
            }}
          }})
        );
      }});
    }}

    function buildAnomalyOutcomeOverlays() {{
      if (currentViewMode === "noaa_only") {{
        return;
      }}
      const historicalPoints = getHistoricalTrackPoints();
      if (historicalPoints.length === 0) {{
        return;
      }}
      const threshold = Number(document.getElementById("scoreThresholdSlider").value || 0.0);
      const anomalyQualifiedPoints = historicalPoints.filter((point) => {{
        const passesThreshold = Number(point.anomalyScore || 0.0) >= threshold;
        const passesFlag = Boolean(point.isAnomaly) || Number(point.anomalyScore || 0.0) >= threshold;
        return passesThreshold && passesFlag;
      }});
      if (anomalyQualifiedPoints.length === 0) {{
        return;
      }}

      const groupedByTrack = new Map();
      anomalyQualifiedPoints.forEach((point) => {{
        if (!point.trackId) {{
          return;
        }}
        if (!groupedByTrack.has(point.trackId)) {{
          groupedByTrack.set(point.trackId, []);
        }}
        groupedByTrack.get(point.trackId).push(point);
      }});

      Array.from(groupedByTrack.entries()).forEach(([trackId, points], index) => {{
        sequenceGroups(points)
          .filter((group) => group.length >= 2)
          .forEach((group, groupIndex) => {{
            const positions = group.map((point) => Cesium.Cartesian3.fromDegrees(point.lon, point.lat, point.alt + 40.0));
            const maxScore = Math.max(...group.map((point) => Number(point.anomalyScore || 0.0)));
            anomalyWindowEntities.push(
              viewer.entities.add({{
                id: `anomaly_window_${{index}}_${{groupIndex}}_${{trackId}}`,
                name: `Anomaly Window ${{trackId}}`,
                polyline: {{
                  positions,
                  width: 9,
                  material: new Cesium.PolylineGlowMaterialProperty({{
                    glowPower: 0.28,
                    taperPower: 0.75,
                    color: Cesium.Color.fromCssColorString("#d2a8ff").withAlpha(0.72)
                  }}),
                  clampToGround: false
                }},
                description: `<h3>Anomaly Window ${{trackId}}</h3><p><b>Samples:</b> ${{group.length}}</p><p><b>Start:</b> ${{group[0].timestamp}}</p><p><b>End:</b> ${{group[group.length - 1].timestamp}}</p><p><b>Peak Score:</b> ${{Number(maxScore).toFixed(4)}}</p>`
              }})
            );
          }});
      }});

      anomalyQualifiedPoints
        .slice()
        .sort((left, right) => Number(right.anomalyScore || 0.0) - Number(left.anomalyScore || 0.0))
        .slice(0, 5)
        .forEach((point, index) => {{
          anomalyPeakEntities.push(
            viewer.entities.add({{
              id: `anomaly_peak_${{index}}`,
              position: Cesium.Cartesian3.fromDegrees(point.lon, point.lat, point.alt + 120.0),
              point: {{
                pixelSize: 16,
                color: Cesium.Color.fromCssColorString("#f778ba"),
                outlineColor: Cesium.Color.WHITE,
                outlineWidth: 2
              }},
              label: {{
                text: `Peak ${{index + 1}}\\nConf ${{normalizeAnomalyConfidence(point.anomalyScore).toFixed(2)}}`,
                font: "13px Segoe UI",
                fillColor: Cesium.Color.WHITE,
                outlineColor: Cesium.Color.BLACK,
                outlineWidth: 2,
                style: Cesium.LabelStyle.FILL_AND_OUTLINE,
                pixelOffset: new Cesium.Cartesian2(0, -30),
                showBackground: true,
                backgroundColor: Cesium.Color.fromCssColorString("rgba(8,16,28,0.82)")
              }},
              description: `<h3>Peak Anomaly Candidate</h3><p><b>Timestamp:</b> ${{point.timestamp}}</p><p><b>Anomaly Score:</b> ${{Number(point.anomalyScore).toFixed(4)}}</p><p><b>Anomaly Confidence:</b> ${{normalizeAnomalyConfidence(point.anomalyScore).toFixed(4)}}</p><p><b>Residual:</b> ${{Number(point.residual).toFixed(2)}} nT</p><p><b>Aircraft Position:</b> ${{Number(point.lat).toFixed(3)}}, ${{Number(point.lon).toFixed(3)}} @ ${{Number(point.alt).toFixed(1)}} m</p>`
            }})
          );
        }});

      historicalPoints
        .filter((point) => point.rangeToVesselM != null && Number.isFinite(Number(point.rangeToVesselM)))
        .slice()
        .sort((left, right) => Number(left.rangeToVesselM) - Number(right.rangeToVesselM))
        .slice(0, 3)
        .forEach((point, index) => {{
          closestApproachEntities.push(
            viewer.entities.add({{
              id: `closest_approach_${{index}}`,
              position: Cesium.Cartesian3.fromDegrees(point.lon, point.lat, point.alt + 80.0),
              point: {{
                pixelSize: 14,
                color: Cesium.Color.fromCssColorString("#ffa657"),
                outlineColor: Cesium.Color.WHITE,
                outlineWidth: 2
              }},
              label: {{
                text: `Closest ${{index + 1}}\\n${{(Number(point.rangeToVesselM) / 1000.0).toFixed(2)}} km`,
                font: "13px Segoe UI",
                fillColor: Cesium.Color.WHITE,
                outlineColor: Cesium.Color.BLACK,
                outlineWidth: 2,
                style: Cesium.LabelStyle.FILL_AND_OUTLINE,
                pixelOffset: new Cesium.Cartesian2(0, -28),
                showBackground: true,
                backgroundColor: Cesium.Color.fromCssColorString("rgba(8,16,28,0.82)")
              }},
              description: `<h3>Closest-Approach Candidate</h3><p><b>Timestamp:</b> ${{point.timestamp}}</p><p><b>Range To Vessel:</b> ${{(Number(point.rangeToVesselM) / 1000.0).toFixed(3)}} km</p><p><b>Anomaly Score:</b> ${{Number(point.anomalyScore || 0.0).toFixed(4)}}</p><p><b>Residual:</b> ${{Number(point.residual).toFixed(2)}} nT</p><p><b>Aircraft Position:</b> ${{Number(point.lat).toFixed(3)}}, ${{Number(point.lon).toFixed(3)}} @ ${{Number(point.alt).toFixed(1)}} m</p>`
            }})
          );
        }});
    }}

    function refreshScene() {{
      clearDynamicEntities();
      buildTracks();
      buildAnomalyOutcomeOverlays();
      buildCurrentMarkers();
      document.getElementById("timeLabel").textContent = timeKeys[currentTimeIndex];
      refreshColorLegend();
      refreshClockInfo();
      notifyTimeChange();
    }}

    function setCurrentTimeIndex(index) {{
      currentTimeIndex = Math.min(Math.max(Number(index) || 0, 0), Math.max(0, timeKeys.length - 1));
      if (typeof timeSlider !== "undefined") {{
        timeSlider.value = String(currentTimeIndex);
      }}
      refreshScene();
    }}

    function notifyTimeChange() {{
      const payload = {{
        type: "mad-ai-time-change",
        source: "viewer",
        timeIndex: currentTimeIndex,
        timestamp: timeKeys[currentTimeIndex] || "",
        altitudeM: Number(currentAltitudeKey || 0.0),
        viewMode: currentViewMode
      }};
      if (window.parent && window.parent !== window) {{
        window.parent.postMessage(payload, "*");
      }}
      if (window.opener && window.opener !== window) {{
        window.opener.postMessage(payload, "*");
      }}
    }}

    function describePointForPanel(point, clickedLat, clickedLon) {{
      const currentValue = getComponentValue(point);
      const secondaryValue = getSecondaryComponentValue(point);
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
        <div><b>Comparison Component:</b> ${{secondaryComponent}} = ${{Number(secondaryValue).toFixed(2)}}</div>
        <div><b>Track ID:</b> ${{point.trackId || "Untracked"}}</div>
        <div><b>Baseline:</b> ${{point.baseline.toFixed(2)}} nT</div>
        <div><b>Residual:</b> ${{point.residual.toFixed(2)}} nT</div>
        <div><b>Anomaly Score:</b> ${{Number(point.anomalyScore).toFixed(4)}}</div>
        <div><b>Anomaly Confidence:</b> ${{normalizeAnomalyConfidence(point.anomalyScore).toFixed(4)}}</div>
        <div><b>Anomaly:</b> ${{point.isAnomaly}}</div>
        <div><b>Aircraft Heading:</b> ${{point.aircraftHeadingDeg == null ? "n/a" : Number(point.aircraftHeadingDeg).toFixed(2) + " deg"}}</div>
        <div><b>Aircraft Speed:</b> ${{point.aircraftSpeedMps == null ? "n/a" : Number(point.aircraftSpeedMps).toFixed(2) + " m/s (" + Number(point.aircraftSpeedKnots).toFixed(2) + " kn)"}}</div>
        <div><b>Vessel Position:</b> ${{point.vesselLat == null ? "n/a" : Number(point.vesselLat).toFixed(3) + ", " + Number(point.vesselLon).toFixed(3)}}</div>
        <div><b>Estimated Vessel Position:</b> ${{point.estimatedVesselLat == null ? "n/a" : Number(point.estimatedVesselLat).toFixed(3) + ", " + Number(point.estimatedVesselLon).toFixed(3)}}</div>
        <div><b>Vessel Heading:</b> ${{point.vesselHeadingDeg == null ? "n/a" : Number(point.vesselHeadingDeg).toFixed(2) + " deg"}}</div>
        <div><b>Vessel Speed:</b> ${{point.vesselSpeedMps == null ? "n/a" : Number(point.vesselSpeedMps).toFixed(2) + " m/s (" + Number(point.vesselSpeedKnots).toFixed(2) + " kn)"}}</div>
        <div><b>Range To Vessel:</b> ${{point.rangeToVesselM == null ? "n/a" : (Number(point.rangeToVesselM) / 1000.0).toFixed(3) + " km"}}</div>
        <div><b>Tracking Error:</b> ${{point.trackingErrorM == null ? "n/a" : (Number(point.trackingErrorM) / 1000.0).toFixed(3) + " km"}}</div>
        <div><b>Tracking Confidence:</b> ${{point.trackingConfidence == null ? "n/a" : Number(point.trackingConfidence).toFixed(4)}}</div>
        <div><b>Threshold Filter:</b> ${{Number(document.getElementById("scoreThresholdSlider").value || 0.0).toFixed(3)}}</div>
        <div><b>Decision:</b> ${{Number(point.anomalyScore || 0.0) >= Number(document.getElementById("scoreThresholdSlider").value || 0.0) ? "above filter" : "below filter"}}</div>
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
      const activePoints = getFilteredPoints();
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
        selectionInfo.textContent = currentViewMode === "noaa_only"
          ? "NOAA-only mode is active. Switch to NOAA + Anomaly to inspect the scored flight samples."
          : "No magnetic sample is available for the current time and altitude.";
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

    function flyToPoint(point) {{
      viewer.camera.flyTo({{
        destination: Cesium.Cartesian3.fromDegrees(point.lon, point.lat, Math.max(point.alt + 400000.0, 400000.0)),
        duration: 1.4
      }});
      updateSurfaceSelection(Cesium.Cartographic.fromDegrees(point.lon, point.lat, point.alt));
    }}

    function exportCsv(filename, rows) {{
      const headers = Object.keys(rows[0] || {{ sample: "" }});
      const content = [headers.join(",")]
        .concat(rows.map((row) => headers.map((header) => JSON.stringify(row[header] ?? "")).join(",")))
        .join("\\n");
      const blob = new Blob([content], {{ type: "text/csv;charset=utf-8;" }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      link.click();
      URL.revokeObjectURL(url);
    }}

    function exportJson(filename, payload) {{
      const blob = new Blob([JSON.stringify(payload, null, 2)], {{ type: "application/json;charset=utf-8;" }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      link.click();
      URL.revokeObjectURL(url);
    }}

    const componentSelect = document.getElementById("componentSelect");
    const secondaryComponentSelect = document.getElementById("secondaryComponentSelect");
    const basemapSelect = document.getElementById("basemapSelect");
    basemapSelect.addEventListener("change", (event) => {{
      loadBasemap(event.target.value);
    }});
    componentOptions.forEach((label) => {{
      const option = document.createElement("option");
      option.value = label;
      option.textContent = label;
      option.selected = label === currentComponent;
      componentSelect.appendChild(option);
      const secondaryOption = document.createElement("option");
      secondaryOption.value = label;
      secondaryOption.textContent = label;
      secondaryOption.selected = label === secondaryComponent;
      secondaryComponentSelect.appendChild(secondaryOption);
    }});
    componentSelect.addEventListener("change", (event) => {{
      currentComponent = event.target.value;
      if (secondaryComponent === currentComponent && componentOptions.length > 1) {{
        secondaryComponent = componentOptions.find((label) => label !== currentComponent) || currentComponent;
        secondaryComponentSelect.value = secondaryComponent;
      }}
      refreshMagneticOverlay();
      refreshScene();
    }});
    secondaryComponentSelect.addEventListener("change", (event) => {{
      secondaryComponent = event.target.value;
      refreshMagneticOverlay();
      refreshScene();
    }});

    const compareToggle = document.getElementById("compareToggle");
    const slider = document.getElementById("slider");
    compareToggle.addEventListener("change", (event) => {{
      compareEnabled = event.target.checked;
      slider.style.display = compareEnabled ? "block" : "none";
      refreshMagneticOverlay();
      refreshScene();
    }});
    const splitHandler = new Cesium.ScreenSpaceEventHandler(slider);
    splitHandler.setInputAction((movement) => {{
      const rect = viewer.scene.canvas.getBoundingClientRect();
      const splitPosition = Math.min(Math.max((movement.endPosition.x - rect.left) / rect.width, 0.05), 0.95);
      viewer.scene.splitPosition = splitPosition;
      slider.style.left = `${{splitPosition * 100}}%`;
    }}, Cesium.ScreenSpaceEventType.MOUSE_MOVE);

    const timeSlider = document.getElementById("timeSlider");
    timeSlider.max = String(Math.max(0, timeKeys.length - 1));
    timeSlider.value = "0";
    timeSlider.addEventListener("input", (event) => {{
      setCurrentTimeIndex(Number(event.target.value));
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

    const scoreThresholdSlider = document.getElementById("scoreThresholdSlider");
    const scoreThresholdLabel = document.getElementById("scoreThresholdLabel");
    scoreThresholdSlider.max = String(Math.max(maxAnomalyScore, 0.001));
    scoreThresholdSlider.value = "0";
    scoreThresholdSlider.addEventListener("input", (event) => {{
      scoreThresholdLabel.textContent = Number(event.target.value).toFixed(3);
      hotspotIndex = -1;
      refreshScene();
    }});
    document.getElementById("anomalyOnlyCheckbox").addEventListener("change", () => {{
      hotspotIndex = -1;
      refreshScene();
    }});
    document.getElementById("showTracksCheckbox").addEventListener("change", () => {{
      refreshScene();
    }});
    document.getElementById("viewModeSelect").addEventListener("change", (event) => {{
      currentViewMode = event.target.value;
      hotspotIndex = -1;
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
        setCurrentTimeIndex((currentTimeIndex + 1) % timeKeys.length);
      }}, 900);
      playPauseBtn.textContent = "Pause";
    }});

    const spinToggleBtn = document.getElementById("spinToggleBtn");
    spinToggleBtn.addEventListener("click", () => {{
      timeSyncEnabled = !timeSyncEnabled;
      if (timeSyncEnabled) {{
        viewer.clock.shouldAnimate = true;
      }} else {{
        viewer.clock.shouldAnimate = false;
      }}
      spinToggleBtn.textContent = timeSyncEnabled ? "Pause Earth Rotation" : "Resume Earth Rotation";
      refreshClockInfo();
    }});

    document.getElementById("goToLatLonBtn").addEventListener("click", () => {{
      const lat = Number(document.getElementById("latInput").value);
      const lon = Number(document.getElementById("lonInput").value);
      if (!Number.isFinite(lat) || !Number.isFinite(lon)) {{
        return;
      }}
      viewer.camera.flyTo({{
        destination: Cesium.Cartesian3.fromDegrees(lon, lat, 450000.0),
        duration: 1.4
      }});
      updateSurfaceSelection(Cesium.Cartographic.fromDegrees(lon, lat, 0.0));
    }});

    document.getElementById("jumpHotspotBtn").addEventListener("click", () => {{
      const hotspots = getFilteredPoints().slice().sort((left, right) => Number(right.anomalyScore || 0.0) - Number(left.anomalyScore || 0.0));
      if (hotspots.length === 0) {{
        return;
      }}
      hotspotIndex = (hotspotIndex + 1) % hotspots.length;
      flyToPoint(hotspots[hotspotIndex]);
    }});

    document.getElementById("exportSelectedBtn").addEventListener("click", () => {{
      const rows = getFilteredPoints()
        .filter((point) => Boolean(point.isAnomaly))
        .map((point) => ({{
          latitude_deg: point.lat,
          longitude_deg: point.lon,
          altitude_m: point.alt,
          timestamp: point.timestamp,
          anomaly_score: point.anomalyScore,
          primary_component: currentComponent,
          primary_value: getComponentValue(point),
          comparison_component: secondaryComponent,
          comparison_value: getSecondaryComponentValue(point),
          baseline_total_nt: point.baseline,
          residual_total_nt: point.residual
        }}));
      if (rows.length > 0) {{
        exportCsv("mad_ai_selected_anomalies.csv", rows);
      }}
    }});

    document.getElementById("exportReportBtn").addEventListener("click", () => {{
      const filteredPoints = getFilteredPoints();
      exportJson("mad_ai_review_bundle.json", {{
        title: "{title}",
        primary_component: currentComponent,
        secondary_component: secondaryComponent,
        compare_enabled: compareEnabled,
        time_slice: timeKeys[currentTimeIndex],
        altitude_layer_m: Number(currentAltitudeKey),
        anomaly_only: document.getElementById("anomalyOnlyCheckbox").checked,
        show_tracks: document.getElementById("showTracksCheckbox").checked,
        score_threshold: Number(scoreThresholdSlider.value || 0.0),
        filtered_point_count: filteredPoints.length,
        top_hotspots: filteredPoints
          .slice()
          .sort((left, right) => Number(right.anomalyScore || 0.0) - Number(left.anomalyScore || 0.0))
          .slice(0, 10)
          .map((point) => ({{
            lat: point.lat,
            lon: point.lon,
            alt: point.alt,
            anomaly_score: point.anomalyScore,
            timestamp: point.timestamp
          }})),
        selection_summary: document.getElementById("selectionInfo").innerText
      }});
    }});

    document.getElementById("exportScreenshotBtn").addEventListener("click", () => {{
      viewer.render();
      const link = document.createElement("a");
      link.href = viewer.scene.canvas.toDataURL("image/png");
      link.download = "mad_ai_globe_screenshot.png";
      link.click();
    }});

    window.addEventListener("message", (event) => {{
      const message = event.data || {{}};
      if (message.type === "mad-ai-set-time-index") {{
        setCurrentTimeIndex(Number(message.timeIndex || 0));
      }}
      if (message.type === "mad-ai-set-timestamp") {{
        setCurrentTimeIndex(findNearestTimeIndex(String(message.timestamp || "")));
      }}
    }});

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

    loadBasemap(currentBasemap);
    refreshMagneticOverlay();
    refreshScene();
    window.setInterval(refreshClockInfo, 1000);

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
