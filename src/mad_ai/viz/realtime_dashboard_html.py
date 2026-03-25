from __future__ import annotations

import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from mad_ai.core.base import BaseVisualizer


class BahamasRealtimeDashboardHTMLBuilder(BaseVisualizer):
    def __init__(
        self,
        title: str = "Bahamas Magnetic Anomaly Realtime Review Dashboard",
        globe_relative_path: str = "cesium_bahamas_noaa_combined.html",
        max_points: int = 3600,
    ) -> None:
        self.title = title
        self.globe_relative_path = globe_relative_path
        self.max_points = max_points

    def render(self, data: pd.DataFrame, output_path: str | Path, summary: dict[str, object] | None = None) -> Path:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("BahamasRealtimeDashboardHTMLBuilder requires a DataFrame input.")

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        frame = data.sort_values("timestamp").reset_index(drop=True).copy()
        frame = self._prepare_frame(frame)
        frame = self._reduce_frame(frame)
        threshold = self._threshold_value(frame, summary or {})
        payload = self._to_payload(frame, threshold)
        html = self._render_html(payload, threshold, output)
        output.write_text(html, encoding="utf-8")
        return output

    def _prepare_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        working = frame.copy()
        working["timestamp"] = pd.to_datetime(working["timestamp"])
        if "estimated_vessel_latitude_deg" not in working.columns:
            working["estimated_vessel_latitude_deg"] = np.nan
        if "estimated_vessel_longitude_deg" not in working.columns:
            working["estimated_vessel_longitude_deg"] = np.nan
        if "tracking_error_m" not in working.columns:
            working["tracking_error_m"] = np.nan
        if "confidence" not in working.columns:
            working["confidence"] = np.nan
        working["abs_residual_total_nt"] = working["residual_total_nt"].abs()
        working["cumulative_distance_km"] = self._cumulative_distance_km(
            working["latitude_deg"],
            working["longitude_deg"],
        )
        working["sample_index"] = np.arange(len(working), dtype=int)
        return working

    def _reduce_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if len(frame) <= self.max_points:
            return frame.reset_index(drop=True)

        must_keep = {0, len(frame) - 1}
        top_anomalies = frame["final_anomaly_score"].nlargest(min(len(frame), max(20, self.max_points // 12))).index
        closest_approach = frame["range_to_vessel_m"].nsmallest(min(len(frame), max(20, self.max_points // 12))).index
        must_keep.update(int(index) for index in top_anomalies.tolist())
        must_keep.update(int(index) for index in closest_approach.tolist())

        uniform_budget = max(self.max_points - len(must_keep), 0)
        uniform_indices = set()
        if uniform_budget > 0:
            stride = max(1, len(frame) // uniform_budget)
            uniform_indices = set(range(0, len(frame), stride))

        merged = sorted(must_keep.union(uniform_indices))
        if len(merged) > self.max_points:
            overflow = len(merged) - self.max_points
            optional = [index for index in merged if index not in must_keep]
            trimmed_optional = optional[overflow:]
            merged = sorted(must_keep.union(trimmed_optional))
        if len(merged) > self.max_points:
            merged = sorted(list(must_keep))[: self.max_points]

        reduced = frame.iloc[merged].copy().reset_index(drop=True)
        reduced["sample_index"] = np.arange(len(reduced), dtype=int)
        reduced["cumulative_distance_km"] = self._cumulative_distance_km(
            reduced["latitude_deg"],
            reduced["longitude_deg"],
        )
        return reduced

    def _threshold_value(self, frame: pd.DataFrame, summary: dict[str, object]) -> float:
        if "final_anomaly_threshold" in frame.columns:
            return float(frame["final_anomaly_threshold"].iloc[0])
        if "threshold" in summary:
            return float(summary["threshold"])
        return float(frame["final_anomaly_score"].quantile(0.975))

    def _to_payload(self, frame: pd.DataFrame, threshold: float) -> list[dict[str, object]]:
        payload: list[dict[str, object]] = []
        for row in frame.itertuples(index=False):
            payload.append(
                {
                    "sampleIndex": int(row.sample_index),
                    "timestamp": pd.Timestamp(row.timestamp).isoformat(),
                    "observedTotalNt": float(row.observed_total_nt),
                    "baselineTotalNt": float(row.baseline_total_nt),
                    "residualTotalNt": float(row.residual_total_nt),
                    "absResidualTotalNt": float(row.abs_residual_total_nt),
                    "finalAnomalyScore": float(row.final_anomaly_score),
                    "finalAnomalyThreshold": float(threshold),
                    "rangeToVesselM": float(row.range_to_vessel_m),
                    "cumulativeDistanceKm": float(row.cumulative_distance_km),
                    "latitudeDeg": float(row.latitude_deg),
                    "longitudeDeg": float(row.longitude_deg),
                    "vesselLatitudeDeg": float(row.vessel_latitude_deg),
                    "vesselLongitudeDeg": float(row.vessel_longitude_deg),
                    "estimatedVesselLatitudeDeg": None if pd.isna(row.estimated_vessel_latitude_deg) else float(row.estimated_vessel_latitude_deg),
                    "estimatedVesselLongitudeDeg": None if pd.isna(row.estimated_vessel_longitude_deg) else float(row.estimated_vessel_longitude_deg),
                    "trackingErrorM": None if pd.isna(row.tracking_error_m) else float(row.tracking_error_m),
                    "trackingConfidence": None if pd.isna(row.confidence) else float(row.confidence),
                    "isAnomaly": bool(getattr(row, "is_anomaly", False)),
                }
            )
        return payload

    def _render_html(self, payload: list[dict[str, object]], threshold: float, output_path: Path) -> str:
        globe_path = Path(self.globe_relative_path)
        if globe_path.is_absolute():
            globe_src = globe_path.name
        else:
            globe_src = globe_path.as_posix()
        html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>__TITLE__</title>
  <style>
    :root {
      --bg: #07111f;
      --panel: rgba(8, 16, 28, 0.92);
      --line: rgba(255,255,255,0.12);
      --text: #e7eef8;
      --muted: #9fb0c6;
      --gold: #ffd166;
      --cyan: #4fc3f7;
      --red: #ff7b72;
      --green: #7ee787;
      --violet: #d2a8ff;
      --blue: #58a6ff;
      --orange: #ffa657;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: radial-gradient(circle at top, #10233a 0%, var(--bg) 55%);
      color: var(--text);
      font-family: Segoe UI, sans-serif;
    }
    .page {
      padding: 18px;
      display: grid;
      grid-template-rows: auto auto 1fr;
      gap: 16px;
      min-height: 100vh;
    }
    .hero, .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: 0 20px 48px rgba(0,0,0,0.28);
    }
    .hero {
      padding: 18px 20px;
      display: grid;
      grid-template-columns: 1.5fr 1fr;
      gap: 18px;
    }
    .hero h1 {
      margin: 0 0 8px 0;
      font-size: 26px;
    }
    .hero p {
      margin: 0;
      color: var(--muted);
      line-height: 1.5;
      font-size: 14px;
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .stat {
      padding: 12px;
      border-radius: 12px;
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.08);
    }
    .stat .label {
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    .stat .value {
      margin-top: 6px;
      font-size: 20px;
      font-weight: 700;
    }
    .controls {
      display: grid;
      grid-template-columns: 1fr auto auto auto;
      gap: 12px;
      align-items: center;
      padding: 16px 18px;
    }
    .controls .readout {
      font-size: 14px;
      color: var(--muted);
    }
    input[type="range"] {
      width: 100%;
      accent-color: var(--cyan);
    }
    button {
      padding: 10px 14px;
      border-radius: 10px;
      border: 0;
      cursor: pointer;
      color: white;
      background: #1f7ae0;
    }
      .grid {
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr);
      grid-template-rows: minmax(280px, 34vh) minmax(280px, 34vh) minmax(260px, 30vh) minmax(240px, 28vh);
      gap: 16px;
    }
    .card {
      padding: 14px 14px 12px 14px;
      display: flex;
      flex-direction: column;
      min-height: 0;
    }
    .card h2 {
      margin: 0 0 4px 0;
      font-size: 16px;
    }
    .card p {
      margin: 0 0 10px 0;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .card svg {
      width: 100%;
      height: 100%;
      border-radius: 12px;
      background: linear-gradient(180deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.01) 100%);
      border: 1px solid rgba(255,255,255,0.06);
      cursor: crosshair;
    }
    .map-card {
      grid-row: span 2;
      padding: 0;
      overflow: hidden;
    }
    .map-header {
      padding: 14px;
      border-bottom: 1px solid var(--line);
      background: rgba(255,255,255,0.03);
    }
    .map-card iframe {
      width: 100%;
      height: calc(100% - 72px);
      border: 0;
      background: #091420;
    }
    .legend {
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      font-size: 12px;
      color: var(--muted);
      margin-top: 8px;
    }
    .legend span::before {
      content: "";
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      margin-right: 6px;
      vertical-align: middle;
    }
    .observed::before { background: var(--gold); }
    .baseline::before { background: var(--cyan); }
    .residual::before { background: var(--red); }
    .score::before { background: var(--green); }
    .threshold::before { background: var(--violet); }
    .range::before { background: var(--blue); }
    .closest::before { background: var(--orange); }
    .footer-note {
      color: var(--muted);
      font-size: 12px;
      margin-top: 8px;
    }
    @media (max-width: 1200px) {
      .hero { grid-template-columns: 1fr; }
      .grid { grid-template-columns: 1fr; grid-template-rows: repeat(5, minmax(260px, 34vh)); }
      .map-card { grid-row: span 1; }
    }
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div>
        <h1>__TITLE__</h1>
        <p>
          Four linked views of the Bahamas run: a realtime anomaly strip chart, vessel proximity versus anomaly,
          a synchronized NOAA plus anomaly globe, and an along-track residual profile. The charts and globe share
          one current-time cursor.
        </p>
      </div>
      <div class="stats">
        <div class="stat"><div class="label">Samples In Dashboard</div><div class="value" id="statSamples">0</div></div>
        <div class="stat"><div class="label">Threshold</div><div class="value" id="statThreshold">0.000</div></div>
        <div class="stat"><div class="label">Current Range</div><div class="value" id="statRange">0.00 km</div></div>
        <div class="stat"><div class="label">Current Score</div><div class="value" id="statScore">0.000</div></div>
      </div>
    </section>
    <section class="controls hero">
      <div>
        <div class="readout" id="currentTimestamp">Timestamp</div>
        <input id="timeSlider" type="range" min="0" max="0" value="0" step="1" />
      </div>
      <button id="playPauseBtn" type="button">Play</button>
      <button id="jumpClosestBtn" type="button">Closest Approach</button>
      <button id="jumpPeakBtn" type="button">Top Anomaly</button>
    </section>
    <section class="grid">
      <article class="card">
        <h2>1. Realtime Anomaly Strip Chart</h2>
        <p>Observed field, NOAA baseline, residual, anomaly score, and threshold share one time cursor.</p>
        <svg id="stripChart" viewBox="0 0 960 300" preserveAspectRatio="none"></svg>
        <div class="legend">
          <span class="observed">Observed Total</span>
          <span class="baseline">NOAA Baseline</span>
          <span class="residual">Residual Total</span>
          <span class="score">Final Anomaly Score</span>
          <span class="threshold">Threshold</span>
        </div>
      </article>
      <article class="card map-card">
        <div class="map-header">
          <h2>3. Globe Track Colored By Anomaly</h2>
          <p>Aircraft and vessel tracks are time-synchronized with the dashboard. Use the globe controls for NOAA-only or NOAA plus anomaly mode.</p>
        </div>
        <iframe id="globeFrame" src="__GLOBE_SRC__" title="Bahamas NOAA and anomaly globe"></iframe>
      </article>
      <article class="card">
        <h2>2. Vessel Proximity vs Anomaly</h2>
        <p>Range to vessel is contrasted against anomaly response to support the causal story without overstating ground truth.</p>
        <svg id="rangeChart" viewBox="0 0 960 300" preserveAspectRatio="none"></svg>
        <div class="legend">
          <span class="range">Range To Vessel</span>
          <span class="residual">Absolute Residual</span>
          <span class="score">Final Anomaly Score</span>
        </div>
      </article>
      <article class="card">
        <h2>Estimated Magnetic Track vs True Vessel Track</h2>
        <p>The magnetic-only vessel estimate is plotted against the reference vessel path in a local plan view around the Bahamas run.</p>
        <svg id="trackChart" viewBox="0 0 960 300" preserveAspectRatio="none"></svg>
        <div class="legend">
          <span class="range">True Vessel Track</span>
          <span class="score">Estimated Magnetic Track</span>
          <span class="closest">Current Time Pair</span>
        </div>
      </article>
      <article class="card" style="grid-column: 1 / -1;">
        <h2>4. Along-Track Residual Profile</h2>
        <p>Residual magnitude is shown against cumulative track distance, with closest approach and peak anomaly markers overlaid.</p>
        <svg id="profileChart" viewBox="0 0 1320 280" preserveAspectRatio="none"></svg>
        <div class="legend">
          <span class="residual">Residual Total</span>
          <span class="closest">Closest Approach</span>
          <span class="score">Top Anomaly</span>
        </div>
        <div class="footer-note">The dashboard is reduced for interaction. The linked globe preserves the existing Cesium review workflow for spatial context.</div>
      </article>
    </section>
  </div>
  <script>
    const dashboardTitle = __TITLE_JSON__;
    const points = __POINTS_JSON__;
    const thresholdValue = __THRESHOLD_JSON__;
    let currentIndex = 0;
    let playTimer = null;
    let suppressViewerEcho = false;

    const closestApproachIndex = points.reduce((best, point, index) =>
      point.rangeToVesselM < points[best].rangeToVesselM ? index : best, 0);
    const peakAnomalyIndex = points.reduce((best, point, index) =>
      point.finalAnomalyScore > points[best].finalAnomalyScore ? index : best, 0);

    function formatTimestamp(value) {
      const date = new Date(value);
      return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
    }

    function formatNumber(value, digits = 2) {
      return Number(value).toFixed(digits);
    }

    function clampIndex(index) {
      return Math.min(Math.max(index, 0), Math.max(points.length - 1, 0));
    }

    function setCurrentIndex(index, syncViewer = true) {
      currentIndex = clampIndex(index);
      document.getElementById("timeSlider").value = String(currentIndex);
      updateReadout();
      drawAllCharts();
      if (syncViewer) {
        postTimeToViewer();
      }
    }

    function postTimeToViewer() {
      const iframe = document.getElementById("globeFrame");
      if (!iframe || !iframe.contentWindow || points.length === 0) {
        return;
      }
      iframe.contentWindow.postMessage({
        type: "mad-ai-set-timestamp",
        source: "dashboard",
        timestamp: points[currentIndex].timestamp
      }, "*");
    }

    function updateReadout() {
      const point = points[currentIndex];
      document.getElementById("currentTimestamp").textContent =
        `${dashboardTitle} | ${formatTimestamp(point.timestamp)} | sample ${point.sampleIndex}`;
      document.getElementById("statSamples").textContent = String(points.length);
      document.getElementById("statThreshold").textContent = formatNumber(thresholdValue, 3);
      document.getElementById("statRange").textContent = `${formatNumber(point.rangeToVesselM / 1000.0, 2)} km`;
      document.getElementById("statScore").textContent = formatNumber(point.finalAnomalyScore, 3);
    }

    function chartPadding() {
      return { left: 58, right: 58, top: 20, bottom: 34 };
    }

    function xPosition(index, width, padding) {
      if (points.length <= 1) {
        return padding.left;
      }
      return padding.left + (index / (points.length - 1)) * (width - padding.left - padding.right);
    }

    function createSeriesPath(values, width, height, yMin, yMax, padding) {
      const span = Math.max(yMax - yMin, 1e-9);
      return values.map((value, index) => {
        const x = xPosition(index, width, padding);
        const y = padding.top + (1.0 - (value - yMin) / span) * (height - padding.top - padding.bottom);
        return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
      }).join(" ");
    }

    function createHorizontalLine(value, width, height, yMin, yMax, padding) {
      const span = Math.max(yMax - yMin, 1e-9);
      const y = padding.top + (1.0 - (value - yMin) / span) * (height - padding.top - padding.bottom);
      return `M ${padding.left} ${y.toFixed(2)} L ${(width - padding.right).toFixed(2)} ${y.toFixed(2)}`;
    }

    function baseSvg(svgId) {
      const svg = document.getElementById(svgId);
      const width = svg.viewBox.baseVal.width || 960;
      const height = svg.viewBox.baseVal.height || 300;
      const padding = chartPadding();
      svg.innerHTML = "";
      const background = document.createElementNS("http://www.w3.org/2000/svg", "g");
      for (let step = 0; step <= 4; step += 1) {
        const y = padding.top + ((height - padding.top - padding.bottom) / 4) * step;
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", String(padding.left));
        line.setAttribute("x2", String(width - padding.right));
        line.setAttribute("y1", String(y));
        line.setAttribute("y2", String(y));
        line.setAttribute("stroke", "rgba(255,255,255,0.10)");
        line.setAttribute("stroke-width", "1");
        background.appendChild(line);
      }
      svg.appendChild(background);
      svg.onclick = (event) => {
        const rect = svg.getBoundingClientRect();
        const ratio = (event.clientX - rect.left) / rect.width;
        setCurrentIndex(Math.round(ratio * (points.length - 1)));
      };
      return { svg, width, height, padding };
    }

    function appendPath(svg, d, stroke, strokeWidth, dashArray = "") {
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", d);
      path.setAttribute("fill", "none");
      path.setAttribute("stroke", stroke);
      path.setAttribute("stroke-width", strokeWidth);
      if (dashArray) {
        path.setAttribute("stroke-dasharray", dashArray);
      }
      svg.appendChild(path);
      return path;
    }

    function appendText(svg, x, y, text, anchor = "start", fill = "rgba(231,238,248,0.88)", size = 11) {
      const node = document.createElementNS("http://www.w3.org/2000/svg", "text");
      node.setAttribute("x", String(x));
      node.setAttribute("y", String(y));
      node.setAttribute("fill", fill);
      node.setAttribute("font-size", String(size));
      node.setAttribute("text-anchor", anchor);
      node.textContent = text;
      svg.appendChild(node);
      return node;
    }

    function appendCursor(svg, x, height, padding, label) {
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", String(x));
      line.setAttribute("x2", String(x));
      line.setAttribute("y1", String(padding.top));
      line.setAttribute("y2", String(height - padding.bottom));
      line.setAttribute("stroke", "rgba(255,255,255,0.72)");
      line.setAttribute("stroke-width", "1.5");
      line.setAttribute("stroke-dasharray", "5 4");
      svg.appendChild(line);
      appendText(svg, x, padding.top + 14, label, "middle", "rgba(255,255,255,0.88)", 11);
    }

    function drawStripChart() {
      const { svg, width, height, padding } = baseSvg("stripChart");
      const observed = points.map((point) => point.observedTotalNt);
      const baseline = points.map((point) => point.baselineTotalNt);
      const residual = points.map((point) => point.residualTotalNt);
      const score = points.map((point) => point.finalAnomalyScore);
      const leftMin = Math.min(...observed, ...baseline, ...residual);
      const leftMax = Math.max(...observed, ...baseline, ...residual);
      const rightMin = 0.0;
      const rightMax = Math.max(thresholdValue, ...score) * 1.12;

      appendPath(svg, createSeriesPath(observed, width, height, leftMin, leftMax, padding), "#ffd166", 2.1);
      appendPath(svg, createSeriesPath(baseline, width, height, leftMin, leftMax, padding), "#4fc3f7", 2.0);
      appendPath(svg, createSeriesPath(residual, width, height, leftMin, leftMax, padding), "#ff7b72", 1.6);
      appendPath(svg, createSeriesPath(score, width, height, rightMin, rightMax, padding), "#7ee787", 1.8, "4 3");
      appendPath(svg, createHorizontalLine(thresholdValue, width, height, rightMin, rightMax, padding), "#d2a8ff", 1.4, "7 4");

      appendText(svg, padding.left, 14, `Field / Residual (${formatNumber(leftMin, 1)} to ${formatNumber(leftMax, 1)} nT)`);
      appendText(svg, width - padding.right, 14, `Score (0 to ${formatNumber(rightMax, 3)})`, "end");

      const cursorX = xPosition(currentIndex, width, padding);
      appendCursor(svg, cursorX, height, padding, "Current");
    }

    function drawRangeChart() {
      const { svg, width, height, padding } = baseSvg("rangeChart");
      const range = points.map((point) => point.rangeToVesselM / 1000.0);
      const absResidual = points.map((point) => point.absResidualTotalNt);
      const score = points.map((point) => point.finalAnomalyScore);
      const leftMin = Math.min(...range);
      const leftMax = Math.max(...range);
      const rightMin = 0.0;
      const rightMax = Math.max(...absResidual, ...score, thresholdValue) * 1.12;

      appendPath(svg, createSeriesPath(range, width, height, leftMin, leftMax, padding), "#58a6ff", 2.0);
      appendPath(svg, createSeriesPath(absResidual, width, height, rightMin, rightMax, padding), "#ffa657", 1.9);
      appendPath(svg, createSeriesPath(score, width, height, rightMin, rightMax, padding), "#7ee787", 1.6, "4 3");
      appendText(svg, padding.left, 14, `Range (${formatNumber(leftMin, 2)} to ${formatNumber(leftMax, 2)} km)`);
      appendText(svg, width - padding.right, 14, `Response (${formatNumber(rightMax, 2)} max)`, "end");

      const cursorX = xPosition(currentIndex, width, padding);
      appendCursor(svg, cursorX, height, padding, "Current");
    }

    function drawProfileChart() {
      const { svg, width, height, padding } = baseSvg("profileChart");
      const distance = points.map((point) => point.cumulativeDistanceKm);
      const residual = points.map((point) => point.residualTotalNt);
      const yMin = Math.min(...residual);
      const yMax = Math.max(...residual);
      appendPath(svg, createSeriesPath(residual, width, height, yMin, yMax, padding), "#ff7b72", 1.9);

      const closestX = xPosition(closestApproachIndex, width, padding);
      const closestBand = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      closestBand.setAttribute("x", String(Math.max(closestX - 8, padding.left)));
      closestBand.setAttribute("y", String(padding.top));
      closestBand.setAttribute("width", "16");
      closestBand.setAttribute("height", String(height - padding.top - padding.bottom));
      closestBand.setAttribute("fill", "rgba(255,166,87,0.18)");
      svg.appendChild(closestBand);

      const topAnomalies = points
        .map((point, index) => ({ point, index }))
        .sort((left, right) => right.point.finalAnomalyScore - left.point.finalAnomalyScore)
        .slice(0, 5);
      topAnomalies.forEach(({ point, index }) => {
        const x = xPosition(index, width, padding);
        const y = padding.top + (1.0 - (point.residualTotalNt - yMin) / Math.max(yMax - yMin, 1e-9)) * (height - padding.top - padding.bottom);
        const marker = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        marker.setAttribute("cx", String(x));
        marker.setAttribute("cy", String(y));
        marker.setAttribute("r", "4.2");
        marker.setAttribute("fill", "#7ee787");
        marker.setAttribute("stroke", "white");
        marker.setAttribute("stroke-width", "1.2");
        svg.appendChild(marker);
      });

      appendText(svg, padding.left, 14, `Residual (${formatNumber(yMin, 1)} to ${formatNumber(yMax, 1)} nT)`);
      appendText(svg, width - padding.right, 14, `Track Distance ${formatNumber(distance[distance.length - 1] || 0.0, 1)} km`, "end");

      const cursorX = xPosition(currentIndex, width, padding);
      appendCursor(svg, cursorX, height, padding, "Current");
    }

    function projectTrack(point, refLatDeg, refLonDeg) {
      const metersPerDegLat = 111320.0;
      const metersPerDegLon = 111320.0 * Math.cos(refLatDeg * Math.PI / 180.0);
      return {
        x: (point.lon - refLonDeg) * metersPerDegLon / 1000.0,
        y: (point.lat - refLatDeg) * metersPerDegLat / 1000.0
      };
    }

    function drawTrackChart() {
      const { svg, width, height, padding } = baseSvg("trackChart");
      const validPoints = points.filter((point) => point.estimatedVesselLatitudeDeg != null && point.estimatedVesselLongitudeDeg != null);
      if (validPoints.length === 0) {
        appendText(svg, width / 2, height / 2, "No estimated vessel track is available in this dashboard.", "middle");
        return;
      }
      const refLatDeg = validPoints.reduce((sum, point) => sum + point.vesselLatitudeDeg, 0.0) / validPoints.length;
      const refLonDeg = validPoints.reduce((sum, point) => sum + point.vesselLongitudeDeg, 0.0) / validPoints.length;
      const trueTrack = validPoints.map((point) => projectTrack({ lat: point.vesselLatitudeDeg, lon: point.vesselLongitudeDeg }, refLatDeg, refLonDeg));
      const estimatedTrack = validPoints.map((point) => projectTrack({ lat: point.estimatedVesselLatitudeDeg, lon: point.estimatedVesselLongitudeDeg }, refLatDeg, refLonDeg));
      const allX = trueTrack.map((point) => point.x).concat(estimatedTrack.map((point) => point.x));
      const allY = trueTrack.map((point) => point.y).concat(estimatedTrack.map((point) => point.y));
      const xMin = Math.min(...allX);
      const xMax = Math.max(...allX);
      const yMin = Math.min(...allY);
      const yMax = Math.max(...allY);
      const spanX = Math.max(xMax - xMin, 1e-6);
      const spanY = Math.max(yMax - yMin, 1e-6);
      const toSvgX = (value) => padding.left + ((value - xMin) / spanX) * (width - padding.left - padding.right);
      const toSvgY = (value) => padding.top + (1.0 - (value - yMin) / spanY) * (height - padding.top - padding.bottom);
      const pathForPoints = (track) => track.map((point, index) => `${index === 0 ? "M" : "L"} ${toSvgX(point.x).toFixed(2)} ${toSvgY(point.y).toFixed(2)}`).join(" ");

      appendPath(svg, pathForPoints(trueTrack), "#58a6ff", 2.2);
      appendPath(svg, pathForPoints(estimatedTrack), "#7ee787", 2.2, "7 4");

      const currentPoint = points[currentIndex];
      if (currentPoint.estimatedVesselLatitudeDeg != null && currentPoint.estimatedVesselLongitudeDeg != null) {
        const currentTrue = projectTrack({ lat: currentPoint.vesselLatitudeDeg, lon: currentPoint.vesselLongitudeDeg }, refLatDeg, refLonDeg);
        const currentEstimated = projectTrack({ lat: currentPoint.estimatedVesselLatitudeDeg, lon: currentPoint.estimatedVesselLongitudeDeg }, refLatDeg, refLonDeg);
        [["#58a6ff", currentTrue], ["#ffa657", currentEstimated]].forEach(([color, point]) => {
          const marker = document.createElementNS("http://www.w3.org/2000/svg", "circle");
          marker.setAttribute("cx", String(toSvgX(point.x)));
          marker.setAttribute("cy", String(toSvgY(point.y)));
          marker.setAttribute("r", "5");
          marker.setAttribute("fill", color);
          marker.setAttribute("stroke", "white");
          marker.setAttribute("stroke-width", "1.2");
          svg.appendChild(marker);
        });
      }
      appendText(svg, padding.left, 14, `Local Easting ${formatNumber(xMin, 1)} to ${formatNumber(xMax, 1)} km`);
      appendText(svg, width - padding.right, 14, `Local Northing ${formatNumber(yMin, 1)} to ${formatNumber(yMax, 1)} km`, "end");
    }

    function drawAllCharts() {
      drawStripChart();
      drawRangeChart();
      drawTrackChart();
      drawProfileChart();
    }

    document.getElementById("timeSlider").max = String(Math.max(points.length - 1, 0));
    document.getElementById("timeSlider").addEventListener("input", (event) => {
      setCurrentIndex(Number(event.target.value));
    });

    document.getElementById("playPauseBtn").addEventListener("click", () => {
      const button = document.getElementById("playPauseBtn");
      if (playTimer) {
        clearInterval(playTimer);
        playTimer = null;
        button.textContent = "Play";
        return;
      }
      playTimer = setInterval(() => {
        setCurrentIndex((currentIndex + 1) % points.length);
      }, 850);
      button.textContent = "Pause";
    });

    document.getElementById("jumpClosestBtn").addEventListener("click", () => {
      setCurrentIndex(closestApproachIndex);
    });

    document.getElementById("jumpPeakBtn").addEventListener("click", () => {
      setCurrentIndex(peakAnomalyIndex);
    });

    window.addEventListener("message", (event) => {
      const message = event.data || {};
      if (message.source === "dashboard") {
        return;
      }
      if (message.type === "mad-ai-time-change" && typeof message.timestamp === "string") {
        const target = new Date(message.timestamp).getTime();
        let bestIndex = 0;
        let bestDistance = Math.abs(new Date(points[0].timestamp).getTime() - target);
        for (let index = 1; index < points.length; index += 1) {
          const distance = Math.abs(new Date(points[index].timestamp).getTime() - target);
          if (distance < bestDistance) {
            bestDistance = distance;
            bestIndex = index;
          }
        }
        setCurrentIndex(bestIndex, false);
      }
    });

    updateReadout();
    drawAllCharts();
    postTimeToViewer();
  </script>
</body>
</html>
"""
        return (
            html.replace("__TITLE__", escape(self.title))
            .replace("__TITLE_JSON__", json.dumps(self.title))
            .replace("__POINTS_JSON__", json.dumps(payload))
            .replace("__THRESHOLD_JSON__", json.dumps(threshold))
            .replace("__GLOBE_SRC__", escape(globe_src))
        )

    def _cumulative_distance_km(self, latitude_deg: pd.Series, longitude_deg: pd.Series) -> pd.Series:
        lat = np.radians(latitude_deg.astype(float).to_numpy())
        lon = np.radians(longitude_deg.astype(float).to_numpy())
        distance_km = np.zeros(len(lat), dtype=np.float64)
        if len(lat) <= 1:
            return pd.Series(distance_km)
        dlat = lat[1:] - lat[:-1]
        dlon = lon[1:] - lon[:-1]
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon / 2.0) ** 2
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1e-12, 1.0 - a)))
        segment_km = 6371.0 * c
        distance_km[1:] = np.cumsum(segment_km)
        return pd.Series(distance_km)
