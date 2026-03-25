# MAD-AI

`MAD-AI` is a Python project for magnetic anomaly detection using:

- a geomagnetic baseline model based on NOAA World Magnetic Model data
- residual feature engineering between observed and baseline magnetic values
- spatial and temporal anomaly models
- 2D/3D visualization, including an interactive Cesium globe

The repo currently includes a working end-to-end prototype with:

- NOAA/WMM-backed baseline generation
- processed dataset builders
- trainable reference anomaly models
- evaluation and threshold calibration
- a globe visualizer with:
  - OpenStreetMap earth layer
  - day/night lighting
  - earth rotation
  - full-earth magnetic overlay
  - altitude slider
  - click-on-surface magnetic inspection
  - overlay legend showing the active value range
  - comparison swipe mode
  - anomaly filtering and hotspot jumps
  - observed anomaly globe generation from CSV inputs
  - real-batch globe generation from folder-based inputs
  - mixed-format ingestion for CSV, Parquet, JSONL, and SQLite sensor batches
  - a larger real-batch scaling config for calibration and tuning workflows

## Repo Layout

```text
MAD-AI/
|-- config/                 # YAML config files
|-- data/                   # raw, cache, and processed data artifacts
|-- docs/                   # architecture and milestone docs
|-- outputs/                # generated models, figures, calibration, evaluation, viewer assets
|-- scripts/                # runnable project entry points
|-- src/mad_ai/             # Python package
`-- tests/unit/             # unit tests
```

Important docs:

- [docs/architecture.md](c:/Users/MrSit/source/repos/MAD-AI/docs/architecture.md)
- [docs/mad-ai-milestone.md](c:/Users/MrSit/source/repos/MAD-AI/docs/mad-ai-milestone.md)

## Requirements

- Python `3.10+`
- Windows PowerShell commands below assume Windows, but the Python code is portable
- internet access is useful for the Cesium/OpenStreetMap viewer layer

## Installation

Create a virtual environment and install the project:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

If you want the optional dependencies used by ML, WMM backends, and development tools:

```powershell
pip install -e .[ml,wmm,dev]
```

If you want the optional desktop UI dependency:

```powershell
pip install -e .[viz]
```

## Quick Start

Build the sample artifacts and run the baseline inference path:

```powershell
python scripts\build_processed_datasets.py
python scripts\run_inference.py
```

Generate the global NOAA dataset and visualization artifacts:

```powershell
python scripts\build_global_noaa_grid.py
python scripts\generate_global_noaa_visualizations.py
```

## Magnetic Backends

The repo supports two magnetic baseline modes:

- `WMM`
  The operational reference path. This uses the NOAA/WMM-derived backend in [model.py](c:/Users/MrSit/source/repos/MAD-AI/src/mad_ai/wmm/model.py) and should be used for physically meaningful magnetic baseline generation.
- `analytic`
  A fast deterministic fallback from [model.py](c:/Users/MrSit/source/repos/MAD-AI/src/mad_ai/wmm/model.py). This is useful for pipeline scaling, tests, and larger bundled example runs where runtime matters more than physical fidelity.

In practice:

- use `WMM` for NOAA-backed baseline analysis and production-style workflows
- use `analytic` for the bundled large scaling config and fast local validation

## Visualizer

The main visualizer is the interactive Cesium globe:

- [outputs/viewer/cesium_global_magnetic_globe.html](c:/Users/MrSit/source/repos/MAD-AI/outputs/viewer/cesium_global_magnetic_globe.html)

Visualizer preview:

![MAD-AI Earth Visualizer](docs/EarthMag.png)

### Recommended Way To Run It

Do not rely on opening the HTML directly with `file://`. Serve the viewer directory over localhost:

```powershell
python scripts\serve_cesium_viewer.py
```

Then open:

```text
http://127.0.0.1:8765/cesium_global_magnetic_globe.html
```

If you already have the local server running, you can just reopen that URL in the browser.

### What The Visualizer Supports

- OpenStreetMap earth layer
- full-earth magnetic overlay for the active component
- magnetic component switching
- comparison swipe mode
- altitude slider
- time slider
- earth rotation
- day/night lighting
- click-on-surface inspection of magnetic values
- live min/mid/max legend for the active overlay
- anomaly-only and score-threshold filtering
- lat/lon jump and hotspot navigation
- export of selected anomalies, review bundle JSON, and screenshots

### How To Use It

1. Start the local server:

```powershell
python scripts\serve_cesium_viewer.py
```

2. Generate or refresh the viewer assets if needed:

```powershell
python scripts\build_global_noaa_grid.py
python scripts\generate_global_noaa_visualizations.py
```

3. Open:

```text
http://127.0.0.1:8765/cesium_global_magnetic_globe.html
```

4. In the viewer:

- use `Displayed Component` to switch between `Total Field`, `Declination`, `Inclination`, and `Residual`
- use the altitude slider to move between `0`, `1000`, `5000`, and `10000 m`
- click anywhere on the earth surface to inspect the nearest magnetic sample on the active altitude/time layer
- use `Pause Earth Spin` if you want a fixed globe view
- read the legend beneath the inspector to understand the current overlay scale

### Observed Anomaly Globe

To score a real observed CSV and build an anomaly-aware globe:

```powershell
python scripts\score_observed_csv_anomalies.py data\raw\sample_sensor.csv
python scripts\build_observed_anomaly_cesium_viewer.py data\raw\sample_sensor.csv
```

Outputs:

- [data/processed/observed_scored/observed_anomaly_scored.csv](c:/Users/MrSit/source/repos/MAD-AI/data/processed/observed_scored/observed_anomaly_scored.csv)
- [outputs/evaluation/observed_anomaly_summary.json](c:/Users/MrSit/source/repos/MAD-AI/outputs/evaluation/observed_anomaly_summary.json)
- [outputs/viewer/cesium_observed_anomaly_globe.html](c:/Users/MrSit/source/repos/MAD-AI/outputs/viewer/cesium_observed_anomaly_globe.html)

The observed anomaly globe supports:

- observed total field
- baseline total field
- residual total field
- spatial anomaly score
- temporal anomaly score
- final anomaly score
- comparison swipe between baseline-style and fused views when applicable

### Real Batch Workflow

For folder-based real data ingestion with schema mapping and split-aware evaluation:

```powershell
python scripts\evaluate_real_batch_models.py
python scripts\build_real_batch_cesium_viewer.py
```

Config:

- [config/real_batch.yaml](c:/Users/MrSit/source/repos/MAD-AI/config/real_batch.yaml)

Sample batch input:

- [data/raw/real_batch](c:/Users/MrSit/source/repos/MAD-AI/data/raw/real_batch)

Generated artifacts:

- [outputs/models/real_batch_spatial.pt](c:/Users/MrSit/source/repos/MAD-AI/outputs/models/real_batch_spatial.pt)
- [outputs/models/real_batch_temporal.pt](c:/Users/MrSit/source/repos/MAD-AI/outputs/models/real_batch_temporal.pt)
- [outputs/calibration/real_batch_thresholds.json](c:/Users/MrSit/source/repos/MAD-AI/outputs/calibration/real_batch_thresholds.json)
- [outputs/evaluation/real_batch_metrics.json](c:/Users/MrSit/source/repos/MAD-AI/outputs/evaluation/real_batch_metrics.json)
- [data/processed/real_batch_scored/real_batch_scored.csv](c:/Users/MrSit/source/repos/MAD-AI/data/processed/real_batch_scored/real_batch_scored.csv)
- [outputs/viewer/cesium_real_batch_globe.html](c:/Users/MrSit/source/repos/MAD-AI/outputs/viewer/cesium_real_batch_globe.html)

This workflow supports:

- multi-file folder ingestion
- mixed source formats across a batch, including CSV, Parquet, JSONL, and SQLite
- schema mapping into the project's canonical sensor columns
- nominal-train, calibration, nominal-eval, and anomalous-eval splits
- real-data recalibration from nominal data
- evaluation on abnormal segments
- calibration robustness reporting across multiple percentiles
- baseline-only residual anomaly scoring alongside fused anomaly scoring
- viewer generation from the scored real batch output
- comparison swipe mode between baseline-only and fused anomaly overlays in the globe

### Larger Real Batch Scaling Workflow

For a larger mixed-format example dataset plus tuning and stronger nominal calibration:

```powershell
python scripts\generate_large_real_batch_dataset.py
python scripts\evaluate_real_batch_models.py config\real_batch_large.yaml
python scripts\tune_real_batch_models.py config\real_batch_large.yaml
python scripts\build_real_batch_cesium_viewer.py config\real_batch_large.yaml outputs\viewer\cesium_real_batch_large_globe.html
```

Config:

- [config/real_batch_large.yaml](c:/Users/MrSit/source/repos/MAD-AI/config/real_batch_large.yaml)

Generated artifacts:

- [outputs/evaluation/real_batch_tuning_leaderboard.csv](c:/Users/MrSit/source/repos/MAD-AI/outputs/evaluation/real_batch_tuning_leaderboard.csv)
- [outputs/evaluation/real_batch_tuning_summary.json](c:/Users/MrSit/source/repos/MAD-AI/outputs/evaluation/real_batch_tuning_summary.json)
- [outputs/evaluation/real_batch_calibration_robustness.json](c:/Users/MrSit/source/repos/MAD-AI/outputs/evaluation/real_batch_calibration_robustness.json)
- [outputs/viewer/cesium_real_batch_large_globe.html](c:/Users/MrSit/source/repos/MAD-AI/outputs/viewer/cesium_real_batch_large_globe.html)

Notes:

- the bundled large scaling config uses the analytic magnetic backend for runtime practicality
- this does not replace the NOAA/WMM-backed workflow; it only makes the larger sample scaling run finish quickly
- the operational NOAA/WMM-backed path remains available through the default WMM workflows and is the reference baseline path
- the generated large dataset is still synthetic and intended for pipeline scaling checks, not production claims

### Current Global Viewer Data

The current global magnetic surface is built from:

- [data/processed/global_noaa/global_wmm_grid.csv](c:/Users/MrSit/source/repos/MAD-AI/data/processed/global_noaa/global_wmm_grid.csv)
- [data/processed/global_noaa/global_wmm_grid_metadata.json](c:/Users/MrSit/source/repos/MAD-AI/data/processed/global_noaa/global_wmm_grid_metadata.json)

Current settings:

- global grid resolution: `2 deg x 2 deg`
- altitude layers: `0`, `1000`, `5000`, `10000 m`
- timestamp: `2026-03-24T00:00:00`

## Key Scripts

### Data And Baseline

- `python scripts\build_processed_datasets.py`
  - builds processed artifacts from sample data
- `python scripts\build_processed_datasets_from_csv.py <input_csv> [output_dir]`
  - builds processed artifacts from a real CSV file
- `python scripts\build_global_noaa_grid.py`
  - builds the multi-altitude global NOAA/WMM dataset
- `python scripts\import_noaa_wmm_coefficients.py`
  - imports the official NOAA WMM coefficient bundle into `data/raw/`

### Models And Inference

- `python scripts\train_spatial.py`
  - trains the spatial reference model
- `python scripts\train_temporal.py`
  - trains the temporal reference model
- `python scripts\calibrate_thresholds.py`
  - calibrates anomaly thresholds from nominal data
- `python scripts\evaluate_models.py`
  - computes evaluation metrics and saves evaluation artifacts
- `python scripts\run_inference.py`
  - runs the anomaly inference path

### Visualization

- `python scripts\generate_visualizations.py`
  - general 2D/3D visual outputs
- `python scripts\generate_noaa_wmm_visualizations.py`
  - NOAA-labelled visual outputs
- `python scripts\generate_global_noaa_visualizations.py`
  - global 2D/3D/interactive NOAA outputs
- `python scripts\serve_cesium_viewer.py`
  - local HTTP server for the viewer assets
- `python scripts\launch_visualizer.py`
  - generates the broader visualizer artifact pack

### Specialized Globe Variants

- `python scripts\build_cesium_globe_viewer.py`
- `python scripts\build_time_series_cesium_globe.py`
- `python scripts\build_multi_altitude_cesium_globe.py`
- `python scripts\build_tracked_cesium_globe.py`
- `python scripts\build_observed_csv_cesium_viewer.py <input_csv>`
- `python scripts\build_observed_anomaly_cesium_viewer.py <input_csv>`
- `python scripts\evaluate_real_batch_models.py [config_path]`
- `python scripts\score_real_batch_folder.py <input_dir> [config_path] [output_csv] [summary_json]`
- `python scripts\build_real_batch_cesium_viewer.py [config_path] [output_html]`
- `python scripts\generate_large_real_batch_dataset.py [output_root]`
- `python scripts\tune_real_batch_models.py [config_path]`

## Testing

Current verification snapshot:

- all project phases are functionally complete
- `47` unit tests are passing
- CI runs the unit suite on pushes and pull requests

Run the unit test suite:

```powershell
python -m unittest discover -s tests\unit -p "test_*.py" -v
```

CI:

- [.github/workflows/ci.yml](c:/Users/MrSit/source/repos/MAD-AI/.github/workflows/ci.yml) runs the unit suite on pushes and pull requests

Run the Cesium viewer-focused tests:

```powershell
python -m unittest tests.unit.test_cesium_viewer -v
```

## Outputs

Important output areas:

- [outputs/models](c:/Users/MrSit/source/repos/MAD-AI/outputs/models)
- [outputs/evaluation](c:/Users/MrSit/source/repos/MAD-AI/outputs/evaluation)
- [outputs/calibration](c:/Users/MrSit/source/repos/MAD-AI/outputs/calibration)
- [outputs/figures](c:/Users/MrSit/source/repos/MAD-AI/outputs/figures)
- [outputs/viewer](c:/Users/MrSit/source/repos/MAD-AI/outputs/viewer)

## Notes

- The global viewer currently runs from a `2 deg` global grid across four altitude layers.
- The interactive globe is best run through `localhost`, not by double-clicking the HTML file.
- The viewer uses OpenStreetMap as the primary earth layer and falls back to local assets where needed.
- The large real-batch scaling config is designed to run quickly by using the analytic backend instead of WMM.
- The NOAA/WMM-backed workflow remains the correct path for real magnetic baseline interpretation.
