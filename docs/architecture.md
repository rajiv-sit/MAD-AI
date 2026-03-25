# MAD-AI Architecture

## Goal

Define a Python-only architecture for magnetic anomaly detection that is modular, testable, and usable for both offline model development and interactive anomaly review.

## Current Repo Structure

```text
MAD-AI/
|-- .github/
|   `-- workflows/
|       `-- ci.yml                    # Unit-test workflow
|-- config/
|   |-- default.yaml
|   |-- inference.yaml
|   |-- training.yaml
|   `-- real_batch.yaml              # Schema mapping and split-based real-data workflow
|-- data/
|   |-- raw/
|   |   |-- noaa_wmm2025/            # Imported NOAA WMM bundle
|   |   |-- real_batch/              # Sample folder-based real-data workflow
|   |   `-- sample_sensor.csv
|   |-- processed/
|   |   |-- global_noaa/
|   |   |-- observed_scored/
|   |   `-- real_batch_scored/
|   `-- cache/                       # WMM cache artifacts
|-- docs/
|   |-- architecture.md
|   `-- mad-ai-milestone.md
|-- outputs/
|   |-- calibration/
|   |-- evaluation/
|   |-- figures/
|   |-- models/
|   `-- viewer/
|-- scripts/
|   |-- train_spatial.py
|   |-- train_temporal.py
|   |-- evaluate_fusion_models.py
|   |-- evaluate_real_batch_models.py
|   |-- score_real_batch_folder.py
|   |-- build_observed_anomaly_cesium_viewer.py
|   |-- build_real_batch_cesium_viewer.py
|   |-- generate_global_noaa_visualizations.py
|   |-- build_global_noaa_grid.py
|   `-- serve_cesium_viewer.py
|-- src/
|   `-- mad_ai/
|       |-- core/
|       |-- datasets/
|       |-- features/
|       |-- inference/
|       |-- ingest/
|       |-- models/
|       |   |-- spatial/
|       |   `-- temporal/
|       |-- utils/
|       |-- visualizer/
|       |-- viz/
|       `-- wmm/
|-- tests/
|   `-- unit/
|-- README.md
`-- pyproject.toml
```

## Architecture Layers

1. Data access layer
   Handles NOAA/WMM baseline access and sensor ingestion.

2. Feature layer
   Aligns observed data with WMM baseline values and computes residual features.

3. Dataset layer
   Builds spatial tensors and temporal sequences for learning and inference.

4. Model layer
   Contains the CNN and LSTM anomaly models plus save/load contracts.

5. Inference layer
   Scores spatial and temporal behavior, calibrates thresholds, and fuses anomaly outputs.

6. Visualization layer
   Produces static figures and an interactive Cesium review globe.

## Core Interfaces

The project keeps stable class contracts in [src/mad_ai/core/base.py](c:/Users/MrSit/source/repos/MAD-AI/src/mad_ai/core/base.py):

- `BaseMagneticModel`
- `BaseDataIngestor`
- `BaseFeatureBuilder`
- `BaseDatasetBuilder`
- `BaseAnomalyModel`
- `BaseVisualizer`
- `BaseViewModel`

These are the main inheritance paths in the current repo:

- `BaseMagneticModel` -> `WMMMagneticModel`
- `BaseDataIngestor` -> `CsvSensorIngestor`, `ParquetSensorIngestor`, `SchemaMappedSensorIngestor`, `BatchSensorIngestor`
- `BaseFeatureBuilder` -> `ResidualFeatureBuilder`, `TemporalFeatureBuilder`
- `BaseDatasetBuilder` -> `SpatialGridBuilder`, `TemporalSequenceBuilder`
- `BaseAnomalyModel` -> `CNNAnomalyModel`, `LSTMAnomalyModel`
- `BaseVisualizer` -> `HeatmapVisualizer`, `ContourMapVisualizer`, `Surface3DVisualizer`, `Globe3DVisualizer`, `AnomalyReportVisualizer`

## Main Components

### WMM Layer

Located under [src/mad_ai/wmm](c:/Users/MrSit/source/repos/MAD-AI/src/mad_ai/wmm).

Responsibilities:

- query geomagnetic baseline values from WMM-backed providers
- cache repeated lookups
- expose total field, declination, inclination, and vector components

Primary implementation:

- [src/mad_ai/wmm/model.py](c:/Users/MrSit/source/repos/MAD-AI/src/mad_ai/wmm/model.py)

### Ingestion Layer

Located under [src/mad_ai/ingest](c:/Users/MrSit/source/repos/MAD-AI/src/mad_ai/ingest).

Responsibilities:

- load CSV and Parquet sensor files
- normalize timestamps
- map non-canonical sensor schemas into the project schema
- batch-load folders of files
- load split-based real-data workflows

Primary implementations:

- [src/mad_ai/ingest/sensor.py](c:/Users/MrSit/source/repos/MAD-AI/src/mad_ai/ingest/sensor.py)

### Feature Layer

Located under [src/mad_ai/features](c:/Users/MrSit/source/repos/MAD-AI/src/mad_ai/features).

Responsibilities:

- compute WMM baseline joins
- compute total/declination/inclination residuals
- add temporal delta and rolling features

Primary implementations:

- [src/mad_ai/features/builders.py](c:/Users/MrSit/source/repos/MAD-AI/src/mad_ai/features/builders.py)

### Dataset Layer

Located under [src/mad_ai/datasets](c:/Users/MrSit/source/repos/MAD-AI/src/mad_ai/datasets).

Responsibilities:

- build spatial grids for CNN training and inference
- build temporal windows for LSTM training and inference

Primary implementations:

- [src/mad_ai/datasets/builders.py](c:/Users/MrSit/source/repos/MAD-AI/src/mad_ai/datasets/builders.py)

### Model Layer

Located under:

- [src/mad_ai/models/spatial](c:/Users/MrSit/source/repos/MAD-AI/src/mad_ai/models/spatial)
- [src/mad_ai/models/temporal](c:/Users/MrSit/source/repos/MAD-AI/src/mad_ai/models/temporal)

Responsibilities:

- train on nominal residual behavior
- produce anomaly scores
- persist checkpoints and training metadata

Primary implementations:

- [src/mad_ai/models/spatial/cnn.py](c:/Users/MrSit/source/repos/MAD-AI/src/mad_ai/models/spatial/cnn.py)
- [src/mad_ai/models/temporal/lstm.py](c:/Users/MrSit/source/repos/MAD-AI/src/mad_ai/models/temporal/lstm.py)

### Inference Layer

Located under [src/mad_ai/inference](c:/Users/MrSit/source/repos/MAD-AI/src/mad_ai/inference).

Responsibilities:

- score observed residuals
- calibrate thresholds
- fuse spatial and temporal outputs
- score global NOAA grids
- score observed CSV and real-batch folders

Primary implementations:

- [src/mad_ai/inference/engine.py](c:/Users/MrSit/source/repos/MAD-AI/src/mad_ai/inference/engine.py)
- [src/mad_ai/inference/observed_scoring.py](c:/Users/MrSit/source/repos/MAD-AI/src/mad_ai/inference/observed_scoring.py)
- [src/mad_ai/inference/global_scoring.py](c:/Users/MrSit/source/repos/MAD-AI/src/mad_ai/inference/global_scoring.py)

### Visualization Layer

Static visualizations:

- [src/mad_ai/viz/plots.py](c:/Users/MrSit/source/repos/MAD-AI/src/mad_ai/viz/plots.py)

Interactive review:

- [src/mad_ai/visualizer/cesium_viewer.py](c:/Users/MrSit/source/repos/MAD-AI/src/mad_ai/visualizer/cesium_viewer.py)

Current interactive capabilities:

- OpenStreetMap earth layer
- day/night lighting and earth rotation
- full-earth magnetic overlays
- component switching
- altitude and time controls
- click-on-surface inspection
- comparison swipe mode
- hotspot jumping
- anomaly-only and score-threshold filtering
- export of selected anomalies, review bundle JSON, and screenshots
- baseline-only versus fused anomaly comparison in the real-batch path

## Runtime Flows

### Global Baseline Flow

1. Build the global NOAA grid.
2. Compute global magnetic overlays.
3. Optionally score anomaly overlays on the global grid.
4. Review in the Cesium globe.

Main scripts:

- [scripts/build_global_noaa_grid.py](c:/Users/MrSit/source/repos/MAD-AI/scripts/build_global_noaa_grid.py)
- [scripts/generate_global_noaa_visualizations.py](c:/Users/MrSit/source/repos/MAD-AI/scripts/generate_global_noaa_visualizations.py)
- [scripts/score_global_noaa_anomalies.py](c:/Users/MrSit/source/repos/MAD-AI/scripts/score_global_noaa_anomalies.py)

### Observed CSV Flow

1. Load a CSV file.
2. Compute WMM residual features.
3. Score with the trained spatial, temporal, and fused models.
4. Build the observed anomaly globe.

Main scripts:

- [scripts/score_observed_csv_anomalies.py](c:/Users/MrSit/source/repos/MAD-AI/scripts/score_observed_csv_anomalies.py)
- [scripts/build_observed_anomaly_cesium_viewer.py](c:/Users/MrSit/source/repos/MAD-AI/scripts/build_observed_anomaly_cesium_viewer.py)

### Real Batch Flow

1. Load split folders from [config/real_batch.yaml](c:/Users/MrSit/source/repos/MAD-AI/config/real_batch.yaml).
2. Train on nominal real-batch splits.
3. Calibrate on nominal calibration splits.
4. Evaluate anomalous splits.
5. Score a target folder.
6. Review baseline-only and fused anomaly overlays in the real-batch globe.

Main scripts:

- [scripts/evaluate_real_batch_models.py](c:/Users/MrSit/source/repos/MAD-AI/scripts/evaluate_real_batch_models.py)
- [scripts/score_real_batch_folder.py](c:/Users/MrSit/source/repos/MAD-AI/scripts/score_real_batch_folder.py)
- [scripts/build_real_batch_cesium_viewer.py](c:/Users/MrSit/source/repos/MAD-AI/scripts/build_real_batch_cesium_viewer.py)

## Data Flow

```text
Observed sensor files + coordinates + timestamps
        |
        v
Ingestion layer
        |
        v
WMM baseline queries
        |
        v
Residual and temporal feature builders
        |
        +-------------------+
        |                   |
        v                   v
Spatial grids         Temporal sequences
        |                   |
        v                   v
CNN model             LSTM model
        |                   |
        +---------+---------+
                  |
                  v
          Fusion engine + thresholds
                  |
        +---------+---------+
        |                   |
        v                   v
 Saved scored artifacts   Cesium review globe
```

## Complexity Notes

- WMM queries are designed to stay near `O(n)` for `n` queried points, with cache-backed reuse for repeated lookups.
- Feature engineering is near-linear and avoids repeated joins where possible.
- Spatial grid building is direct-placement `O(n)` in the common case.
- Temporal window building is bounded by configured window sizes.
- Online scoring does not retrain models and uses persisted checkpoints and calibration artifacts.

## Testing and CI

Unit tests live in [tests/unit](c:/Users/MrSit/source/repos/MAD-AI/tests/unit).

Current status:

- `44` passing unit tests
- CI workflow at [.github/workflows/ci.yml](c:/Users/MrSit/source/repos/MAD-AI/.github/workflows/ci.yml)

## Design Rules

- Python only
- class-based design
- inheritance where shared behavior is meaningful
- polymorphism through stable base interfaces
- prefer composition over deep inheritance trees
- keep online inference lightweight
- keep heavy preprocessing out of the interactive review path
- keep this document aligned with the code
