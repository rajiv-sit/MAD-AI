# MAD-AI Architecture

## Goal

Define a Python-only architecture for magnetic anomaly detection that is modular, testable, object-oriented, and scalable enough for both offline training and online anomaly inference.

## Project Structure

```text
MAD-AI/
├── .venv/                          # Local virtual environment (optional, not committed)
├── config/                         # YAML/JSON configs for data, training, inference, thresholds
│   ├── default.yaml
│   ├── training.yaml
│   └── inference.yaml
├── data/
│   ├── raw/                        # Raw sensor files and source datasets
│   ├── interim/                    # Cleaned and aligned intermediate data
│   ├── processed/                  # Model-ready tensors, grids, sequences
│   └── cache/                      # Cached WMM query results and reusable artifacts
├── docs/
│   ├── architecture.md             # This document
│   └── milestone.md                # Project milestone and delivery plan
├── notebooks/                      # Research notebooks and validation experiments
├── outputs/
│   ├── figures/                    # Heatmaps, plots, anomaly overlays
│   ├── reports/                    # Evaluation summaries and anomaly reports
│   └── logs/                       # Training and inference logs
├── scripts/                        # Entry-point scripts for training, visualization, inference
│   ├── train_spatial.py
│   ├── train_temporal.py
│   ├── run_inference.py
│   ├── generate_heatmaps.py
│   └── launch_visualizer.py
├── src/
│   └── mad_ai/
│       ├── __init__.py
│       ├── core/                   # Base classes, shared interfaces, config loading
│       ├── wmm/                    # WMM access and baseline magnetic modeling
│       ├── ingest/                 # Sensor readers and normalization
│       ├── features/               # Residual features and preprocessing
│       ├── datasets/               # Spatial grids and temporal sequences
│       ├── models/
│       │   ├── spatial/            # CNN anomaly models
│       │   └── temporal/           # RNN/LSTM anomaly models
│       ├── inference/              # Scoring, thresholding, and fusion
│       ├── viz/                    # Heatmaps, diagnostic plots, and visualizer UI
│       ├── visualizer/             # Interactive visualizer application layer
│       └── utils/                  # File helpers, caching, metrics, timing
├── tests/
│   ├── unit/                       # Unit tests for modules and class contracts
│   ├── integration/                # Pipeline-level tests
│   └── fixtures/                   # Small test datasets
├── pyproject.toml                  # Python packaging, dependencies, tooling
├── README.md
└── mad-ai-milestone.md
```

## Architectural Style

The project uses a layered Python architecture:

1. Data access layer
   Handles WMM queries and sensor ingestion.

2. Feature layer
   Aligns observed readings with baseline values and computes residual features.

3. Dataset layer
   Produces tensors for spatial and temporal models.

4. Model layer
   Trains and runs anomaly detection models.

5. Inference layer
   Combines model outputs into a final anomaly decision.

6. Visualization layer
   Produces heatmaps, diagnostics, reports, and an interactive review surface.

## Object-Oriented Design

The codebase should use classes for all major subsystems, with inheritance and polymorphism where it improves extensibility and keeps orchestration code clean.

### Core Base Classes

```python
class BaseMagneticModel:
    def get_field(self, lat: float, lon: float, alt: float, timestamp=None) -> dict:
        raise NotImplementedError


class BaseDataIngestor:
    def load(self, source: str):
        raise NotImplementedError


class BaseFeatureBuilder:
    def transform(self, data):
        raise NotImplementedError


class BaseDatasetBuilder:
    def build(self, data):
        raise NotImplementedError


class BaseAnomalyModel:
    def train(self, train_data, val_data=None) -> None:
        raise NotImplementedError

    def score(self, batch):
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError


class BaseVisualizer:
    def render(self, data, output_path: str) -> None:
        raise NotImplementedError


class BaseViewModel:
    def build_view_state(self, data):
        raise NotImplementedError
```

### Inheritance Examples

- `BaseMagneticModel` -> `WMMMagneticModel`
- `BaseDataIngestor` -> `CsvSensorIngestor`, `ParquetSensorIngestor`, `LiveSensorIngestor`
- `BaseFeatureBuilder` -> `ResidualFeatureBuilder`, `TemporalFeatureBuilder`
- `BaseDatasetBuilder` -> `SpatialGridBuilder`, `TemporalSequenceBuilder`
- `BaseAnomalyModel` -> `CNNAnomalyModel`, `LSTMAnomalyModel`
- `BaseVisualizer` -> `HeatmapVisualizer`, `ReportVisualizer`
- `BaseViewModel` -> `MapViewModel`, `TimelineViewModel`, `AnomalyReviewViewModel`

### Polymorphism Requirement

Training and inference code should operate against base interfaces where possible. That allows the pipeline to swap a temporal model, spatial model, or data source without rewriting orchestration logic.

Example:

```python
def run_training(model: BaseAnomalyModel, dataset_builder: BaseDatasetBuilder, data):
    dataset = dataset_builder.build(data)
    model.train(dataset)
```

## Main Components

### 1. WMM Module

Responsibility:

- query WMM using `lat`, `lon`, `alt`, and optional time
- return baseline magnetic components
- cache repeated regional lookups when useful

Primary class:

- `WMMMagneticModel`

Expected outputs:

- declination
- inclination
- total intensity
- horizontal intensity
- north/east/down components when supported

### 2. Ingestion Module

Responsibility:

- load raw sensor readings
- normalize coordinate systems, units, and timestamps
- validate required columns

Primary classes:

- `CsvSensorIngestor`
- `ParquetSensorIngestor`
- `LiveSensorIngestor`

### 3. Feature Engineering Module

Responsibility:

- align observations with WMM baseline
- compute residual features
- generate temporal deltas and rolling statistics

Primary classes:

- `ResidualFeatureBuilder`
- `TemporalFeatureBuilder`

Key features:

- `observed_total - wmm_total`
- `observed_declination - wmm_declination`
- `observed_inclination - wmm_inclination`
- local spatial gradients
- moving averages and sequence deltas

### 4. Dataset Module

Responsibility:

- convert tabular features into model-ready tensors

Primary classes:

- `SpatialGridBuilder`
- `TemporalSequenceBuilder`

Outputs:

- CNN tensors shaped as region grids with feature channels
- RNN/LSTM tensors shaped as fixed-length sequences

### 5. Spatial Model Module

Responsibility:

- learn normal spatial magnetic structure
- score deviations from normal patterns

Primary class:

- `CNNAnomalyModel`

Possible strategies:

- autoencoder reconstruction error
- one-class embedding distance
- prediction error against baseline targets

### 6. Temporal Model Module

Responsibility:

- learn normal temporal magnetic behavior
- score unusual sequence evolution

Primary class:

- `LSTMAnomalyModel`

Possible strategies:

- sequence autoencoder
- sequence forecasting error
- hidden-state distance thresholding

### 7. Inference Module

Responsibility:

- run spatial and temporal models
- combine scores
- apply thresholds
- emit final anomaly flags

Primary classes:

- `SpatialScorer`
- `TemporalScorer`
- `AnomalyFusionEngine`

Output contract:

```python
{
    "spatial_score": float,
    "temporal_score": float,
    "final_score": float,
    "is_anomaly": bool
}
```

### 8. Visualization Module

Responsibility:

- generate magnetic heatmaps
- show anomaly overlays
- export evaluation plots and reports
- support interactive inspection of magnetic fields, residuals, and anomaly scores

Primary classes:

- `HeatmapVisualizer`
- `AnomalyReportVisualizer`

### 9. Visualizer Application

Responsibility:

- provide an operator-facing interface for exploring magnetic data
- display WMM baseline maps and observed sensor maps side by side
- visualize residual magnitude and anomaly regions
- inspect temporal sequences and anomaly-score timelines
- load saved model outputs and evaluation artifacts

Recommended implementation:

- Python desktop application using `PySide6` and `matplotlib`
- start simple with a local desktop tool instead of a web app
- keep plotting logic separate from UI widgets

Primary classes:

- `VisualizerApp`
- `MapPanel`
- `TimelinePanel`
- `AnomalyTablePanel`
- `MapViewModel`
- `TimelineViewModel`

Key screens:

- regional magnetic heatmap view
- observed versus WMM residual map
- anomaly-score timeline
- anomaly event list with drill-down details

Key inputs:

- processed grid data
- processed temporal sequences
- saved inference outputs
- cached WMM baseline values

Key outputs:

- interactive anomaly review
- exported figures
- exported anomaly summaries

## Data Flow

```text
Sensor Data + Coordinates
        |
        v
Sensor Ingestor
        |
        v
WMM Baseline Query
        |
        v
Feature Builder
        |
        +--------------------+
        |                    |
        v                    v
Spatial Grid Builder   Temporal Sequence Builder
        |                    |
        v                    v
CNN Model              LSTM/RNN Model
        |                    |
        +---------+----------+
                  |
                  v
         Anomaly Fusion Engine
                  |
        +---------+----------+
        |                    |
        v                    v
  Reports and Artifacts    Visualizer App
```

## Complexity Considerations

Time and space complexity must be treated as design requirements, not afterthoughts.

### WMM Queries

- target: `O(n)` for `n` points queried
- memory: `O(n)` for stored outputs
- optimization: cache repeated coordinate/time queries for fixed regions

### Feature Engineering

- target: near-linear scaling with batch/vectorized transforms
- avoid repeated joins and repeated WMM recomputation for the same samples
- memory usage should avoid duplicating large intermediate tables unnecessarily

### Spatial Grid Construction

- target: `O(n)` when point-to-grid indexing is direct
- avoid full-grid rescans for each data point
- keep grid resolution bounded and configurable

### Temporal Sequence Construction

- target: `O(n)` to `O(n * w)` depending on sliding-window size `w`
- reuse buffers where possible
- set an explicit maximum sequence length for online inference

### Model Inference

- online anomaly scoring should be bounded and predictable
- inference should not trigger training-time preprocessing or full-dataset scans
- keep both spatial and temporal models batchable

### Storage

- use compact formats such as Parquet, NumPy arrays, or framework tensor files
- persist processed datasets instead of recomputing expensive transformations repeatedly
- version model checkpoints with thresholds and config metadata

## Recommended Entrypoints

- `scripts/generate_heatmaps.py`
  Generate WMM-based validation heatmaps for a region.

- `scripts/train_spatial.py`
  Train the CNN-based anomaly detector.

- `scripts/train_temporal.py`
  Train the LSTM/RNN-based anomaly detector.

- `scripts/run_inference.py`
  Score new observations and emit anomaly results.

- `scripts/launch_visualizer.py`
  Open the local visualizer for reviewing heatmaps, residuals, and anomaly outputs.

## Testing Strategy

### Unit Tests

- WMM query correctness and schema validation
- ingestion schema checks
- feature-generation correctness
- dataset shape validation
- model score output contracts

### Integration Tests

- raw sensor input to processed features
- processed features to tensors
- tensors to anomaly scores
- end-to-end scoring on a small fixture dataset

## Design Rules

- Python only
- class-based architecture
- inheritance only where shared behavior is meaningful
- polymorphism through stable base interfaces
- favor composition over deep inheritance trees
- keep online inference lightweight
- keep visualization responsive by loading cached artifacts rather than recomputing heavy pipelines in the UI thread
- keep documentation in `docs/architecture.md` aligned with implementation
