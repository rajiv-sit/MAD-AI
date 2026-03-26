# MAD-AI Milestone

## Objective

Deliver an end-to-end AI-based magnetic anomaly detection pipeline that:

1. Ingests World Magnetic Model (WMM) baseline data for a target region.
2. Generates magnetic field features from latitude, longitude, and altitude inputs.
3. Produces visual validation outputs such as regional magnetic heatmaps.
4. Structures the data for both spatial and temporal learning.
5. Trains spatial and temporal models on normal magnetic behavior.
6. Detects and flags deviations from the learned baseline as anomalies.

## Final Deliverable

A working prototype that accepts geographic and time-based magnetic inputs, compares observed values against WMM-derived expectations, and outputs anomaly scores, anomaly flags, visual verification artifacts, and a visualizer for inspection.

## Current Status

Implementation is complete at the prototype milestone level and includes a working end-to-end system with an interactive globe visualizer.

Completed so far:

- Python package layout under `src/mad_ai/`
- object-oriented base classes for magnetic models, dataset builders, anomaly models, and visualizers
- ingestion, feature engineering, spatial-grid, and temporal-sequence builders
- trainable PyTorch spatial and temporal anomaly models
- integrated score-fusion pipeline
- heatmap generation and an upgraded visualizer foundation with residual-map and anomaly-event outputs
- disk-backed WMM caching and processed-dataset persistence
- threshold calibration and model evaluation outputs
- a CSV-based preprocessing path for real sensor files
- unit tests for the added implementation files and runnable scripts
- direct NOAA coefficient import into the repo under `data/raw/noaa_wmm2025/`
- full-earth NOAA magnetic overlays draped over an OpenStreetMap Cesium globe
- altitude-aware global magnetic overlays for `0`, `1000`, `5000`, and `10000 m`
- earth rotation and day/night lighting in the interactive globe
- click-on-surface magnetic inspection using the active altitude and time layer
- observed CSV anomaly scoring exported to processed CSV and evaluation summary artifacts
- overlay legend support for active magnetic and anomaly components
- anomaly-capable observed globe generation from scored CSV artifacts
- denser global NOAA overlays generated at the current `2 deg x 2 deg` default resolution
- observed residual model checkpoints, evaluation metrics, and calibrated thresholds saved for reuse by the observed globe workflow
- standalone spatial and temporal training scripts now save per-model checkpoints, calibrated thresholds, score comparisons, and validation histograms
- fused anomaly calibration now reuses the standalone spatial and temporal thresholds and saves a dedicated fusion threshold artifact with fused metrics and plots
- the Cesium review flow now supports comparison swipe mode, anomaly filtering, hotspot jumps, lat/lon search, and export actions for review outputs
- real-data folder ingestion now supports schema mapping, split-aware batch loading, batch scoring, and a real-batch Cesium review path
- the real-batch review path now supports baseline-only residual anomaly overlays, fused anomaly overlays, robustness reports, and CI-backed regression checks
- quality-scaling support now includes a larger mixed-format real-batch dataset generator, stronger real nominal calibration, broader model-tuning sweeps, and JSONL/SQLite source connectors
- bundled dataset manifests now carry documented provenance, split paths, schema mappings, coordinate conventions, units, and data-quality notes for the real-batch workflows
- the repo now includes a Bahamas real-data workflow with aircraft state, vessel state, and aircraft-borne magnetic total field ingestion from `.asc` data
- the Bahamas review path now includes a combined NOAA-only / NOAA-plus-anomaly globe plus a linked four-view realtime dashboard
- the Bahamas dashboard now links strip-chart, vessel-range, synchronized globe, and along-track residual views with a shared time cursor

Current limitations:

- the WMM layer now tries real Python backends first, but still keeps an analytic fallback for portability
- the larger bundled scaling workflow uses the analytic backend intentionally for runtime practicality; the NOAA/WMM-backed workflow remains the reference baseline path
- the spatial and temporal models now use trainable PyTorch autoencoder baselines, but they still need stronger architecture tuning and evaluation
- the visualizer is now interactive through a local Cesium globe, but it is browser-based rather than a richer native desktop review tool
- the bundled default real-batch example is intentionally small, so its metrics are for pipeline verification rather than performance claims
- denser full-earth builds are now practical at the current `2 deg x 2 deg` grid, but further scaling still needs care
- broader operational connectors beyond the current file-based CSV, Parquet, JSONL, and SQLite paths may still be needed depending on deployment
- the Bahamas vessel path currently comes from the dataset ship-navigation reference fields
- anomaly in the Bahamas workflow comes from aircraft magnetometer residuals against the magnetic baseline
- vessel tracking is not yet estimated from magnetic measurements alone
- true inverse magnetic tracking still needs a vessel state model, magnetic forward model, initialization logic, and a filter-based tracker

Immediate next steps:

1. replace the synthetic large-batch example with larger operational datasets
2. validate the tuned settings and calibrated thresholds on platform-specific real nominal distributions
3. define the first vessel state model and magnetic forward-model interface for inverse tracking
4. add filter-based vessel tracking from magnetic measurements and compare it against the provided ship-reference path
5. expand source connectors further only where actual deployments need them
6. raise test coverage further and add script-level regression checks where useful
7. consider a native desktop review shell only if the browser-based globe becomes limiting

## Quality Status

Current unit-test coverage for `src/mad_ai` is 85%.

Coverage summary:

- 47 unit tests are passing
- strong coverage exists for dataset, feature, ingest, visualization, and model save/load flows
- the largest remaining gaps are in config parsing branches, inference helper branches, abstract base classes, and WMM fallback branches
- new ingest coverage now includes JSONL and SQLite connectors plus the large mixed-format real-batch generator

## Implementation Language

Python is the primary implementation language for this project.

Reasoning:

- WMM access is readily available through Python libraries and NOAA-compatible tooling.
- Data preprocessing, visualization, and ML training are fastest to implement in Python.
- CNN and RNN/LSTM workflows are well supported through `pytorch` or `tensorflow`.

## System Architecture

### Architecture Goal

Build a modular Python pipeline that separates baseline magnetic modeling, sensor data ingestion, feature engineering, model training, and anomaly inference.

The implementation should use object-oriented design with clear class boundaries, inheritance where it improves extensibility, and polymorphism for interchangeable model and data-processing components.

### Object-Oriented Design Requirements

The codebase should explicitly use:

- classes for all major pipeline components
- inheritance for shared behavior across related modules
- polymorphism so multiple model types or data processors can be used through common interfaces
- encapsulation for model state, configuration, thresholds, and preprocessing rules

Recommended base abstractions:

- `BaseMagneticModel`
  Shared interface for any magnetic baseline provider.

- `BaseDataIngestor`
  Shared interface for loading sensor data from different sources.

- `BaseFeatureBuilder`
  Shared interface for creating derived features.

- `BaseDatasetBuilder`
  Shared interface for converting tabular data into model-ready tensors.

- `BaseAnomalyModel`
  Shared interface for train, save, load, and score operations.

- `BaseVisualizer`
  Shared interface for heatmaps, plots, and anomaly reports.

Example inheritance structure:

- `BaseDataIngestor` -> `CsvSensorIngestor`, `ParquetSensorIngestor`, `JsonlSensorIngestor`, `SqliteSensorIngestor`, `LiveSensorIngestor`
- `BaseMagneticModel` -> `WMMMagneticModel`, `AnalyticMagneticModel`
- `BaseDatasetBuilder` -> `SpatialGridBuilder`, `TemporalSequenceBuilder`
- `BaseAnomalyModel` -> `CNNAnomalyModel`, `LSTMAnomalyModel`
- `BaseVisualizer` -> `HeatmapVisualizer`, `AnomalyReportVisualizer`

Polymorphism expectation:

- training and inference pipelines should depend on abstract base classes rather than concrete implementations where practical
- the same orchestration code should be able to call different anomaly models through a common interface such as `train()`, `predict()`, and `score()`
- alternative magnetic models or dataset builders should be swappable with minimal pipeline changes

### High-Level Components

1. **WMM Baseline Service**
   Inputs geographic coordinates and altitude, then returns expected magnetic values from WMM.

2. **Sensor Ingestion Module**
   Reads measured magnetic sensor data and standardizes timestamps, coordinates, altitude, and units.

3. **Feature Engineering Pipeline**
   Joins observed readings with WMM baseline values and computes residual features such as:
   - observed minus expected field strength
   - observed minus expected declination
   - observed minus expected inclination
   - spatial neighborhood features
   - temporal delta and trend features

4. **Spatial Dataset Builder**
   Converts regional magnetic samples into grid tensors suitable for CNN training and inference.

5. **Temporal Dataset Builder**
   Converts ordered sensor readings into fixed-length sequences suitable for RNN/LSTM training and inference.

6. **Spatial Anomaly Model**
   A CNN-based model that learns normal spatial magnetic structure and outputs a spatial anomaly score.

7. **Temporal Anomaly Model**
   An RNN/LSTM-based model that learns normal temporal magnetic behavior and outputs a temporal anomaly score.

8. **Detection Fusion Layer**
   Combines spatial and temporal anomaly scores, applies thresholds, and produces the final anomaly flag.

9. **Visualization and Reporting Layer**
   Produces heatmaps, anomaly overlays, evaluation plots, and sample reports for verification.

### Data Flow

1. Query WMM using `lat`, `lon`, `alt`, and optional date.
2. Ingest observed magnetic sensor readings.
3. Align observed readings with WMM baseline values.
4. Compute residual and derived features.
5. Split data into:
   - spatial grids for CNN input
   - temporal sequences for RNN/LSTM input
6. Train normal-pattern models.
7. Run new data through both models during inference.
8. Fuse anomaly scores and emit a final anomaly decision.

### Recommended Python Package Layout

- `src/mad_ai/wmm/`
  WMM access, coefficient loading, and baseline field queries.

- `src/mad_ai/ingest/`
  Sensor data readers, schema validation, unit normalization, and timestamp handling.

- `src/mad_ai/features/`
  Residual feature generation and preprocessing utilities.

- `src/mad_ai/datasets/`
  Spatial grid builders and temporal sequence builders.

- `src/mad_ai/models/spatial/`
  CNN model definitions, training, and evaluation.

- `src/mad_ai/models/temporal/`
  RNN/LSTM model definitions, training, and evaluation.

- `src/mad_ai/inference/`
  End-to-end anomaly scoring, thresholding, and fusion logic.

- `src/mad_ai/viz/`
  Heatmaps, validation plots, and anomaly visualization tools.

- `tests/`
  Unit and pipeline tests.

### Recommended Runtime Stages

- **Offline stage**
  WMM setup, data preparation, visualization, model training, and threshold calibration.

- **Online stage**
  Real-time or batch ingestion of new measurements, WMM lookup, feature computation, model inference, and anomaly flagging.

### Storage Model

- Raw sensor data stored as CSV, Parquet, or database records.
- WMM baseline outputs cached for repeated regional queries when useful.
- Processed training datasets stored in reproducible versioned files.
- Trained model checkpoints stored with metadata and thresholds.

### Initial Interface Contracts

- `get_wmm_features(lat, lon, alt, date) -> dict`
- `load_sensor_readings(source) -> DataFrame`
- `build_spatial_grids(data) -> tensor`
- `build_temporal_sequences(data) -> tensor`
- `score_spatial_anomaly(grid) -> float`
- `score_temporal_anomaly(sequence) -> float`
- `fuse_anomaly_scores(spatial_score, temporal_score) -> dict`

### Complexity Requirements

Time and space complexity should be considered during design and implementation, especially for grid generation, sequence generation, training data preparation, and inference.

Targets and expectations:

- WMM query stage:
  Aim for linear time in the number of queried points, `O(n)`, with memory proportional to stored outputs, `O(n)`.

- Spatial grid construction:
  Aim for `O(n)` to place `n` sampled points into a grid when indexing is direct.
  Avoid repeated full-grid scans that push preprocessing toward `O(n * g)` where `g` is grid size.

- Temporal sequence construction:
  Aim for `O(n)` to `O(n * w)` depending on sliding-window size `w`.
  Reuse buffers where possible to reduce intermediate allocations.

- Feature engineering:
  Prefer vectorized batch transforms with predictable linear scaling.
  Avoid repeated joins or recomputation of WMM baseline values for the same coordinates when caching is possible.

- Inference:
  Keep per-sample or per-window scoring bounded and predictable.
  The online path should avoid retraining, full-dataset scans, or unnecessary visualization work.

- Storage:
  Use compact tabular or tensor formats and avoid duplicating large raw and processed datasets in memory.

Implementation guidance:

- cache repeated WMM lookups for fixed regions or repeated coordinate queries
- use batch processing for training and inference
- keep model input shapes bounded and documented
- define maximum supported grid resolution and sequence length early
- profile preprocessing stages before scaling up to larger regions or longer time windows

## Milestone Scope

### Phase 1: WMM Access and Baseline Generation

**Goal:** Establish a reliable magnetic reference model.

Status: Functionally complete

Tasks:

- Access NOAA's official WMM source.
- Download the WMM coefficients or install a supported Python library such as `geomagmodels`.
- Build a small Python module that takes `lat`, `lon`, `alt`, and optional timestamp/date as input.
- Return core magnetic outputs such as declination, inclination, total field strength, and component vectors if available.

Deliverables:

- WMM data source documented.
- Reproducible environment setup instructions.
- Baseline magnetic field query script or module.

Progress update:

- `src/mad_ai/wmm/model.py` exists and exposes `WMMMagneticModel`
- real Python backend support has been added using `geomag`, with optional `wmm2020` fallback support
- repeated WMM queries are cached in memory to keep lookup cost close to `O(1)` for repeated points
- a deterministic analytic fallback remains in place so the rest of the pipeline still runs when backend packages are unavailable
- the larger scaling config uses the analytic backend intentionally so larger bundled runs finish quickly without changing the reference WMM workflow

Acceptance criteria:

- The system can generate magnetic values for arbitrary coordinates in the region of interest.
- Outputs are numerically consistent across repeated runs with the same inputs.

### Phase 2: Visual Verification

**Goal:** Confirm that the WMM integration behaves plausibly across space.

Status: Functionally complete

Tasks:

- Define the geographic bounds and resolution of the region of interest.
- Sample the region into a regular grid.
- Generate magnetic feature maps using `matplotlib`.
- Create at least one heatmap each for total field strength and a directional feature such as declination or inclination.

Deliverables:

- Grid sampling script.
- Heatmap images for the chosen region.
- Initial visualizer-ready plot/export pipeline.
- Short validation note describing whether patterns look physically plausible.

Progress update:

- `scripts/generate_heatmaps.py` is implemented
- `src/mad_ai/viz/plots.py` can render heatmaps
- `scripts/launch_visualizer.py` exports baseline-map, residual-map, timeline, and anomaly-event review artifacts
- `scripts/generate_global_noaa_visualizations.py` generates global 2D/3D visual products
- the Cesium globe now visualizes full-earth magnetic overlays over OpenStreetMap
- the globe now exposes anomaly overlay modes when anomaly columns are present

Acceptance criteria:

- Heatmaps render successfully from generated WMM values.
- Spatial gradients appear continuous and free of obvious data gaps or indexing errors.

### Phase 3: Dataset Structuring

**Goal:** Prepare data for spatial and temporal anomaly models.

Status: Functionally complete

Tasks:

- Define the baseline dataset schema.
- Organize magnetic values into spatial grids for CNN input.
- Organize readings into time sequences for RNN/LSTM input.
- Label baseline data as normal operating behavior.
- If real sensor data exists, align observed data with WMM baseline features and residuals.

Deliverables:

- Dataset specification document.
- Preprocessing pipeline for grid and sequence generation.
- Saved training-ready datasets.

Progress update:

- `src/mad_ai/features/builders.py` implements residual and temporal feature builders
- `src/mad_ai/datasets/builders.py` implements spatial-grid and temporal-sequence builders
- processed dataset artifacts can now be persisted to disk
- a CSV-based build path exists for real sensor inputs
- split-aware real batch ingestion now exists through schema-mapped folder loading
- split-aware real batch ingestion now supports mixed CSV, Parquet, JSONL, and SQLite folders
- the global NOAA dataset currently ships as a multi-altitude grid with `2 deg x 2 deg` spacing
- observed anomaly scoring can now persist anomaly-enriched CSV artifacts for globe review
- `scripts/generate_large_real_batch_dataset.py` now creates a larger mixed-format dataset for scaling checks

Acceptance criteria:

- CNN-ready tensors can be generated from regional magnetic grids.
- RNN/LSTM-ready sequences can be generated from time-ordered readings.
- Feature engineering is reproducible from raw input data.

### Phase 4: Spatial Model Training

**Goal:** Learn normal spatial magnetic patterns.

Status: Functionally complete

Tasks:

- Build and train a CNN on spatial magnetic grids.
- Use normal-only data or baseline-centered targets depending on the anomaly strategy.
- Track reconstruction error, prediction error, or embedding distance as the anomaly signal.

Deliverables:

- CNN training script.
- Saved model checkpoint.
- Training metrics and validation plots.

Progress update:

- `scripts/train_spatial.py` exists
- `src/mad_ai/models/spatial/cnn.py` now provides a trainable PyTorch convolutional autoencoder with normalization and mini-batch training
- model checkpoints are saved to `outputs/models/`
- observed-residual training now saves reusable model checkpoints and evaluation artifacts under `outputs/models/` and `outputs/evaluation/`
- `scripts/train_spatial.py` now saves a calibrated spatial threshold, score comparison CSV, and validation histogram
- `scripts/tune_real_batch_models.py` now saves tuning leaderboards and best-candidate summaries for larger real-batch runs

Acceptance criteria:

- The CNN converges on the training data without obvious instability.
- A quantitative thresholding strategy exists for spatial anomaly scoring.

### Phase 5: Temporal Model Training

**Goal:** Learn how magnetic readings evolve over time.

Status: Functionally complete

Tasks:

- Build and train an RNN or LSTM on ordered magnetic sequences.
- Capture temporal drift, noise patterns, and normal transitions.
- Define anomaly scores from forecast error, reconstruction error, or hidden-state distance.

Deliverables:

- RNN/LSTM training script.
- Saved temporal model checkpoint.
- Sequence evaluation metrics.

Progress update:

- `scripts/train_temporal.py` exists
- `src/mad_ai/models/temporal/lstm.py` now provides a trainable PyTorch LSTM autoencoder with normalization, mini-batch training, and stronger recurrent layers
- model checkpoints are saved to `outputs/models/`
- observed-residual training now saves reusable temporal checkpoints and evaluation artifacts under `outputs/models/` and `outputs/evaluation/`
- `scripts/train_temporal.py` now saves a calibrated temporal threshold, score comparison CSV, and validation histogram
- temporal tuning is now included in the larger real-batch search space through configurable sequence-length sweeps

Acceptance criteria:

- The temporal model learns stable normal-sequence behavior.
- A quantitative thresholding strategy exists for temporal anomaly scoring.

### Phase 6: Anomaly Detection Integration

**Goal:** Combine model outputs into a usable detection workflow.

Status: Functionally complete

Tasks:

- Run new observations through the spatial model, temporal model, or both.
- Compute anomaly scores relative to the learned baseline.
- Define decision logic for flagging anomalies.
- Output anomaly reports and visual overlays where applicable.

Deliverables:

- Integrated inference pipeline.
- Anomaly scoring and flagging logic.
- Sample anomaly report or demo output.

Progress update:

- `src/mad_ai/inference/engine.py` implements score fusion and anomaly thresholding
- calibrated thresholds can be saved and reused from `outputs/calibration/thresholds.json`
- `scripts/run_inference.py` runs end to end on sample data and can load saved calibration output
- `scripts/evaluate_models.py` saves nominal-vs-anomalous comparison artifacts
- `scripts/score_global_noaa_anomalies.py` now writes global anomaly scores back into the global NOAA dataset
- `scripts/score_observed_csv_anomalies.py` persists anomaly-scored observed CSV outputs
- the global Cesium globe can now render full-earth anomaly overlays from saved scored artifacts
- `scripts/evaluate_observed_residual_models.py` now trains reusable observed-residual models, calibrates thresholds from nominal observed distributions, and saves comparison metrics and plots
- `scripts/evaluate_fusion_models.py` now calibrates a fused threshold using the standalone spatial and temporal thresholds as component scales
- `scripts/run_inference.py` and `scripts/evaluate_models.py` now consume the fused calibration artifact
- the observed anomaly globe now uses the fused calibration path rather than a separate ad hoc threshold
- `scripts/evaluate_real_batch_models.py` now trains and evaluates on schema-mapped real batch splits from folders
- `scripts/score_real_batch_folder.py` now scores arbitrary real-data folders against the trained real-batch models
- `scripts/evaluate_real_batch_models.py` now honors configurable magnetic backends so larger synthetic scaling runs can use the analytic backend
- `scripts/tune_real_batch_models.py` now performs broader parameter sweeps over window size, sequence length, stride, and fusion weights
- `scripts/score_bahamas_realtime_anomalies.py` now scores the Bahamas real dataset and writes anomaly-enriched outputs
- the Bahamas review flow now supports linked strip-chart, vessel-range, globe, and along-track analysis

Acceptance criteria:

- New inputs can be scored end-to-end.
- The system flags intentionally injected or known abnormal patterns above threshold.
- False positives are reviewed on nominal data and documented.

### Phase 7: Visualizer

**Goal:** Provide a usable interface for reviewing magnetic maps, residuals, and anomaly results.

Status: Functionally complete

Tasks:

- Build a Python visualizer application.
- Display WMM baseline heatmaps for the region of interest.
- Display observed magnetic data and residual maps.
- Display temporal anomaly-score timelines.
- Show anomaly events and allow drill-down into event details.
- Support exporting figures and anomaly summaries.

Deliverables:

- Visualizer application entry point.
- Map and timeline views.
- Sample exported visual outputs.

Progress update:

- `src/mad_ai/visualizer/app.py` and `src/mad_ai/visualizer/viewmodels.py` are implemented
- the current visualizer generates baseline-map, residual-map, anomaly-timeline, and anomaly-event CSV artifacts
- `src/mad_ai/visualizer/cesium_viewer.py` now provides an interactive Cesium globe
- the globe supports OpenStreetMap earth imagery, day/night lighting, earth rotation, altitude selection, and full-earth magnetic overlays
- the globe no longer depends on persistent magnetic sample markers for the global field view; users can click the earth surface to inspect magnetic details
- the globe now supports anomaly overlay modes and an active overlay legend/color scale
- observed anomaly CSV workflows can now generate a dedicated anomaly-aware observed globe
- the global globe is currently built from a `2 deg x 2 deg` multi-altitude NOAA dataset and uses surface overlays rather than sparse points for the main magnetic field view
- the observed anomaly globe now consumes calibrated observed-residual checkpoints and thresholds rather than relying on a one-off fixed-threshold scoring pass
- the review flow now supports comparison swipe mode, score-threshold filtering, anomaly-only filtering, hotspot jumps, direct lat/lon search, and export of selected anomalies, review bundles, and screenshots
- `scripts/build_real_batch_cesium_viewer.py` now generates a real-batch Cesium globe from folder-based scored inputs
- the real-batch viewer can now compare baseline residual anomaly overlays against fused anomaly overlays in the same globe session
- richer desktop-native drill-down UI is optional rather than required at the current prototype stage
- `scripts/build_bahamas_noaa_combined_viewer.py` now generates the Bahamas NOAA-only / NOAA-plus-anomaly globe
- `scripts/build_bahamas_realtime_dashboard.py` now generates a linked HTML dashboard for the Bahamas workflow

Acceptance criteria:

- A user can load generated artifacts and inspect anomalies visually.
- The visualizer can show baseline, observed, and residual data without rerunning training.
- The application remains responsive for normal review-sized datasets.
- The globe can inspect magnetic values from the surface overlay at the active altitude and time setting.
- The globe can expose anomaly overlays and observed-data anomaly results from saved artifacts.
- The globe can compare baseline-only and fused anomaly overlays in the same review session.

## Suggested Technical Outputs

- `wmm/` for model access and baseline generation
- `data/` for raw, processed, and sampled magnetic datasets
- `notebooks/` or `viz/` for visual verification
- `models/cnn/` for spatial training
- `models/rnn/` for temporal training
- `inference/` for anomaly scoring and reporting
- `mad-ai-milestone.md` for project tracking

## Dependencies

- Python
- `geomagmodels` or NOAA official WMM package
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tensorflow` or `pytorch`

## Risks and Mitigations

- WMM alone is a global baseline, not local anomaly truth.
  Mitigation: compare real sensor measurements against WMM expectations and train on residual behavior.

- Insufficient anomaly examples may limit supervised evaluation.
  Mitigation: use unsupervised or semi-supervised anomaly detection based on normal-only training.

- Spatial and temporal resolutions may be mismatched.
  Mitigation: standardize coordinate, altitude, and timestamp handling early in preprocessing.

## Definition of Done

This milestone is complete when:

1. WMM data is accessible and queryable from Python.
2. Regional magnetic heatmaps can be generated for verification.
3. Data can be transformed into both CNN and RNN/LSTM training formats.
4. A spatial model has been trained and saved.
5. A temporal model has been trained and saved.
6. An integrated inference pipeline produces anomaly scores and flags.
7. A visualizer can display maps, residuals, and anomaly outputs.
8. The prototype is demonstrated on sample normal data and at least one abnormal or injected-anomaly scenario.

## Verification Snapshot

Latest local verification completed on March 25, 2026:

- `python -m unittest discover -s tests\unit -p 'test_*.py' -v` passed
- unit-test coverage for `src/mad_ai` measured at 85%
- the current unit suite count is 47 passing tests
- `python scripts\train_spatial.py` ran successfully
- `python scripts\train_temporal.py` ran successfully
- `python scripts\train_spatial.py` saved spatial metrics, threshold, comparison CSV, and histogram artifacts
- `python scripts\train_temporal.py` saved temporal metrics, threshold, comparison CSV, and histogram artifacts
- `python scripts\build_processed_datasets.py` ran successfully
- `python scripts\calibrate_thresholds.py` ran successfully
- `python scripts\evaluate_models.py` ran successfully
- `python scripts\run_inference.py` ran successfully
- `python scripts\evaluate_fusion_models.py` saved fused calibration, metrics, comparison CSV, and histogram artifacts
- `python scripts\launch_visualizer.py` generated baseline-map, residual-map, timeline, and anomaly-event outputs
- `python scripts\build_global_noaa_grid.py` built the multi-altitude global NOAA dataset
- `python scripts\generate_global_noaa_visualizations.py` generated the current global globe and overlay assets
- `python scripts\score_global_noaa_anomalies.py` wrote anomaly fields into the global NOAA dataset
- `python scripts\evaluate_observed_residual_models.py` saved observed-residual checkpoints, calibration, metrics, and score histograms
- `python scripts\score_observed_csv_anomalies.py data\raw\sample_sensor.csv` generated an observed scored CSV and summary
- `python scripts\build_observed_anomaly_cesium_viewer.py data\raw\sample_sensor.csv` generated the observed anomaly globe
- the observed anomaly globe now includes comparison swipe, filter, search, hotspot-jump, and export controls
- `python scripts\evaluate_real_batch_models.py` generated real-batch models, calibration, metrics, and histogram outputs
- `python scripts\evaluate_real_batch_models.py config\real_batch_large.yaml` generated larger real-batch models, calibration, metrics, and histogram outputs
- `python scripts\generate_large_real_batch_dataset.py` generated the larger mixed-format scaling dataset
- `python scripts\tune_real_batch_models.py config\real_batch_large.yaml` generated the tuning leaderboard and summary outputs
- `python scripts\build_real_batch_cesium_viewer.py` generated the real-batch anomaly globe
- `python scripts\build_real_batch_cesium_viewer.py config\real_batch_large.yaml outputs\viewer\cesium_real_batch_large_globe.html` generated the larger real-batch anomaly globe
- `python scripts\score_bahamas_realtime_anomalies.py inspection\external_mad_repo\data\raw\mad_data.asc data\processed\bahamas\bahamas_mad_scored.csv outputs\evaluation\bahamas_mad_summary.json analytic stable_window` generated the Bahamas scored run
- `python scripts\build_bahamas_noaa_combined_viewer.py outputs\viewer\cesium_bahamas_noaa_combined.html data\processed\global_noaa\global_wmm_grid.csv data\processed\bahamas\bahamas_mad_scored.csv` generated the linked Bahamas globe
- `python scripts\build_bahamas_realtime_dashboard.py data\processed\bahamas\bahamas_mad_scored.csv outputs\evaluation\bahamas_mad_summary.json outputs\viewer\bahamas_realtime_dashboard.html cesium_bahamas_noaa_combined.html` generated the four-view Bahamas dashboard
- `.github/workflows/ci.yml` now enforces the unit suite on pushes and pull requests

## Success Metric

The project succeeds when it can distinguish normal magnetic patterns from anomalous ones with a documented and reproducible workflow, using WMM as the environmental baseline and machine learning models as the anomaly detector.
