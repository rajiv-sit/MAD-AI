# TODO List

This file tracks the main implementation gaps that remain after the prototype milestone is complete.

## 1. Real Data Scale

- [ ] Replace bundled synthetic/sample real-batch data with larger operational datasets.
- [ ] Separate nominal, abnormal, and uncertain segments using documented dataset definitions.
- [ ] Add dataset manifests describing source, time span, platform, altitude range, and label quality.
- [ ] Record data quality issues such as missing timestamps, bad coordinates, and unit inconsistencies.

Acceptance criteria:

- At least one larger nominal dataset and one abnormal dataset can run end to end through the pipeline.
- Dataset provenance and split definitions are documented.

## 2. Magnetic Backend Fidelity

- [ ] Run the larger workflows with the NOAA/WMM-backed baseline, not only the analytic backend.
- [ ] Benchmark WMM runtime on larger observed and global workflows.
- [ ] Profile and optimize remaining bottlenecks in WMM-heavy preprocessing.
- [ ] Document when `analytic` is acceptable and when `WMM` is required.

Acceptance criteria:

- A larger real-data workflow completes successfully with the WMM backend.
- Runtime and resource costs are measured and documented.

## 3. Model Quality

- [ ] Tune CNN architecture beyond the current baseline autoencoder.
- [ ] Tune LSTM architecture beyond the current baseline autoencoder.
- [ ] Compare alternative anomaly strategies such as forecasting error, embedding distance, or transformer-style temporal models if justified.
- [ ] Add structured experiments for window size, sequence length, latent size, dropout, and weighting.
- [ ] Track best-performing configurations with reproducible config files.

Acceptance criteria:

- Model changes are compared with saved metrics, not just anecdotal visual results.
- A documented best configuration exists for the target dataset family.

## 4. Calibration Credibility

- [ ] Calibrate thresholds on larger real nominal datasets.
- [ ] Evaluate threshold robustness by platform, altitude band, and mission type where applicable.
- [ ] Compare global thresholds versus segmented thresholds.
- [ ] Save calibration reports with percentile sweeps and false-positive analysis.
- [ ] Add explicit review of nominal false positives and abnormal true positives.

Acceptance criteria:

- Thresholds are justified by real nominal distributions.
- Calibration reports exist for the target operational grouping strategy.

## 5. Evaluation Quality

- [ ] Add evaluation on real abnormal segments or externally confirmed anomaly windows.
- [ ] Produce confusion metrics per dataset and per operational segment.
- [ ] Save ROC-like or precision-recall style analysis where labels are reliable enough.
- [ ] Add side-by-side comparison between baseline-only residual scoring and fused model scoring on real data.

Acceptance criteria:

- Real-data evaluation artifacts exist and are reproducible.
- Model and baseline-only comparisons are documented.

## 6. Source Connectors

- [ ] Add connectors only where actual deployment needs them.
- [ ] Possible next connectors:
  - API-based ingestion
  - database-backed ingestion
  - message-stream ingestion
  - live sensor feed adapters
- [ ] Add schema mapping templates for each real source family.
- [ ] Add validation for units, coordinate systems, and timestamp conventions per source.

Acceptance criteria:

- Each added connector is covered by tests and an example config.
- Connector additions are driven by real source requirements, not speculative scope.

## 7. Global Baseline Scaling

- [ ] Evaluate whether denser than `2 deg x 2 deg` global grids are useful enough to justify cost.
- [ ] If needed, optimize for `1 deg` or sub-`2 deg` NOAA/WMM global grids.
- [ ] Add optional parallelized or chunked global build paths if they materially help.
- [ ] Benchmark overlay generation time and viewer asset size at higher resolutions.

Acceptance criteria:

- A documented decision exists on the target global grid resolution.
- If higher resolution is adopted, generation remains operationally practical.

## 8. Visualizer Hardening

- [ ] Validate the viewer on larger real scored datasets.
- [ ] Improve review UX where analysts actually struggle.
- [ ] Consider a native desktop shell only if the browser review flow becomes limiting.
- [ ] Add stronger export/report workflows if needed for analyst handoff.
- [ ] Add clearer legends and help text for anomaly overlays and threshold interpretation.

Acceptance criteria:

- Reviewers can inspect larger real datasets without major usability issues.
- Exported outputs support the intended review workflow.

## 9. Coverage And Test Hardening

- [ ] Raise unit-test coverage above the current `85%`.
- [ ] Add targeted tests for:
  - config parsing branches
  - inference helper branches
  - abstract base class behavior
  - WMM fallback branches
- [ ] Add more script-level regression tests where runtime is reasonable.
- [ ] Keep CI stable with the growing workflow surface.

Acceptance criteria:

- Coverage reaches the next agreed target.
- CI validates the critical workflows without excessive runtime.

## 10. Production Hardening

- [ ] Define whether this project remains a research prototype or becomes an operational service.
- [ ] If operationalized, add:
  - packaging and deployment strategy
  - environment/config management
  - logging and monitoring
  - error handling and recovery
  - artifact/version retention policy
  - access control for deployed review tools

Acceptance criteria:

- Deployment assumptions are explicit.
- Operational requirements are implemented only where they are truly needed.

## 11. Documentation Follow-Up

- [ ] Keep README, architecture, and milestone docs aligned with future changes.
- [ ] Add a dedicated experiment/results summary if tuning becomes more extensive.
- [ ] Add data-governance notes if real operational datasets are introduced.

Acceptance criteria:

- Docs remain current after major implementation changes.

## Suggested Order

1. Real data scale
2. Calibration credibility
3. Model quality
4. Evaluation quality
5. Magnetic backend fidelity
6. Source connectors as needed
7. Coverage and production hardening
