# TODO List

This file tracks the work required after the prototype milestone.

The repo is already feature-complete enough for pipeline demos. The remaining work is mostly about credibility, scale, and operational discipline rather than adding more prototype surface area.

## Working Rules

- Do not add new scope until a real dataset or review workflow requires it.
- Prefer work that improves evidence quality over work that only improves feature count.
- Keep the NOAA/WMM-backed path as the reference baseline.
- Treat the analytic backend as a speed tool for scaling checks, tests, and selected tuning runs.
- Update `README.md`, `docs/architecture.md`, and `docs/mad-ai-milestone.md` whenever a major item here changes state.

## Current Priority Order

1. Real data scale and provenance
2. Calibration credibility
3. Evaluation quality
4. Magnetic backend fidelity at scale
5. Model quality tuning
6. Visualizer hardening
7. Coverage and CI hardening
8. Source connectors only where needed
9. Global baseline scaling
10. Production hardening if the project leaves research mode

## Phase 1: Evidence Foundation

Goal: replace "prototype works" evidence with real, documented datasets and reproducible split definitions.

### 1. Real Data Scale

- [ ] Replace bundled synthetic or sample real-batch inputs with at least one larger nominal dataset and one abnormal dataset.
- [ ] Separate nominal, abnormal, and uncertain segments using documented split definitions.
- [ ] Add per-dataset manifests with source, time span, platform, altitude range, units, coordinate convention, and label quality.
- [ ] Record data quality issues such as missing timestamps, bad coordinates, duplicate records, unit mismatches, and sparse coverage.
- [ ] Store any dataset-specific schema mapping alongside the dataset manifest or config.

Primary outputs:

- dataset manifest files
- documented split definitions
- reproducible config entries for each dataset family

Exit criteria:

- At least one larger nominal dataset and one abnormal dataset run end to end through the pipeline.
- Provenance, split logic, and known data quality limits are documented.

Dependencies:

- none; this is the starting gate for most remaining work

### 2. Calibration Credibility

- [ ] Calibrate thresholds on larger real nominal datasets rather than only bundled examples.
- [ ] Compare global thresholds against segmented thresholds by platform, altitude band, or mission type where those groupings are real.
- [ ] Save calibration reports with percentile sweeps, score distributions, and false-positive analysis.
- [ ] Add explicit review notes for nominal false positives and abnormal true positives.
- [ ] Document which threshold set is the operational default for each dataset family.

Primary outputs:

- calibration report JSON and CSV artifacts
- saved threshold decisions with dataset scope
- reviewer notes for false positives and true positives

Exit criteria:

- Thresholds are justified by real nominal distributions.
- Calibration artifacts exist for the intended operational grouping strategy.

Dependencies:

- real dataset manifests and split definitions

### 3. Evaluation Quality

- [ ] Evaluate on real abnormal segments or externally confirmed anomaly windows.
- [ ] Produce confusion metrics per dataset and per operational segment.
- [ ] Save ROC-like or precision-recall style analysis where labels are reliable enough.
- [ ] Compare baseline-only residual scoring against fused model scoring on the same real data.
- [ ] Document where labels are weak, incomplete, or only approximate.

Primary outputs:

- reproducible real-data evaluation artifacts
- per-dataset comparison summaries
- baseline-only versus fused scoring comparison

Exit criteria:

- Real-data evaluation artifacts are reproducible from configs and scripts.
- Model versus baseline-only behavior is documented on real scored data.

Dependencies:

- real dataset availability
- calibrated thresholds

## Phase 2: Scale The Reference Path

Goal: prove that the physically meaningful WMM-backed path remains practical once the data gets larger.

### 4. Magnetic Backend Fidelity

- [ ] Run the larger workflows with the NOAA/WMM-backed baseline, not only the analytic backend.
- [ ] Benchmark WMM runtime on larger observed, real-batch, and global workflows.
- [ ] Profile WMM-heavy preprocessing to identify the main cost centers.
- [ ] Optimize only the bottlenecks that materially affect operational use.
- [ ] Document when `analytic` is acceptable and when `WMM` is required.

Primary outputs:

- runtime benchmark notes
- profiling summaries
- documented backend selection guidance

Exit criteria:

- A larger real-data workflow completes successfully with the WMM backend.
- Runtime and resource costs are measured and documented.

Dependencies:

- real larger datasets

### 5. Model Quality

- [ ] Tune the CNN architecture beyond the current baseline autoencoder.
- [ ] Tune the LSTM architecture beyond the current baseline autoencoder.
- [ ] Run structured experiments over window size, sequence length, latent size, dropout, stride, and fusion weighting.
- [ ] Compare alternative anomaly strategies only if the current baselines plateau on real evaluation data.
- [ ] Track best-performing configurations with reproducible config files and saved metrics.

Primary outputs:

- experiment leaderboard
- reproducible best-config files
- metric-backed model comparison notes

Exit criteria:

- Model changes are judged by saved metrics, not visual anecdotes.
- A documented best configuration exists for each target dataset family that matters.

Dependencies:

- real-data evaluation loop
- credible thresholding

## Phase 3: Review Workflow Hardening

Goal: make the review path usable on larger real scored datasets without overbuilding the UI.

### 6. Visualizer Hardening

- [ ] Validate the viewer on larger real scored datasets and record where it slows down or becomes confusing.
- [ ] Improve the review UX only where analyst workflows actually struggle.
- [ ] Add stronger export or report flows if analyst handoff requires them.
- [ ] Add clearer legends, threshold explanations, and overlay help text.
- [ ] Consider a native desktop shell only if the browser flow becomes a real blocker.

Primary outputs:

- viewer usability notes
- targeted UX fixes
- analyst handoff exports if needed

Exit criteria:

- Reviewers can inspect larger real datasets without major usability failures.
- Exported outputs support the intended review workflow.

Dependencies:

- larger scored real datasets

### 7. Coverage And Test Hardening

- [ ] Raise unit-test coverage above the current `85%`.
- [ ] Add targeted tests for config parsing branches, inference helper branches, abstract base class behavior, and WMM fallback branches.
- [ ] Add script-level regression tests where runtime is reasonable.
- [ ] Keep CI stable as workflow coverage expands.
- [ ] Add regression checks for the highest-value scored-artifact paths before broadening to lower-value scripts.

Primary outputs:

- additional unit tests
- focused regression tests
- stable CI coverage of critical workflows

Exit criteria:

- Coverage reaches the next agreed target.
- CI protects the critical workflows without becoming too slow to use.

Dependencies:

- none for unit tests
- selected stable workflows for regression coverage

## Phase 4: Optional Expansion

Goal: expand only where real deployment pressure exists.

### 8. Source Connectors

- [ ] Add connectors only where actual deployment needs them.
- [ ] Candidate families include API ingestion, database-backed ingestion, message streams, and live sensor feeds.
- [ ] Add schema mapping templates and validation rules per real source family.
- [ ] Validate units, coordinate systems, and timestamp conventions for each added connector.

Primary outputs:

- tested connector implementations
- example configs
- source-family schema templates

Exit criteria:

- Each added connector has tests and an example config.
- Connector work is justified by real source requirements, not speculation.

Dependencies:

- concrete deployment or ingestion requirements

### 9. Global Baseline Scaling

- [ ] Decide whether denser than `2 deg x 2 deg` global grids are actually worth the cost.
- [ ] If justified, optimize for `1 deg` or sub-`2 deg` NOAA/WMM global grids.
- [ ] Add chunked or parallel global build paths only if they materially help.
- [ ] Benchmark overlay generation time, artifact size, and viewer responsiveness at higher resolutions.

Primary outputs:

- documented grid-resolution decision
- higher-resolution build support only if justified
- performance notes for viewer asset size and generation cost

Exit criteria:

- There is a documented resolution decision.
- If higher resolution is adopted, generation stays operationally practical.

Dependencies:

- benchmark data from the WMM-backed path
- viewer validation results

### 10. Production Hardening

- [ ] Decide explicitly whether the project remains a research prototype or becomes an operational service.
- [ ] If operationalized, add packaging and deployment strategy.
- [ ] Add environment and config management.
- [ ] Add logging, monitoring, failure handling, and recovery behavior.
- [ ] Define artifact retention, versioning, and access control for review tools.

Primary outputs:

- deployment decision record
- operational design notes
- production controls only if justified

Exit criteria:

- Deployment assumptions are explicit.
- Operational requirements are implemented only where they are truly needed.

Dependencies:

- a decision to leave research mode

## Documentation Follow-Up

- [ ] Keep `README.md`, `docs/architecture.md`, and `docs/mad-ai-milestone.md` aligned with the current state.
- [ ] Add an experiment or results summary if tuning work becomes deeper.
- [ ] Add data-governance notes before bringing in real operational datasets.

Exit criteria:

- Docs remain current after major implementation changes.

## Next Execution Sprint

This is the recommended near-term sprint before adding any new feature scope.

1. Land dataset manifests and split definitions for the first larger nominal and abnormal datasets.
2. Re-run real-batch calibration on those datasets and save explicit false-positive analysis.
3. Produce per-dataset real-data evaluation artifacts and baseline-only versus fused comparisons.
4. Run one larger workflow with the WMM backend and record runtime and bottlenecks.
5. Add the highest-value missing tests around config parsing, inference branches, and WMM fallback behavior.

## Deferred Until Needed

- native desktop review shell
- new connector families without an actual source requirement
- production service packaging before there is an operational deployment target
- denser global grids before the current `2 deg x 2 deg` product is shown to be insufficient
