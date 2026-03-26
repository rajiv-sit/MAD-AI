# TODO List

This file tracks the work required after the prototype milestone.

The repo already has a working prototype. The remaining work is mostly about:

- proving behavior on real data
- making the WMM-backed path practical at scale
- clarifying what is already done versus what is still only a next step

## How To Read This File

Status meanings:

- `[done]` implemented and reflected in the repo today
- `[active]` partially implemented or currently being hardened
- `[next]` recommended next execution items
- `[blocked]` needs real data, labels, or another dependency before it can be completed
- `[later]` intentionally deferred

## Current Repo Status

### Already Done

- `[done]` global NOAA/WMM baseline generation and Cesium globe review flow
- `[done]` observed CSV anomaly scoring and anomaly-aware globe generation
- `[done]` real-batch folder ingestion with CSV, Parquet, JSONL, and SQLite support
- `[done]` dataset manifests for the real-batch workflow
- `[done]` baseline-only and fused anomaly scoring comparison in the real-batch path
- `[done]` Bahamas `.asc` real-data ingestion with aircraft state, vessel state, and aircraft-borne magnetometer readings
- `[done]` Bahamas NOAA-only / NOAA-plus-anomaly combined globe
- `[done]` Bahamas linked four-view dashboard:
  strip chart, vessel range comparison, synchronized globe, and along-track residual profile
- `[done]` docs updated to explain that the current Bahamas workflow is anomaly review with known vessel reference
- `[done]` first-pass magnetic vessel tracking package:
  vessel state types, dipole forward model, initialization logic, single-track and multi-hypothesis trackers
- `[done]` Bahamas tracking evaluation artifacts and estimated-versus-true vessel comparison views
- `[done]` globe and dashboard overlays for:
  aircraft track, true vessel track, and estimated magnetic vessel track

### Important Current Limitations

- `[active]` the full Bahamas workflow is practical with the `analytic` backend, but not yet practical end to end with full-file `WMM`
- `[active]` the Bahamas vessel path shown in the viewer comes from the dataset ship-navigation fields
- `[active]` anomaly is computed from aircraft magnetometer residuals against the baseline
- `[active]` vessel tracking is now estimated from magnetic measurements and aircraft geometry, but only with a first-pass dipole approximation
- `[active]` current magnetic tracking accuracy is still weak:
  Bahamas mean tracking error is still roughly `13 km`
- `[blocked]` real abnormal labels or confirmed anomaly windows are still limited

### Highest-Priority Next Work

1. `[next]` replace bundled example evidence with larger real nominal and abnormal datasets plus documented split definitions
2. `[next]` calibrate thresholds on larger real nominal data and save explicit false-positive analysis
3. `[next]` produce reproducible real-data evaluation artifacts and baseline-only versus fused comparisons
4. `[next]` improve magnetic vessel tracking quality beyond the current first-pass implementation
5. `[next]` make at least one larger real workflow practical with the `WMM` backend

## Working Rules

- Do not add new scope until a real dataset or review workflow requires it.
- Prefer work that improves evidence quality over work that only improves feature count.
- Keep the NOAA/WMM-backed path as the reference baseline.
- Treat the analytic backend as a speed tool for scaling checks, tests, and selected tuning runs.
- Update `README.md`, `docs/architecture.md`, and `docs/mad-ai-milestone.md` whenever a major item here changes state.

## Priority Roadmap

### 1. Real Data Scale And Provenance `[active]`

Why this matters:

- most remaining credibility work depends on real datasets and clean split definitions

Current state:

- `[done]` the repo has a Bahamas real-data workflow
- `[done]` the repo has real-batch manifests and config-driven dataset handling
- `[blocked]` the repo still does not have a complete larger nominal plus abnormal evaluation corpus with documented split definitions

Remaining work:

- [ ] Replace bundled synthetic or sample real-batch inputs with at least one larger nominal dataset and one abnormal dataset.
- [ ] Separate nominal, abnormal, and uncertain segments using documented split definitions.
- [ ] Add per-dataset manifests with source, time span, platform, altitude range, units, coordinate convention, and label quality.
- [ ] Record data quality issues such as missing timestamps, bad coordinates, duplicate records, unit mismatches, and sparse coverage.
- [ ] Store any dataset-specific schema mapping alongside the dataset manifest or config.

Done so far:

- Bahamas `.asc` ingestion is implemented
- real-batch manifests are implemented
- bundled real-batch manifests now include explicit split paths, inline schema mappings, coordinate conventions, units, and documented data quality issues

Exit criteria:

- At least one larger nominal dataset and one abnormal dataset run end to end through the pipeline.
- Provenance, split logic, and known data quality limits are documented.

Dependencies:

- none; this is the starting gate for most remaining work

### 2. Calibration Credibility `[active]`

Why this matters:

- thresholds are only credible if they come from real nominal distributions

Current state:

- `[done]` threshold calibration exists in the repo
- `[done]` calibration artifacts are saved for current example workflows
- `[blocked]` calibration is still too dependent on bundled examples and limited real nominal evidence

Remaining work:

- [ ] Calibrate thresholds on larger real nominal datasets rather than only bundled examples.
- [ ] Compare global thresholds against segmented thresholds by platform, altitude band, or mission type where those groupings are real.
- [ ] Save calibration reports with percentile sweeps, score distributions, and false-positive analysis.
- [ ] Add explicit review notes for nominal false positives and abnormal true positives.
- [ ] Document which threshold set is the operational default for each dataset family.

Exit criteria:

- Thresholds are justified by real nominal distributions.
- Calibration artifacts exist for the intended operational grouping strategy.

Dependencies:

- real dataset manifests and split definitions

### 3. Evaluation Quality `[active]`

Why this matters:

- the repo needs reproducible evidence on real abnormal or confirmed anomaly segments

Current state:

- `[done]` evaluation scripts and artifacts exist
- `[done]` baseline-only versus fused comparison exists in the real-batch path
- `[blocked]` evaluation still needs stronger real abnormal labels or confirmed anomaly windows

Remaining work:

- [ ] Evaluate on real abnormal segments or externally confirmed anomaly windows.
- [ ] Produce confusion metrics per dataset and per operational segment.
- [ ] Save ROC-like or precision-recall style analysis where labels are reliable enough.
- [ ] Compare baseline-only residual scoring against fused model scoring on the same real data.
- [ ] Document where labels are weak, incomplete, or only approximate.

Exit criteria:

- Real-data evaluation artifacts are reproducible from configs and scripts.
- Model versus baseline-only behavior is documented on real scored data.

Dependencies:

- real dataset availability
- calibrated thresholds

### 4. Magnetic Tracking From Measurements `[active]`

Why this matters:

- the current Bahamas workflow visualizes anomaly with a known vessel reference path
- the repo now has a first-pass inverse tracking implementation, but it is not yet accurate enough to treat as operational tracking

Current state:

- `[done]` aircraft magnetometer residuals are scored
- `[done]` the viewer shows the provided vessel-reference path
- `[done]` vessel state types, forward-model interfaces, and initialization logic exist
- `[done]` single-track and multi-hypothesis magnetic vessel trackers exist
- `[done]` estimated vessel tracks can be compared against the provided reference vessel track
- `[done]` the dashboard and Cesium globe now show the estimated magnetic vessel track against the true vessel track
- `[active]` the current tracker is a first-pass dipole-based inverse tracker with weak full-flight accuracy
- `[active]` the current evaluation result is still only about `13 km` mean tracking error on the Bahamas run

Done so far:

- explicit vessel state model and tracking observation types
- dipole magnetic forward model plus geometry-aware variant
- magnetic-bearing-grid initialization from anomaly geometry
- constant-velocity magnetic tracker
- multi-hypothesis magnetic tracker
- estimated-versus-reference vessel evaluation artifacts
- linked visual comparison of true vessel track versus estimated magnetic track

Remaining work:

- [ ] Improve the magnetic forward model beyond the current scalar dipole approximation.
- [ ] Add stronger relative-geometry constraints from aircraft motion and closest-approach structure.
- [ ] Estimate vessel heading and speed more robustly instead of relying on simple propagation.
- [ ] Score and compare tracking quality by segment, not only whole-flight mean error.
- [ ] Identify where tracking confidence is strong, weak, or ambiguous and surface that in artifacts.
- [ ] Test whether WMM-backed baseline features materially improve inverse tracking quality.
- [ ] Compare against stronger estimator families if the current beam tracker plateaus:
  EKF, UKF, particle filter, or more structured multi-hypothesis tracking.

Exit criteria:

- The repo can produce a vessel-track estimate from magnetic measurements and aircraft state without directly consuming the reference vessel track as the track solution.
- Estimated vessel-track quality is evaluated against the provided ship-reference path.
- Tracking error is materially below the current first-pass result and is good enough to support the intended review story.

Dependencies:

- real datasets with aircraft state, magnetometer observations, and reference vessel track for validation
- a usable WMM-backed or otherwise defensible baseline path

### 5. Magnetic Backend Fidelity At Scale `[active]`

Why this matters:

- the physically meaningful path is `WMM`, not only `analytic`

Current state:

- `[done]` both `WMM` and `analytic` backends exist
- `[done]` the repo documents `WMM` as the reference baseline path
- `[blocked]` full-file Bahamas and larger runs are still too slow with the current `WMM` path

Remaining work:

- [ ] Run the larger workflows with the NOAA/WMM-backed baseline, not only the analytic backend.
- [ ] Benchmark WMM runtime on larger observed, real-batch, and global workflows.
- [ ] Profile WMM-heavy preprocessing to identify the main cost centers.
- [ ] Optimize only the bottlenecks that materially affect operational use.
- [ ] Document when `analytic` is acceptable and when `WMM` is required.

Exit criteria:

- A larger real-data workflow completes successfully with the WMM backend.
- Runtime and resource costs are measured and documented.

Dependencies:

- real larger datasets

### 6. Model Quality `[active]`

Current state:

- `[done]` CNN and LSTM anomaly baselines exist
- `[done]` tuning support exists for the larger real-batch path
- `[done]` the tuning workflow now searches core model hyperparameters:
  window size, sequence length, fusion weighting, latent channels, hidden size, dropout, learning rate, and batch size
- `[done]` tuning now saves a reusable best-config artifact in addition to the leaderboard
- `[active]` stronger tuning and real-data comparison are still needed

Remaining work:

- [ ] Tune the CNN architecture beyond the current baseline autoencoder if the broader hyperparameter search still plateaus.
- [ ] Tune the LSTM architecture beyond the current baseline autoencoder if the broader hyperparameter search still plateaus.
- [ ] Run the expanded structured experiments on real datasets that matter operationally, not only the bundled large synthetic batch.
- [ ] Compare alternative anomaly strategies only if the current baselines plateau on real evaluation data.
- [ ] Track best-performing configurations per dataset family with reproducible config files and saved metrics.

Exit criteria:

- Model changes are judged by saved metrics, not visual anecdotes.
- A documented best configuration exists for each target dataset family that matters.

Dependencies:

- real-data evaluation loop
- credible thresholding

### 7. Visualizer Hardening `[active]`

Current state:

- `[done]` the Cesium globe is usable for NOAA, observed, real-batch, and Bahamas review
- `[done]` the Bahamas dashboard is linked to the globe
- `[active]` the review UX still needs clearer analyst-facing explanations and larger-dataset hardening

Remaining work:

- [ ] Validate the viewer on larger real scored datasets and record where it slows down or becomes confusing.
- [ ] Improve the review UX only where analyst workflows actually struggle.
- [ ] Add stronger export or report flows if analyst handoff requires them.
- [ ] Add clearer legends, threshold explanations, and overlay help text.
- [ ] Keep the linked Bahamas four-view dashboard aligned with the Cesium globe as the real-data review path evolves.
- [ ] Make current-time synchronization and anomaly-color interpretation clearer for analyst review.
- [ ] Consider a native desktop shell only if the browser flow becomes a real blocker.

Exit criteria:

- Reviewers can inspect larger real datasets without major usability failures.
- Exported outputs support the intended review workflow.

Dependencies:

- larger scored real datasets

### 8. Coverage And Test Hardening `[active]`

Current state:

- `[done]` CI exists
- `[done]` unit coverage is around `85%`
- `[active]` a few important inference, config, and WMM fallback areas still need better protection

Remaining work:

- [ ] Raise unit-test coverage above the current `85%`.
- [ ] Add targeted tests for config parsing branches, inference helper branches, abstract base class behavior, and WMM fallback branches.
- [ ] Add script-level regression tests where runtime is reasonable.
- [ ] Keep CI stable as workflow coverage expands.
- [ ] Add regression checks for the highest-value scored-artifact paths before broadening to lower-value scripts.

Exit criteria:

- Coverage reaches the next agreed target.
- CI protects the critical workflows without becoming too slow to use.

Dependencies:

- none for unit tests
- selected stable workflows for regression coverage

### 9. Source Connectors `[later]`

Current state:

- `[done]` file-based CSV, Parquet, JSONL, and SQLite support exists
- `[later]` new connectors should only be added for real source requirements

Remaining work:

- [ ] Add connectors only where actual deployment needs them.
- [ ] Candidate families include API ingestion, database-backed ingestion, message streams, and live sensor feeds.
- [ ] Add schema mapping templates and validation rules per real source family.
- [ ] Validate units, coordinate systems, and timestamp conventions for each added connector.

### 10. Global Baseline Scaling `[later]`

Current state:

- `[done]` current globe runs from a `2 deg x 2 deg` multi-altitude global product
- `[later]` denser global grids should only happen if the current resolution is clearly insufficient

Remaining work:

- [ ] Decide whether denser than `2 deg x 2 deg` global grids are actually worth the cost.
- [ ] If justified, optimize for `1 deg` or sub-`2 deg` NOAA/WMM global grids.
- [ ] Add chunked or parallel global build paths only if they materially help.
- [ ] Benchmark overlay generation time, artifact size, and viewer responsiveness at higher resolutions.

### 11. Production Hardening `[later]`

Current state:

- `[later]` this is still a research prototype unless an operational deployment decision is made

Remaining work:

- [ ] Decide explicitly whether the project remains a research prototype or becomes an operational service.
- [ ] If operationalized, add packaging and deployment strategy.
- [ ] Add environment and config management.
- [ ] Add logging, monitoring, failure handling, and recovery behavior.
- [ ] Define artifact retention, versioning, and access control for review tools.

## Documentation Follow-Up `[active]`

Current state:

- `[done]` `README.md`, `docs/architecture.md`, and `docs/mad-ai-milestone.md` were updated to reflect the Bahamas workflow and current tracking limitation

Remaining work:

- [ ] Keep `README.md`, `docs/architecture.md`, and `docs/mad-ai-milestone.md` aligned with the current state.
- [ ] Add an experiment or results summary if tuning work becomes deeper.
- [ ] Add data-governance notes before bringing in real operational datasets.
- [ ] Keep the docs explicit that the current Bahamas workflow is anomaly detection with known vessel reference, not magnetometer-only vessel tracking.

## Next Execution Sprint `[next]`

Recommended near-term sprint:

1. Land dataset manifests and split definitions for the first larger nominal and abnormal datasets.
2. Re-run real-batch calibration on those datasets and save explicit false-positive analysis.
3. Produce per-dataset real-data evaluation artifacts and baseline-only versus fused comparisons.
4. Define the first vessel state model and magnetic forward-model interface for inverse tracking.
5. Improve the first-pass magnetic tracker and reduce the current Bahamas tracking error materially below the present `~13 km` mean.
6. Run one larger workflow with the WMM backend and record runtime and bottlenecks.
7. Add the highest-value missing tests around config parsing, inference branches, and WMM fallback behavior.

## Deferred Until Needed `[later]`

- native desktop review shell
- new connector families without an actual source requirement
- production service packaging before there is an operational deployment target
- denser global grids before the current `2 deg x 2 deg` product is shown to be insufficient
