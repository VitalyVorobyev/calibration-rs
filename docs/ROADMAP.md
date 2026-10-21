# calibration-rs Roadmap

Where the project is going and in what order. Design reasoning lives in
[ADRs](adrs/README.md), tasks in the [backlog](backlog.md), released changes
in the [CHANGELOG](../CHANGELOG.md).

## Goal

The leading calibration library for Rust and Python. It distinguishes
itself in four ways:

- every result carries an honest uncertainty, from the measured feature
  noise to the projection and the 3D points derived from it;
- quality gates turn that uncertainty into pass/warn/fail verdicts;
- multiple-view geometry is built on the same uncertainty, so a
  measurement from a calibrated system comes with error bars;
- the same capabilities are available in Rust and Python, explained in an
  interactive book, and usable without code in the desktop app.

## State

Pre-1.0 (0.x): breaking changes are allowed, each listed in the CHANGELOG
with migration notes. The published packages are listed in AGENTS.md.

## Phases

The program runs in two phases. **Phase 1** holds every breaking change
and ends with v1.0. **Phase 2** adds capability in 1.x without breaking
the API. The book and app tracks are not bound by semver, so they run
alongside both phases. Track names below match the backlog headings.

### Phase 1 — Foundation, then v1.0

| Track | Goal | Gate |
|---|---|---|
| API | One configuration vocabulary. Every problem type takes the same camera-model spec (projection, distortion, sensor), and factors and bundles are generic over it. One naming scheme for inputs, outputs and step functions, identical in Rust and Python. | Every problem accepts every distortion and sensor model; the facade compile-surface test and the binding parity check pass |
| UQ | Uncertainty as a product: per-feature covariance from the detectors, a variance-factor test of the noise model, joint covariance for rigs, and a quality report with verdicts | Predicted scatter matches Monte-Carlo and rendered-image scatter (proof pack); `calib-bench accept` gates on the quality report |
| PY | Python parity: sessions and steps, manifest-driven runs, detectors, camera models with vectorized projection, quality report; types generated from the same JSON schemas as the app | The parity map in CI lists no unmapped facade item |
| Data | The public regression suite fetches its images from their publishers | `calib-bench suite fetch` plus a scheduled freeze check |

### v1.0 exit criteria

1. **Acceptance** — `calib-bench accept` passes every on-disk registered
   dataset through the seeded official route, gated by the quality report.
2. **Public regression suite** — a versioned public suite (observations in
   the repository, images fetched from their publishers and verified by
   checksum) gates every PR, with reference results compared against OpenCV.
3. **Uncertainty** — every result carries its parameter covariance,
   projection uncertainty and quality report. Rig results include the
   intrinsics. Covariance is computed from per-feature detector noise and
   validated against Monte-Carlo and rendered-image scatter.
4. **Soundness** — every shipped algorithm family has a proof pack
   ([`notes/`](notes/README.md)): math note, synthetic-GT matrix test,
   property tests where an invariant exists, and a committed regression
   baseline with a drift gate.
5. **Frozen API** — facade-only consumers; one camera-model spec across all
   problem types; normalized configs
   ([ADR 0024](adrs/0024-api-vocabulary.md)); typed errors; no duplicate
   public type names and no deprecated shims; the API unchanged across two
   consecutive minor releases.
6. **Python parity** — every facade capability is in the Python package,
   enforced in CI.
7. **App CI green** — lint (zero warnings), format, typecheck, unit,
   component and smoke tests, generated IPC types in sync.
8. **Docs current** — README, crate READMEs, the book, ADRs and this file
   describe the code as it is, and user-facing docs carry no internal
   references (`cargo xtask check-docs`).

### Phase 2 — Growth in 1.x

| Track | Goal | Gate |
|---|---|---|
| CAM | Kannala-Brandt fisheye, EUCM, double sphere and telecentric projection in every problem type, plus model selection | Per model: math note, synthetic matrix test, Python, book chapter; OpenCV comparison for fisheye |
| CAL | Rigs whose cameras do not all see the reference camera, multi-target scenes, more hand-eye initializers, robot-pose and target-flatness uncertainty, cross-frame projection uncertainty | Synthetic-GT matrix per feature |
| MVG | Metrology-grade geometry: observations with covariance, triangulated points and derived measurements (lengths, angles, planes) with σ that includes the calibration's covariance, BA on the optimization IR with covariance, stereo from a rig export to depth with σ | Monte-Carlo agreement for every propagated σ; comparison against PoseLib/pycolmap |
| IO | Interchange formats (OpenCV, ROS, Kalibr, COLMAP, mrcal) and undistort/rectify maps; a user CLI that runs a manifest end to end and exits by the quality verdict | Round-trip tests per format; CLI integration tests |
| PERF | Released solver speed-ups, analytic reprojection Jacobians, published timings | `calib-bench solver` against committed numbers |

### Alongside both phases

| Track | Goal | Gate |
|---|---|---|
| Book | One user guide: tutorials merged in; Rust/Python tabs; chapters on detection, the manifest and its schema reference, uncertainty, MVG, interop and the app; interactive demos running the library in the browser through a wasm build | `check-docs`; every demo runs in the published book |
| App | Rebuilt frontend on the `@vitavision/workbench` shell, organized as Project → Dataset → Detect → Calibrate → Diagnose → Use. Saved runs, reports, solver progress with cancel, installers that run without a source checkout | App gates (AGENTS.md §10) plus a frontend review with no high-severity findings |

## Ordering

Phase 1 starts with the camera-model spec. It is the one design every
later track depends on: new models, rig covariance, and the Python and
MVG surfaces. Then:

1. The result-changing UQ fixes.
2. Per-feature covariance.
3. The quality report.
4. Python parity.

Phase 2 tracks are independent of each other and are picked by need.

## Standing decisions

The binding design decisions are the [ADRs](adrs/README.md). Policies that
are not design:

- The default solver backend changes only on a measured comparison and the
  maintainer's decision.
- Cross-platform only: the library, its comparison harnesses and its CI
  jobs run on Linux, macOS and Windows. Linux-only tools (mrcal, Kalibr)
  are compared through their published results, not run in the suite.
- A change that alters results is measured first, opt-in until the
  evidence is in, and lands with a migration note.

## Out of scope

Each item states what would bring it in.

- **Splined (non-parametric) camera models**: when a dataset shows the
  parametric models leave structured residuals above the detector noise.
- **Camera–IMU, rolling shutter and time-offset calibration**: when a
  project needs it. It is a new estimation domain (continuous-time
  trajectories), not an extension of the current problem types.
- **Full structure-from-motion** (incremental SfM, pose graphs, loop
  closure; ADR 0006): no trigger. That belongs in a dedicated SfM library.
- **`rerun.io` / `egui` as the UI.**
- **Live camera capture in the app**: gated on the maintainer's go
  (backlog APP-LIVE).
- **A community-grade contribution surface and multi-platform install
  docs.**
