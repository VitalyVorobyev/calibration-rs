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
- the same capabilities are available in Rust, Python and a command-line
  tool, explained in an interactive book, and usable without code in the
  desktop app.

## State

Phase 1b is done and the v1.0 exit criteria hold (swept 2026-11-12). The
release is a `1.0.0-rc` series: the `semver-rc` CI job fails a pull request
that breaks the API of the latest rc, and 1.0.0 follows an rc that needed no
break. From 1.0 the version rule is standard SemVer (AGENTS.md §8). The
published packages are listed in AGENTS.md.

## Phases

**Phase 1** (done, 0.12) built the foundation. **Phase 1b** makes the
library clear to use, extensible and measured, and holds every remaining
breaking change; it ends with v1.0 (done, 1.0.0-rc.1). **Phase 2** adds capability in 1.x
without breaking the API. The book and app tracks are not bound by semver
and run alongside. Track names below match the backlog headings.

### Phase 1 — Foundation (done)

One camera-model spec across every problem type (`#[non_exhaustive]`
enums, Kannala-Brandt fisheye as the second projection, agreeing with
OpenCV's `fisheye` within 0.01σ); parameter uncertainty whose stated σ
matches real split-half scatter (`r` 1.03 pooled); the quality report;
Python parity enforced in CI; the public regression suite fetched from its
publishers; the API audit and the public-dependency policy.

### Phase 1b — Usable, extensible, measured (done)

The 0.12 review found a library that calibrates well but is not ready to
freeze: a first real-image run needs a source checkout, a user cannot add
a residual or compose a pipeline without forking, and multiple-view
geometry takes a bare `K`, carries no covariance and has no demonstration.

| Track | Goal | Gate |
|---|---|---|
| USE | A newcomer reaches a result on their own images from the book alone: a workflow chooser, a real-image quickstart, sample data fetched without a checkout, runnable docs.rs examples, capture guidance | Every quickstart and example runs from a crates.io / PyPI install; `check-docs` |
| CLI | `vcal`: manifest in, export and report out, plus `init`, `check` and sample fetch, as a thin layer over the dataset runner; one HTML report shared with the app | CLI integration tests on the public suite data; the report renders every workflow |
| EXT | Build your own pipeline: a public residual trait and IR builder with covariance over labelled blocks, the closed-form initializers on the facade, and a book chapter on each | A downstream test crate that uses only the facade defines a residual, solves and reads σ |
| MVG | Measurements with error bars: MVG takes calibrated cameras (any camera model, rig exports), observations carry covariance, triangulation, BA, stereo and a metrology API return σ that includes the calibration's covariance; BA and every σ through the public problem builder (ADR 0028) | Monte-Carlo agreement for every propagated σ; a real-data walk-through measuring known board lengths with predicted σ checked against the actual error; Python bindings |
| CORRECT | Rig initialization through a co-visibility graph, more hand-eye initializers, progress and cancellation through the solver | Synthetic-GT matrix per feature |
| LEAN | Shared step logic of the problem types factored once; MVG and geometry free of duplicate solvers | Library code lines down, results bit-identical |

### v1.0 exit criteria

1. **Acceptance** — `calib-bench accept` passes every on-disk registered
   dataset through the seeded official route, gated by the quality report.
2. **Public regression suite** — a versioned public suite (observations in
   the repository, images fetched from their publishers and verified by
   checksum) gates every PR, with reference results compared against OpenCV
   wherever OpenCV has an equivalent calibration.
3. **Uncertainty** — every result carries its parameter covariance,
   projection uncertainty and quality report, validated by Monte-Carlo and
   against real-image scatter (split-half consistency). The detector's
   per-corner covariance does not weight results by default
   (`notes/feature-covariance-study.md`).
4. **Soundness** — every shipped algorithm family has a proof pack
   ([`notes/`](notes/README.md)): math note, synthetic-GT matrix test,
   property tests where an invariant exists, and a committed regression
   baseline with a drift gate.
5. **Frozen API** — facade-only consumers; one camera-model spec across all
   problem types and MVG; normalized configs
   ([ADR 0024](adrs/0024-api-vocabulary.md)); typed errors; one path per
   public symbol (the per-problem seeds and step results share names across
   problem modules by design, ADR 0024) and no deprecated shims; no breaking
   change through a `1.0.0-rc` series (`semver-checks` against the last rc
   on every PR).
6. **Python parity** — every facade capability, MVG included, is in the
   Python package, enforced in CI.
7. **Measurements** — a measurement from a calibrated system (point,
   distance, depth) comes with σ that includes the calibration's
   covariance, demonstrated on public real data.
8. **Extensible** — a user residual and a composed pipeline work from the
   facade alone, exercised by MVG bundle adjustment.
9. **Onboarding** — the book's quickstarts, the examples and `vcal` work
   from a crates.io / PyPI install, without a source checkout.
10. **App CI green** — lint (zero warnings), format, typecheck, unit,
    component and smoke tests, generated IPC types in sync.
11. **Docs current** — README, crate READMEs, the book, ADRs and this file
    describe the code as it is, and user-facing docs carry no internal
    references (`cargo xtask check-docs`).
12. **Public dependencies** — only `nalgebra` reaches the facade's
    signatures; its version follows the solver backends, and a bump is a
    breaking release (AGENTS.md §7). 1.0 ships on nalgebra 0.34 and does not
    wait for the backends' move to 0.35; that bump is a 2.0 change
    (NALGEBRA-035).

### Phase 2 — Growth in 1.x

| Track | Goal | Gate |
|---|---|---|
| CAM | EUCM, double sphere and telecentric projection in every problem type, plus model selection | Per model: math note, synthetic matrix test, Python, book chapter; OpenCV comparison for fisheye |
| CAL | Multi-target scenes, robot-pose and target-flatness uncertainty, cross-frame projection uncertainty | Synthetic-GT matrix per feature |
| MVG | Two-view robustness and uncertainty (optimal triangulation, F/E/H covariance, GRIC, LO-RANSAC), comparison against PoseLib/pycolmap | Monte-Carlo agreement; published comparison |
| PERF | Released solver speed-ups, analytic reprojection Jacobians, published timings | `calib-bench solver` against committed numbers |

### Alongside both phases

| Track | Goal | Gate |
|---|---|---|
| Book | The one user guide; next, a results page generated from the regression suite and interactive demos running the library in the browser through a wasm build (MVG σ, planar σ, the camera models) | `check-docs`; every demo runs in the published book |
| App | Rebuilt frontend on the `@vitavision/workbench` shell, organized as Project → Dataset → Detect → Calibrate → Diagnose → Use. Saved runs, reports, solver progress with cancel, installers that run without a source checkout | App gates (AGENTS.md §12) plus a frontend review with no high-severity findings |
| Bindings and internals | One source per type and per helper: Python types generated from the Rust schemas, MVG results with a schema, the fixed problems built through the public builder | No hand-written copy of a Rust type in Python; results bit-identical or measured |

## Ordering

Phase 1b ran USE, the CLI, EXT, the MVG chain, CORRECT and LEAN, then a
pre-release review that froze the wire contract and the facade. The
release is a `1.0.0-rc` series. The browser demos follow MVG σ, since they
show it.

Phase 2 tracks are independent of each other and are picked by need.

## Landscape

Rust calibration crates on crates.io, as of 2026-10-21. No other crate
combines several calibration problems with parameter covariance; hand-eye,
laser-plane and Scheimpflug calibration exist nowhere else in Rust.

| Crate | Licence | Scope | Relation to us |
|---|---|---|---|
| `camera-intrinsic-calibration` | GPL-3.0 | Intrinsics and multi-camera extrinsics; EUCM, UCM, KB4, F-theta; a CLI with Rerun | Closest in scope, about 30× our downloads. No covariance, no hand-eye. GPL, so a comparison baseline only (BENCH-COMPARE), never a code source |
| `camera-intrinsic-model` | Apache-2.0 | Camera models, undistort and remap, stereo rectification | Models only. Reading its formats would fall under the interchange-format item in [Out of scope](#out-of-scope) |
| `apex-solver`, `apex-camera-models` | Apache-2.0 | Least squares, bundle adjustment refining intrinsics, 9 camera models | New and fast-moving; no calibration pipelines, no covariance. The one to watch |
| `opencv` | MIT | Bindings to all of OpenCV | Needs a native OpenCV build; covariance limited to standard deviations |
| `mrcal` | — | Bindings to the C mrcal, which has uncertainty | Needs a native build, Linux-only, practically unused |
| `sophus`, `rust-cv`, `kornia-rs`, `cam-geom`, `sqpnp`, `factrs` | permissive | Lie groups, camera geometry, PnP, solvers | No calibration; `rust-cv` inactive since 2020; `factrs` is one of our backends |

The gap to the closest rival is adoption, not capability. The levers are
Phase 1b's onboarding and CLI, a published comparison against OpenCV
(BOOK-RESULTS), and measurements with error bars, which no rival offers.
Revisit the table when a crate adds calibration pipelines or covariance.

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

Each item states why it is out or what would bring it in.

- **Splined (non-parametric) camera models**: when a dataset shows the
  parametric models leave structured residuals above the detector noise.
- **Camera–IMU, rolling shutter and time-offset calibration**: when a
  project needs it. It is a new estimation domain (continuous-time
  trajectories), not an extension of the current problem types.
- **Full structure-from-motion** (incremental SfM, pose graphs, loop
  closure; ADR 0006): no trigger. That belongs in a dedicated SfM library.
- **`rerun.io` / `egui` as the UI.**
- **Live camera capture in the app**: the app works on recorded datasets.
- **Interchange formats and remap maps** (OpenCV, ROS, Kalibr, COLMAP,
  mrcal files; undistort/rectify lookup tables): exports are JSON with a
  published schema, and image remapping belongs to the image-processing
  library that consumes the calibration.
- **A community-grade contribution surface and multi-platform install
  docs.**
