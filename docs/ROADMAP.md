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

Phase 1b is done and the v1.0 exit criteria hold (swept 2026-11-12):
(1) acceptance through the seeded route; (2) the public regression suite
gates every PR; (3) every result carries its covariance, checked by
Monte-Carlo and real split-half scatter; (4) a proof pack per algorithm
family; (5) a frozen facade with no break through the rc series; (6) Python
parity; (7) measurements with σ on public real data; (8) a user residual and
pipeline from the facade alone; (9) onboarding from a crates.io / PyPI
install; (10) app CI green; (11) current docs; (12) only `nalgebra` in facade
signatures (AGENTS.md §7).

The workspace version is `1.0.0-rc.1`; the tag is pending (backlog D4-RELEASE).
The release is a `1.0.0-rc` series and 1.0.0 follows an rc that needed no
break; the version rules are AGENTS.md §8, the published packages
[`RELEASE-RUNBOOK.md`](RELEASE-RUNBOOK.md) §2.

## Phases

**Phase 1** (0.12) built the foundation: one camera-model spec, parameter
uncertainty matched to real scatter, the quality report, Python parity, the
public regression suite. **Phase 1b** (to 1.0) made the library usable from
an install, extensible through a public problem builder, measured (MVG with
σ, the CLI) and held every breaking change; the CHANGELOG has the detail.
**Phase 2** adds capability in 1.x without breaking the API. The tracks
marked *alongside* are not bound by semver and run in both phases. Track
names below are the backlog headings.

### Phase 2 — Growth in 1.x

| Track | Goal | Gate |
|---|---|---|
| Parameter uncertainty | Cross-frame projection uncertainty, robust-loss and composed-covariance edge cases, the robot-prior default | Monte-Carlo agreement for every new σ |
| Camera models | EUCM, double sphere and telecentric projection in every problem type, plus model selection; a camera-fixed residual field once one is shown to exist | Per model: math note, synthetic matrix test, Python, book chapter; OpenCV comparison for fisheye |
| Calibration problems | Scheimpflug tilt determined only when the data determine it (with priors, a systematics safeguard and a named quality message), a focus-tied tilt, non-ideal target geometry, a sturdier tilt seed and basin, multi-target scenes, report sections for hand-eye and laser | Synthetic-GT matrix per feature; real-data gates where an entry names one |
| Multiple-view geometry | Two-view robustness and uncertainty (optimal triangulation, F/H covariance, GRIC, LO-RANSAC, P3P/EPnP in RANSAC) | Monte-Carlo agreement |
| Solver backends and performance | A σ-based stopping rule, upstreamed solver work, analytic reprojection Jacobians, external comparisons (PoseLib, pycolmap, Ceres), published timings | `calib-bench solver` against committed numbers; published comparison |

### Alongside both phases

| Track | Goal | Gate |
|---|---|---|
| Datasets and the regression suite | View identity in every export, more licensed public entries | `calib-bench suite check` on both backends |
| Bindings and internals | One source per type and per helper: Python types generated from the Rust schemas, MVG results with a schema, the fixed problems built through the public builder | No hand-written copy of a Rust type in Python; results bit-identical or measured |
| Book | The one user guide; next, a results page generated from the regression suite and interactive demos running the library in the browser through a wasm build (MVG σ, planar σ, the camera models) | `check-docs`; every demo runs in the published book |
| App | The desktop app and its browser report viewer (`calib-viewer`), organized as Project → Dataset → Detect → Calibrate → Diagnose → Use; next, the report viewer grows an error map, a quality panel, run comparison and feedback notes, and the app's run dispatch and wire types come from the library | App gates (AGENTS.md §12) plus a frontend review with no high-severity findings |

## Ordering

Phase 2 tracks are independent of each other and are picked by need. The
browser demos follow the wasm build.

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
onboarding from an install and the CLI, a published comparison against OpenCV
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
- A change that alters results follows AGENTS.md §5 (measured first,
  opt-in until the evidence is in).

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
