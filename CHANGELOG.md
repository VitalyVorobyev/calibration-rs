# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.0] - 2026-08-16

Upstream detector-library migration: `calib-targets` 0.9 → 0.12.1,
`chess-corners` 0.11 → 1.x, `ringgrid` 0.7 → 0.11. Each carried breaking
API changes, and one carried a behavioural change that moves calibration
numbers — hence a minor bump rather than a patch.

### Changed (breaking, pre-1.0)

- **Problem configs reject unknown fields.** All sixteen pipeline `*Config`
  types (the eight problem configs and the grouped sub-structs they share)
  now carry `#[serde(deny_unknown_fields)]`; the generated JSON Schemas say
  `additionalProperties: false` to match. Previously a mistyped key was
  silently dropped and the field fell back to its Rust default — a wrong
  calibration with no error, which the Python test suite had been guarding
  against by hand. **Migration:** fix the key. Any config that was already
  correct is unaffected.
- **`CameraParams::build` returns `Result`.** It could panic on a
  non-invertible sensor homography, and `CameraParams` deserializes straight
  from an export file — so a hand-edited or corrupted export aborted the
  process instead of returning an error.
  `PlanarIntrinsicsParams::build_camera` follows.
- **`vision-calibration-optim` no longer exposes the optimization IR.**
  `ProblemIR`, `ResidualBlock`, `FactorKind`, `ParamSlotSpec`,
  `ManifoldKind`, `CameraModelDesc`, `LaserChain`, `ReprojChain`,
  `SensorKind`, `ProjectionKind`, `FixedMask`, `BackendKind`,
  `BackendSolution`, `solve_with_backend`, and the `pack_*`/`unpack_*`
  helpers are now private — 22 items with no consumer anywhere, which
  publishing would have frozen into the semver contract. They were also a
  hazard: the factor kernels assert invariants only this crate's builders
  establish, so a hand-built IR panicked inside the solver loop. Callers use
  the `optimize_*` entry points.
- **`vision-geometry` items are reached through their module.** The crate
  root globbed five submodules, giving every solver two public paths and
  silently widening the root whenever a submodule gained a `pub fn`. Use
  `vision_geometry::epipolar::fundamental_8point` rather than
  `vision_geometry::fundamental_8point`; `GeometryError` and `Result` stay at
  the root. Every in-tree caller already used the module path.
- **`ReprojLevel::Laser` is removed.** It was never constructed by any
  builder — a public enum variant, and therefore a semver commitment, to a
  feature that does not exist.
- **`threshold_mode` is gone from the detector-override vocabulary.**
  `chess-corners` 1.0 collapsed its `Threshold::{Absolute, Relative}` enum
  into a single absolute `f32`; the `"relative"` mode (a fraction of the
  image maximum response) has no successor. `ChessThresholdMode` is
  deleted from `vision-calibration-detect` and `vision-calibration-dataset`,
  along with the bench registry's parallel `BenchChessThresholdMode` /
  `ChessCornersDetectorOverride`. `ChessCornersDetectorSpec` keeps only
  `threshold_value`, now documented as an absolute floor on the raw ChESS
  response. **Migration:** delete the `threshold_mode` key from manifests
  and bench registries; `deny_unknown_fields` rejects it rather than
  ignoring it. `"absolute"` values carry over unchanged — the workspace
  default is still `15.0`. `"relative"` values have no equivalent and must
  be re-tuned as absolute thresholds.
- **The bench registry's ChESS override is now
  `vision_calibration_dataset::ChessCornersDetectorSpec`** (re-exported
  from `vision_calibration_bench::registry`) instead of a third
  redeclaration of the same two fields.
- **Detector behaviour: `min_corner_strength` now defaults to `33.0`**
  (was `0.0`) via `calib-targets` 0.12, dropping weak, defocused corners
  before the grid builder. `GraphBuildAlgorithm` is gone — `calib-targets`
  collapsed its two grid builders into one, so the former `Topological`
  opt-in is the only path. Chessboard and ChArUco detections change
  accordingly; committed Fit baselines were re-run.
- **`vision-calibration-detect` no longer depends on `chess-corners`**, and
  `calib-targets` is taken with `default-features = false, features =
  ["image"]` — its default set pulled `clap` and a CLI binary into every
  consumer's dependency tree.

### Added

- `DetectError::Backend { detector, message }` in
  `vision-calibration-detect`, for detector-backend failures that are not
  "no target in this frame". Previously the ringgrid wrapper flattened
  every backend error into an empty feature list, so a misconfigured board
  surfaced much later as an unexplained initialisation failure.
- **Facade exports for three error payloads and the detection-cache item
  type.** `vision_calibration::Error` has `Core`/`Linear`/`Optim` variants
  whose payload types the facade never exported, so a facade-only consumer
  could match a variant but not inspect it; they are now `core::Error`,
  `linear::Error` and `optim::Error`. Likewise
  `detect::{Feature, DetectError, reject_ambiguous_detection}` — without
  `Feature`, the exported `DetectionCache` trait could not be implemented
  through the facade at all.
- `reject_ambiguous_detection` in `vision-calibration-detect`, applied by
  all four detectors. A planar target's image-to-board map is a homography
  and so injective; a detection that labels one pixel as several board
  points has mislabelled its grid, and the whole view is rejected (reported
  as a skipped frame) rather than contributing a handful of correspondences
  with no consistent pose.

### Fixed

- **ChArUco and chessboard corner labels are projectively consistent again**,
  via the `calib-targets` 0.12.1 floor. 0.12.0's grid builder could accept a
  mirrored symmetry element when merging lattice components on a thin
  overlap, gluing a sub-block in with a reversed axis — plausible edge
  lengths and pixel spacings, but a labelling no homography fits. It could
  also re-label one physical corner at a run of lattice coordinates. Both
  survive a corner-count check while making the view's pose meaningless. On a
  six-camera ChArUco rig with 720×540 tiles the worst affected camera sat at
  10.9 px mean reprojection under 0.12.0. **Anyone who ran 0.12.0 through
  `vision-calibration` 0.8.0-rc builds should re-run detection** — the
  detection cache keys on image content and detector parameters, not on the
  detector's version, so stale entries will be reused. Delete the cache
  directory to force re-detection.
- **`app/src/schemas/*.json` were being reformatted by Prettier**, drifting
  from `cargo xtask emit-schemas` output. The `.prettierignore` entry meant
  to protect them named `schemas-generated` (the *wire*-schema directory)
  and never matched `src/schemas`. Both are ignored now.
- **`cargo xtask emit-schemas --check` now runs in CI.** Only the
  `app/src-tauri` wire-schema drift check was wired up, so a change to any
  of the nine user-facing config types could silently leave the app's
  schema-driven forms stale.

### Internal

- `nalgebra`, `faer` and `faer-ext` stay at 0.34 / 0.23 / 0.7. They are
  pinned by `tiny-solver` 0.18, which `vision-calibration-optim` exchanges
  both nalgebra and faer types with; 0.35 / 0.24 / 0.8 do not compile until
  tiny-solver moves. Documented at the pin site.
- `criterion` 0.5 → 0.8; benches switched to `std::hint::black_box`
  (`criterion::black_box` is deprecated in 0.8).
- Removed the dead `num-dual` workspace-dependency entry — no crate
  declared it.
- `actions/checkout` v6 → v7 across all workflows.
- Deleted unused IR helpers, four byte-identical copies of
  `format_init_source`, and nineteen `*State` accessors that existed only to
  be called by their own unit tests.
- Float sorts in `vision-mvg`'s degeneracy/homography paths and
  `vision-calibration-linear`'s laser-plane fit use `total_cmp` instead of
  `partial_cmp(..).unwrap()`, which panicked on a NaN produced upstream by a
  divergent solve.
- `session::current_timestamp` returns `0` rather than panicking on a
  pre-1970 clock. It stamps every `session.export()`.
- The manifest's ChESS override is lowered to the detector's option type by
  an exhaustive destructuring rather than an untyped JSON round-trip, so a
  new manifest knob that is not wired through fails to compile.

## [0.7.0] - 2026-07-10

`0.7.0` reshapes the configuration vocabulary, adds the multiple-view
geometry surface (bundle adjustment, Scheimpflug-aware rectification, dense
stereo matching), introduces device-spec seeding, and finishes the
typed-error sweep across the remaining crates. It is the largest batch of
pre-1.0 breaking changes since `0.5.0`, collected here rather than dribbled
across patch releases.

### Changed (breaking, pre-1.0)

- **Shared, grouped configuration structs.** All eight problem configs move
  onto shared grouped structs — `IntrinsicsInitConfig` / `SolverConfig` /
  `RobotPoseConfig` / `HandeyeInitConfig`
  (`vision_calibration_pipeline::common::config`, facade re-exported) —
  and one fix-mask idiom (`CameraFixMask` / `ScheimpflugFixMask`) instead
  of boolean trios. `fix_intrinsics` + `fix_distortion` collapse into
  `CameraFixMask`; `fix_first_pose` becomes `fix_poses = [0]`.
  `fix_first_rig_pose` and `fix_first_camera_extrinsic` are **deleted** —
  `reference_camera_idx` is now the sole gauge choice (this also fixed an
  index-0 hard-coding bug; several committed baselines improved measurably,
  e.g. `stereo_charuco` 0.5509 → 0.5359 px mean reprojection).
  `RigExtrinsicsConfig`, `RigHandeyeConfig`, `LaserlineDeviceConfig`, and
  `RigLaserlineDeviceConfig` all reshape to the grouped form; five old
  `RigHandeye*` sub-structs are gone. `SensorMode::Scheimpflug`'s
  `fix_scheimpflug_in_intrinsics` field is renamed `fix_scheimpflug`. See
  [ADR 0024](docs/adrs/0024-config-vocabulary.md).
- **Public API cleanup.** Deleted the deprecated
  `pixel_to_gripper_point` shim and `LaserlinePlaneSolver::from_view`;
  merged the duplicate `ScheimpflugFixMask` definitions into the one optim
  type (pipeline re-exports it); consolidated residual/histogram
  diagnostics under `vision_calibration::analysis`. New facade modules
  (`dataset` / `dataset_runner` / `detect`) plus core `Mat3`,
  `distort_to_pixel`, `pixel_to_normalized`, `CameraModel` let consumers go
  facade-only.
- **Export `kind` discriminator is now required.** All eight
  `*Export` types gain a shared `ExportKind` tag (snake_case serde +
  JsonSchema) set at every construction site; deserialization is strict —
  an export JSON without `kind` is now a hard error instead of a
  best-effort field-shape guess. **Pre-0.7.0 export JSON must be
  regenerated** before the app or a saved fixture can load it again.
- **Two-view geometry solvers relocated.**
  `homography` / `epipolar` / `triangulation` / `camera_matrix` move from
  `vision_calibration::linear` to a new `vision_calibration::geometry`
  module backed by the `vision-geometry` crate;
  `vision-calibration-linear` now depends on `vision-geometry` and drops
  ~1,800 LoC of duplicated solvers. `vision-geometry` / `vision-mvg` gain
  typed `GeometryError` / `MvgError` (`thiserror`, `#[non_exhaustive]`),
  replacing `anyhow` on their public surface.
- **`vision-calibration-optim` / `-detect` / `-pipeline` / `-linear` are
  fully typed-error, `anyhow`-free on their library surfaces.**
  New/changed error variants across all four crates (e.g. optim gains
  `Error::numerical`, linear gains `NoValidMotionPairs`); no
  `vision-calibration*` library crate carries `anyhow` in
  `[dependencies]` any more (dev-only, for doctests).
- **`PlanarIntrinsicsParams.camera` generalized to the model-agnostic
  `CameraParams` enum.** Was the concrete Brown-Conrady
  `PinholeCamera`; `pinhole_camera()` now returns `Result` (errors for
  extended models). `PlanarIntrinsicsConfig`, `ScheimpflugIntrinsicsConfig`,
  and rig `SensorMode::Scheimpflug` gain `distortion_model: DistortionKind`
  (`#[serde(default)]` → Brown-Conrady5, so existing configs keep working)
  selecting between Brown-Conrady5, Rational8, ThinPrism9, and Division1.
- **Python bindings.** Result cameras are now polymorphic
  (`PinholeCamera` / `PinholeScheimpflugCamera` with a `Distortion` union)
  instead of one fixed dataclass shape; **`PinholeBrownConradyScheimpflugCamera`
  is renamed `PinholeScheimpflugCamera`.** `distortion_model` and
  `SensorMode` (`Pinhole` / `Scheimpflug`) are now mirrored on the
  relevant config dataclasses.

### Added

- **`vision-mvg::bundle_adjust`.** Frozen-intrinsics bundle
  adjustment refining camera poses + 3D structure by reprojection error,
  behind the `refine` feature; gauge fix via `fix_first_camera` (anchors
  the lowest-index *observed* camera).
- **`vision-mvg::rectification::rectify_stereo_pair`.**
  Scheimpflug-aware stereo rectification — sensor tilt collapses to a
  homography on the normalized plane (`H_tilt⁻¹`), so a tilted-sensor pair
  rectifies through standard Fusiello/Bouguet rectification and reduces
  exactly to pinhole rectification at zero tilt. Validated against the
  real Scheimpflug oracle dataset at 3.4e-13 px worst-case row
  disagreement.
- **`vision-mvg::dense`.** Pure-Rust dense stereo matcher: ZNCC block
  matching over summed-area tables with parabolic sub-pixel refinement and
  three invalidation filters (min-correlation, uniqueness, left-right
  consistency), plus opt-in Hirschmüller semi-global (SGM) cost
  aggregation. ADR 0015 amended: the matcher ships pure-Rust in
  `vision-mvg`; an OpenCV SGBM baseline is confined to the unpublished,
  benchmark-only `vision-calibration-bench` crate.
- **Facade `vision_calibration::mvg`.** The MVG surface
  (`pose_recovery`, `robust`, `cheirality`, `degeneracy`, `triangulation`,
  `homography`, `rectification`, `residuals`, `dense`, `types`, `error`) is
  now reachable from the facade, mirroring the `geometry` module; `refine`
  gates `bundle_adjust`.
- **`vision_calibration_dataset::device_spec` +
  `vision_calibration_pipeline::device_seed`
  ([ADR 0023](docs/adrs/0023-device-spec-seed-derivation.md)).**
  A `DeviceSpec` sidecar schema (datasheet-natural units: focal length,
  pixel pitch, resolution, Scheimpflug mount angles, rig mechanical
  layout) that derives ADR 0011 manual-init seeds — intrinsics, rig
  layout, hand-eye mounts — instead of hand-coded per-example constants.
  Facade re-export `vision_calibration::device_seed`; spec-seeded init is
  now the **official calibration route**.
- **`calib-bench accept` / `calib-bench basin`.** One-command
  acceptance harness iterating every registered dataset through the seeded
  official route with a hard per-entry gate, committed Fit-record
  baselines and drift gates (`--regression-tol`, `--freeze-baselines`),
  and a convergence-basin study subcommand sweeping focal / tilt /
  principal-point perturbations around the ADR 0023 seed.
- **Proof-pack standard.** `docs/notes/README.md` documents the
  math-note + synthetic-GT matrix test + property test + committed
  Fit record pattern; math notes and matrix tests landed for planar
  intrinsics, Scheimpflug intrinsics, hand-eye, rig extrinsics, laserline
  bundle, and two-view/triangulation.
- **App: Depth workspace.** Dense stereo matching (block/SGM
  toggle) through the rectifier, plus depth-from-disparity reprojection
  and an interactive 3D point cloud view (React-Three-Fiber, code-split).
- **App: schema-driven TypeScript wire types.** `JsonSchema`
  derives on all eight pipeline `Export` types generate
  `app/src/types/generated/` via a `schema-export`-gated `emit_schemas`
  binary; hand-written shape-sniffing (`exportShape.ts`) is deleted in
  favor of the generated, `kind`-grounded `detectExportKind`.
- **App: run progress, cancel, and diagnostics.** Streamed
  per-stage run progress with a working cancel button; sortable per-pose
  residual stats table; cross-camera residual matrix; single-camera
  laserline plane rendering in the 3D viewer.
- **App: internal design system.** Shared `components/ui` set
  (`Button`, `Panel`, `SectionHeader`, `Select`, `Table`, `Banner`, `Badge`,
  `EmptyState`) adopted across all five workspaces; dark-mode contrast
  fixes (light-mode `--brand` was failing WCAG AA).
- **App: unsigned desktop bundles (`app-bundle.yml`).** macOS
  (`.app` / `.dmg`) and Linux (AppImage / `.deb`) bundles built on
  `workflow_dispatch` and `v*` tag pushes, uploaded as workflow-run
  artifacts. Explicitly unsigned/unnotarized — see the workflow for
  documented signing prerequisites.
- **App CI.** ESLint 9 + Prettier + typecheck
  jobs; 35 Vitest component tests; 7 Playwright smoke tests; repo-root
  resolution no longer hard-codes an absolute path.
- **Python: `run_rig_handeye_laserline` binding + `check_binding_parity.py`
  guard.** Closes the last problem type that had no Python binding; the
  parity script now runs in CI.
- New tutorials: multiple-view geometry, distortion-model selection,
  single-camera hand-eye, and an app walkthrough.
- ADR 0023 (DeviceSpec schema) and ADR 0024 (config vocabulary).

### Fixed

- **Exact 180° rotation mis-convergence.** nalgebra's iterative
  `Rotation3::from_matrix` silently mis-converges on exact 180° rotations
  (wrong axis); the row-major pose loader and `base_se3_gripper` now use
  an SVD polar-decomposition `nearest_rotation` instead. Robot poses at
  exactly 180° (common on the rtv3d rigs) previously diverged hand-eye
  calibration to ~1700 px reprojection error.
- `bundle_adjust`'s gauge anchor now falls back to the lowest-index
  *observed* camera instead of always fixing camera 0 (which left the
  gauge free when camera 0 had no observations).
- NaN-rejection regressions reintroduced by the typed-error migration
  (`ensure!(x > 0.0)` → `if x <= 0.0` is not NaN-equivalent) across 10
  validation sites (robust-loss scales, robot-pose sigmas, bound checks).
- Device-spec seed translations now convert the spec's millimetre drawing
  units to the pipeline's metre world unit (caught before any consumer
  shipped).
- The synthetic dense-stereo fixture encoded the wrong disparity-sign
  convention, silently failing any matcher that followed the documented
  `d = x_left - x_right` convention.

## [0.6.0] - 2026-06-17

`0.6.0` promotes `vision-geometry` and `vision-mvg` to the crates.io
publish set (nine publishable crates total). This section backfills the
`Unreleased` entry that was never renamed/dated across the tag.

### Changed
- **BREAKING (`vision-calibration-optim`): camera model as data in the
  factor IR** ([ADR 0020](docs/adrs/0020-camera-model-as-data-factor-ir.md)).
  The 18 enumerated `FactorKind` variants
  (`ReprojPointPinhole4Dist5Scheimpflug2HandEyeRobotDelta`, …) are
  replaced by four families — `ReprojPoint`, `LaserPointToPlane`,
  `LaserLineDistance`, `Se3TangentPrior` — that carry a
  `CameraModelDesc` (projection × distortion × sensor) and a
  `ReprojChain`/`LaserChain` as data. Parameter layouts and validation
  are derived from the descriptors; the backend monomorphizes residual
  kernels once per factor through a single dispatch table. Numerics
  are bit-identical on every production path (pinned by golden-value
  tests). Adding a future camera model is one descriptor variant + one
  kernel + one dispatch row instead of new variants per chain. The
  unused `math::projection` helpers were removed.

### Added
- **Pinhole rig laserline support.**
  `RigHandeyeExport::to_upstream_calibration` and
  `pixel_to_gripper_point` accept pinhole rig hand-eye exports
  (`sensors == None`), substituting exact zero-tilt sensors.

### Fixed
- The `laserline_device` export path computes target residuals through
  the shared generic `compute_planar_target_residuals_views` helper
  instead of a stale inlined projection loop.

## [0.5.1] - 2026-05-23

### Fixed
- **Docs publishing (`publish-docs.yml`).** Repair 7 rustdoc broken
  intra-doc-links in `vision-calibration-dataset` and
  `vision-calibration-detect` (both new crates in `0.5.0`) that
  surfaced as hard errors under `RUSTDOCFLAGS=-D warnings`. The
  workflow ran cleanly under the older `0.4.x` workspace because the
  affected files did not exist. No public API change.
- **PyPI release pipeline (`release-pypi.yml`).** Re-sync
  `crates/vision-calibration-py/pyproject.toml` `project.version` with
  the workspace version. The mismatch (pyproject was stuck at `0.3.0`
  while the workspace shipped `0.4.0` and `0.5.0`) tripped the
  `Verify tag/version sync` gate, so the wheel/sdist build and PyPI
  upload were skipped for both prior tags. Wheels for `0.5.1` are the
  first PyPI upload since `0.3.0`.

## [0.5.0] - 2026-05-21

`0.5.0` bundles two pre-1.0 breaking efforts plus the first desktop
viewer scaffold: the rig-family sensor-axis refactor (ADR 0013) and a
batched public-API-surface revision applied before the library's
contract stabilizes. Pre-1.0, breaking changes are expected and are
collected here in a single minor bump rather than dribbled across
several `0.x.y` releases. The API revision draws three boundaries that
debugging and algorithm work had blurred: stable *results* (`*Export` +
typed `step_*` return values), opt-in *diagnostics* (`session.log()` /
`session.metadata()`), and implementation *internals* (now hidden or
sealed). The three surfaces are documented on the facade's own module docs.

### Added
- **Diagnose UI scaffold ([ADR 0014](docs/adrs/0014-tauri-desktop-app.md)).**
  - `vision_calibration_core::{ImageManifest, FrameRef, PixelRect}` — new
    viewer-facing image-data contract. Pose-major frame list with optional
    per-frame ROI for tiled multi-camera images. Re-exported from
    `vision_calibration::core`.
  - `PlanarIntrinsicsExport` gains an optional
    `image_manifest: Option<ImageManifest>` field. Serde-skipped when
    absent so existing exports remain byte-identical. The other `*Export`
    types gain it in later releases.
  - New example
    `cargo run -p vision-calibration --example planar_synthetic_with_images`
    deterministically renders a 5-pose 9×6 checkerboard fixture
    (`target/fixtures/planar_synthetic_with_images/`) — `export.json`
    with manifest + 5 PNGs — and is the source of truth for the v0
    diagnose viewer's input contract.
  - New regression test
    `crates/vision-calibration/tests/planar_synthetic_with_images.rs`
    pins fixture residuals (mean < 0.5 px, max < 1.5 px) and verifies
    every manifest entry maps to a rendered PNG.
  - New top-level `app/` directory carrying the Tauri 2 + React +
    TypeScript desktop shell. The Rust backend (`app/src-tauri/`) is
    excluded from the workspace via `Cargo.toml`'s
    `exclude = ["app"]`. Two Tauri commands: `load_export` and
    `load_image`. One UI surface: file-open + (pose, camera) dropdown
    + canvas with per-feature residual arrows colored by error
    bucket. See `app/README.md`.
- **ADR 0014** ([`docs/adrs/0014-tauri-desktop-app.md`](docs/adrs/0014-tauri-desktop-app.md))
  records the Tauri 2 + React + TypeScript framework choice (vs
  rerun.io and egui), the decision to build diagnosis before the run
  workflow, the initial viewer-only scope, and the `ImageManifest`
  Export-side contract.
- **ADR 0013** ([`docs/adrs/0013-rig-family-sensor-axis-refactor.md`](docs/adrs/0013-rig-family-sensor-axis-refactor.md))
  records the rig family sensor-axis refactor decision: composition over
  traits, single-axis collapse, alternatives considered.
- **Typed `step_*` return values.** Every step function now returns a
  typed, non-`Option` result instead of `()` — e.g.
  `step_init -> PlanarInitResult`, `step_optimize -> PlanarOptimizeResult`,
  with analogous `*InitResult` / `*OptimizeResult` (and rig-stage
  variants) per problem type. Consumers read step outputs directly
  rather than fishing intermediate values out of `session.state`.
- **`linear::prelude`.** A curated re-export module covering the most
  common `vision-calibration-linear` items, available both on the
  `linear` crate and through `vision_calibration::linear::prelude`.
- **`CalibrationSession::log()` / `metadata()`.** Accessor methods
  returning immutable views of the session log and metadata — the
  documented introspection channel that replaces direct field access.
- **`vision_calibration_pipeline::common`.** New module holding the
  shared step-option structs, re-exported from the facade as
  `vision_calibration::common`.

### Changed (breaking, pre-1.0)
- **Rig family sensor-axis refactor (ADR 0013).** The pinhole and Scheimpflug
  rig modules collapse into a single workflow per problem family. Five rig
  sibling modules become three (~−2,300 LoC across PRs #36, #37, #38).
  - `vision_calibration_pipeline::rig_scheimpflug_extrinsics` — module
    deleted. Migrate to `rig_extrinsics::RigExtrinsicsProblem` with
    `RigExtrinsicsConfig::sensor = SensorMode::Scheimpflug { … }`.
  - `vision_calibration_pipeline::rig_scheimpflug_handeye` — module
    deleted. Migrate to `rig_handeye::RigHandeyeProblem` with
    `RigHandeyeConfig::sensor = SensorMode::Scheimpflug { … }`.
  - `RigExtrinsicsProblem::Output` is now `RigExtrinsicsOutput::{Pinhole,
    Scheimpflug}`; `RigHandeyeProblem::Output` is now
    `RigHandeyeOutput::{Pinhole, Scheimpflug}`. Use accessor methods
    (`cam_to_rig()`, `cameras()`, `sensors()`, `mean_reproj_error()`, …)
    instead of `output.params.*`.
  - `RigExtrinsicsExport` and `RigHandeyeExport` gain
    `sensors: Option<Vec<ScheimpflugParams>>` (`None` for pinhole, `Some(_)`
    for Scheimpflug). `RigExtrinsicsExport` also gains
    `rig_se3_target: Vec<Iso3>` (always populated; pinhole exports
    previously omitted it).
  - `RigIntrinsicsManualInit` (rig_extrinsics) and
    `RigHandeyeIntrinsicsManualInit` gain
    `per_cam_sensors: Option<Vec<ScheimpflugParams>>` for Scheimpflug
    seeds.
  - `RigHandeyeBaConfig` gains `refine_scheimpflug_in_handeye_ba: bool`.
  - `SensorMode` lives in `crate::rig_family::SensorMode` and is
    re-exported by both `rig_extrinsics::SensorMode` and
    `rig_handeye::SensorMode` (single source of truth).
  - `RigHandeyeExport::to_upstream_calibration` (used by
    `rig_laserline_device`) now returns `Result<…>` and errors on pinhole
    rigs.
  - Facade `pixel_to_gripper_point` accepts `&RigHandeyeExport` and errors
    when sensors are absent.
  - Python: `run_rig_scheimpflug_extrinsics`, `run_rig_scheimpflug_handeye`,
    and their dataclasses are removed; `RigHandeyeResult` gains a
    `sensors: list[ScheimpflugSensor] | None` field plus a `to_payload()`
    method that round-trips the unified export. The Python
    `pixel_to_gripper_point` accepts `RigHandeyeResult`.
  - A handful of advanced Scheimpflug-only intrinsics knobs
    (`initial_cameras`, `initial_sensors`, `fallback_to_shared_init`,
    `fix_intrinsics_when_overridden`, `fix_intrinsics_in_percam_ba`,
    `fix_distortion_in_percam_ba`) are dropped. The most load-bearing
    default — `DistortionFixMask::radial_only()` for Scheimpflug
    per-camera intrinsics refinement — is preserved as a hard-coded
    constant.

#### Public-API surface revision

Every break and the concrete migration a consumer must apply:

| Change | Before | After |
|--------|--------|-------|
| `session.state.*` is no longer public — `CalibrationSession::state` is `pub(crate)`, and the seven `*State` structs (`PlanarState`, `SingleCamHandeyeState`, `RigExtrinsicsState`, `RigHandeyeState`, `ScheimpflugIntrinsicsState`, `LaserlineDeviceState`, `RigLaserlineDeviceState`) are `pub(crate)`. Step functions now return their own typed, non-`Option` result. | `step_init(&mut session, None)?; let k = session.state.initial_intrinsics.as_ref().unwrap();` | `let init = step_init(&mut session, None)?; let k = &init.intrinsics;` |
| Session introspection moved to accessors. `CalibrationSession::log` / `::metadata` fields are `pub(crate)`. | `&session.log`, `&session.metadata` | `session.log()`, `session.metadata()` |
| `step_set_*` manual-seed functions renamed to `step_*_with_seed`. | `step_set_init(&mut s, manual, None)?`, `step_set_intrinsics_init(...)`, `step_set_handeye_init(...)`, `step_set_rig_init(...)`, `step_set_intrinsics_init_all(...)` | `step_init_with_seed(...)`, `step_intrinsics_init_with_seed(...)`, `step_handeye_init_with_seed(...)`, `step_rig_init_with_seed(...)`, `step_intrinsics_init_all_with_seed(...)` |
| Facade `linear` no longer glob-re-exports `vision-calibration-linear`. | `use vision_calibration::linear::*;` | import the module (`vision_calibration::linear::homography`), a specific item, or `vision_calibration::linear::prelude::*` for the curated common set |
| Facade `optim` no longer glob-re-exports `vision-calibration-optim`. The curated set is `LaserPlane`, `HandEyeMode`, `RobustLoss`, `LaserlineMeta`, `LaserlineView`, `RobotPoseMeta`, plus the `compute_*_feature_residuals` helpers; typical consumers go through `pipeline`. | `use vision_calibration::optim::*;` | name the specific item (`vision_calibration::optim::LaserPlane`) or use the `pipeline` workflow |
| Facade `synthetic` no longer glob-re-exports `vision_calibration_core::synthetic`. | `use vision_calibration::synthetic::*;` | `use vision_calibration::synthetic::{planar, noise};` |
| `vision-calibration-linear` no longer flattens its modules into the crate root — items live only at their module path. | `linear::dlt_homography`, `linear::CameraMatrixDecomposition`, `linear::DistortionFitOptions` | `linear::homography::dlt_homography`, `linear::camera_matrix::CameraMatrixDecomposition`, `linear::distortion_fit::DistortionFitOptions` (or `linear::prelude` for common items) |
| `vision-calibration-optim` no longer re-exports `core` types. | `vision_calibration_optim::{RigDataset, RigViewObs, View}` | `vision_calibration_core::{RigDataset, RigViewObs, View}` |
| `pixel_to_gripper_point` moved into the `rig_laserline_device` module. | `vision_calibration::pixel_to_gripper_point(...)` | `vision_calibration::rig_laserline_device::pixel_to_gripper_point(...)` — the old crate-root path remains as a `#[deprecated]` alias for one release |
| `#[non_exhaustive]` added to growth-prone public types: all `*Export`, `*Config`, `*ManualInit`, and `*Result` structs/enums (including `PlanarRunResult`, `RigExtrinsicsOutput`, `RigHandeyeOutput`, `LogEntry`, `SessionMetadata`, the per-problem `*Options` structs, and the diagnostic types `ReprojectionStats`, `FeatureResidualHistogram`, `PerFeatureResiduals`, `TargetFeatureResidual`, `LaserFeatureResidual`, `FrameRef`, `ImageManifest`, `PixelRect`). | `Config { a, b }` (bare struct literal) | `Config { a, b, ..Default::default() }`, or a constructor / `Config::default()` then field assignment. Serde round-trips are unchanged. |
| `ProblemType` is now sealed. The seven problem types are a closed set (ADR 0013); a new `pub(crate)` `ProblemState` supertrait blocks downstream `impl`. | `impl ProblemType for MyProblem { ... }` | not supported — use one of the seven built-in problem types |
| `Detector` (`vision-calibration-detect`) is now sealed via a private supertrait. | `impl Detector for MyDetector { ... }` | not supported downstream — use the provided detectors |
| `vision_calibration_core::test_utils` is no longer public API. The module is `#[doc(hidden)]` and gated behind a non-default `test-utils` feature. | `use vision_calibration_core::test_utils::*;` | enable `features = ["test-utils"]` on the dev-dependency, or migrate to `vision_calibration_core::synthetic` helpers |
| RANSAC scaffolding (`Estimator`, `RansacOptions`, `RansacResult`, `ransac_fit`) is `#[doc(hidden)]`. Still `pub` for cross-crate use, but no longer part of the documented surface. | (documented API) | treat as internal; do not rely on it |
| Shared step-option structs hoisted to a new `vision_calibration_pipeline::common` module. `IntrinsicsInitOptions`, `IntrinsicsOptimizeOptions`, `HandeyeInitOptions`, `HandeyeOptimizeOptions` are now single types; they remain re-exported from each problem module, so existing per-module paths still resolve. | `planar_intrinsics::IntrinsicsInitOptions` (a distinct per-problem copy) | `vision_calibration_pipeline::common::IntrinsicsInitOptions` (canonical); per-module re-exports unchanged |

## [0.4.0] - 2026-04-29

### Added
- Scheimpflug rig calibration family:
  - `optim::optimize_rig_extrinsics_scheimpflug` + `RigExtrinsicsScheimpflugParams/SolveOptions/Estimate`.
  - `optim::optimize_handeye_scheimpflug` + `HandEyeScheimpflugParams/SolveOptions/Estimate` (EyeInHand).
  - `optim::optimize_rig_laserline` + `RigLaserlineDataset/Upstream/SolveOptions/Estimate` — per-camera
    laser-plane calibration against a frozen upstream rig calibration, with plane output expressed in
    rig frame.
  - `LaserPlane::transform_by(&Iso3)` utility for frame-to-frame plane transforms.
- Three new session-API pipelines in `vision_calibration_pipeline`:
  - `rig_scheimpflug_extrinsics` (4 steps)
  - `rig_scheimpflug_handeye` (6 steps, EyeInHand)
  - `rig_laserline_device` (2 steps, consumes a frozen `RigScheimpflugHandeyeExport`)
- Facade helper `vision_calibration::pixel_to_gripper_point(cam_idx, pixel, rig_cal, laser_planes_rig)`
  maps a laser pixel in any camera to a 3D point in the robot gripper frame by chaining undistort →
  rig-frame ray → plane intersection → hand-eye transform.
- New IR factor kinds `ReprojPointPinhole4Dist5Scheimpflug2{TwoSE3,HandEye,HandEyeRobotDelta}` with
  matching TinySolver adapters and autodiff-ready residual generics.
- New private example crate `vision-calibration-examples-private` (publish = false) with
  `examples/puzzle_130x130_rig.rs` running the full pipeline on a sensor dataset.

## [0.3.0] - 2026-04-12

### Added
- Typed `Error` enum (using `thiserror`) exposed from every workspace crate:
  `vision-calibration-core`, `vision-calibration-linear`, `vision-calibration-optim`,
  and `vision-calibration-pipeline` now return structured, matchable error variants instead of `anyhow::Error`.
- `# Errors` sections added across fallible public APIs in `vision-calibration-linear`,
  `vision-calibration-optim`, and `vision-calibration-pipeline` rustdoc.
- Optional `tracing` feature on `vision-calibration-core` instruments `ransac_fit` with spans
  (off by default, no runtime cost when disabled).
- Boundary validation for Python inputs with `PyValueError` (high-level `run_*` APIs).
- CI job enforcing the declared MSRV and a typing-stub coverage check for the Python package.

### Changed
- **MSRV bumped to 1.88** (workspace-wide `rust-version = "1.88"`).
- `vision-calibration-core`: renamed `choose_multiple` → `sample` in the RANSAC sampling API.
- Replaced `Option<Vec<_>>` with `Vec<_>` for weights fields across public configs (empty = unweighted).
- Documented the rationale for the `RUSTSEC-2024-0436` audit ignore.
- Dropped redundant empty `[features]` blocks in `vision-calibration-optim`.

### Breaking
- Minor release bump to `0.3.0` for the migration from `anyhow::Error` to typed `Error`
  in `vision-calibration-{core,linear,optim,pipeline}` public signatures.
- `choose_multiple` → `sample` rename in `vision-calibration-core`.
- MSRV raised to 1.88.
- `weights: Option<Vec<_>>` replaced by `weights: Vec<_>` in public config structs.

## [0.2.0] - 2026-03-07

### Added
- New high-level Scheimpflug intrinsics workflow:
  - Rust API: `vision_calibration::scheimpflug_intrinsics::run_calibration`
  - Python API: `vision_calibration.run_scheimpflug_intrinsics`
- Typed Python camera/result payload models for high-level workflows (planar, hand-eye, rig, laserline, Scheimpflug)
- Synthetic integration tests for Scheimpflug calibration in `vision-calibration`
- Facade API compile-surface integration tests to catch accidental public API regressions
- Session schema metadata validation tests for JSON session compatibility checks

### Changed
- Enforced workspace layering by moving Scheimpflug solver implementation to `vision-calibration-pipeline` and keeping `vision-calibration` as facade re-export
- Hardened facade/API contracts:
  - `#[non_exhaustive]` added to public config/export/error structs and enums
  - `vision-calibration` now enforces `#![deny(missing_docs)]`
  - Session import now rejects schema metadata mismatches with explicit errors
- Expanded rustdoc/book/readme coverage with updated workflow usage snippets
- CI and release hardening:
  - `cargo clippy` now runs with `--all-features`
  - `cargo test` now runs with `--all-features`
  - Python extension build + runtime tests are part of CI/release checks
- Python high-level bindings are now typed-first end-to-end:
  - top-level `run_*` APIs require typed dataset/config dataclasses
  - dict/list compatibility paths moved to explicit low-level raw helpers in `vision_calibration._api`
  - `vision_calibration.types` reduced to low-level compatibility surface

### Breaking
- Minor release bump to `0.2.0` due public API contract hardening and Python binding migration
- Python high-level runners are typed-only and no longer accept raw mapping/list payloads
- Dict-based high-level result access patterns (`camera`/`cameras`/`estimate`/`stats`/`raw` as mappings) were replaced by typed model fields
- Several low-level type aliases are no longer re-exported at the top-level Python package; import from `vision_calibration.types` only when using low-level compatibility APIs

## [0.1.2]

### Added
- **Iterative intrinsics estimation** in `vision-calibration-linear`: New `IterativeIntrinsicsSolver` for jointly estimating camera intrinsics (K) and Brown-Conrady distortion without requiring ground truth distortion preprocessing
- **Distortion fitting** in `vision-calibration-linear`: New `DistortionSolver` for closed-form estimation of radial (k1, k2, k3) and tangential (p1, p2) distortion coefficients from homography residuals
- **Shared test utilities** in `vision-calibration-core`: New `test_utils` module providing common calibration test data structures (`CalibrationView`, `ViewDetections`, `CornerInfo`) and helper functions
- **Realistic calibration tests**: New integration tests demonstrating full calibration pipeline without ground truth distortion (in `stereo_linear.rs` and `planar_intrinsics_real_data.rs`)
- Comprehensive documentation for all new modules with usage examples and algorithm descriptions
- Linear solver additions in `vision-calibration-linear`: camera matrix DLT + RQ decomposition, linear triangulation, 7-point fundamental, 5-point essential + decomposition, P3P, and EPnP
- New `vision-calibration-linear` README with algorithm overview and usage notes

### Changed
- Updated `vision-calibration-linear` lib.rs to export new `distortion_fit` and `iterative_intrinsics` modules
- Refactored test files to use shared utilities from `vision-calibration-core::test_utils`, eliminating code duplication
- Updated CLAUDE.md with detailed documentation of iterative intrinsics feature and typical workflow
- Expanded rustdoc across `vision-calibration-linear` algorithms and updated top-level README to reflect new solver coverage
