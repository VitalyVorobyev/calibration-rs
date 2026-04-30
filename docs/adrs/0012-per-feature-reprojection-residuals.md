# ADR 0012: Per-Feature Reprojection Residuals on Export Types

- Status: Accepted
- Date: 2026-04-30

## Context

Every `*Export` type in `vision-calibration-pipeline` currently exposes only
`mean_reproj_error: f64` plus `per_cam_reproj_errors: Vec<f64>`. There is no way
for a downstream consumer (the planned Tauri/React diagnose UI in roadmap track
B5; ad-hoc analyses; the puzzle 130x130 viewer) to drill into per-(view, camera,
feature) errors without re-running geometry — re-projecting target points and
recomputing laser point-to-plane residuals using the calibrated camera +
extrinsics + target poses.

The existing real-world consumer of "diagnose"-style data is the puzzle 130x130
example viewer (`crates/vision-calibration-examples-private/examples/puzzle_130x130_rig/viewer.rs:592-612`),
which today hand-rolls per-feature reprojection inside the example crate. The
React diagnose UI (B5) needs the same data shape, served through the public
JSON exported by `*Export` types so the UI does not have to bundle calibration
math.

The richest aggregate stats already present in the workspace are in
`RigHandeyeLaserlinePerCamStats`
(`crates/vision-calibration-optim/src/problems/rig_handeye_laserline_bundle.rs:233-257`),
which carries per-camera mean / max / 5-bucket histograms but never per-feature.

## Decision

Add a `per_feature_residuals: PerFeatureResiduals` field to every `*Export`
type, populated in `ProblemType::export()`. A small set of shared record types
in `vision-calibration-core` defines the per-feature shape, and helpers in
`-core` and `-optim` produce the records from a calibrated camera + dataset +
recovered poses.

```rust
// crates/vision-calibration-core/src/types/residual.rs

pub struct TargetFeatureResidual {
    pub pose: usize,
    pub camera: usize,
    pub feature: usize,
    pub target_xyz_m: [f64; 3],
    pub observed_px: [f64; 2],
    pub projected_px: Option<[f64; 2]>,
    pub error_px: Option<f64>,
}

pub struct LaserFeatureResidual {
    pub pose: usize,
    pub camera: usize,
    pub feature: usize,
    pub observed_px: [f64; 2],
    pub residual_m: Option<f64>,
    pub residual_px: Option<f64>,
    pub projected_line_px: Option<[[f64; 2]; 2]>,
}

pub struct FeatureResidualHistogram {
    pub bucket_edges_px: [f64; 4],  // [1.0, 2.0, 5.0, 10.0]
    pub counts: [usize; 5],         // [<=1, <=2, <=5, <=10, >10]
    pub count: usize,
    pub mean: f64,
    pub max: f64,
}

pub struct PerFeatureResiduals {
    pub target: Vec<TargetFeatureResidual>,
    pub laser: Vec<LaserFeatureResidual>,
    pub target_hist_per_camera: Option<Vec<FeatureResidualHistogram>>,
    pub laser_hist_per_camera: Option<Vec<FeatureResidualHistogram>>,
}
```

### Indexing convention

- **Pose-major flat `Vec`s.** Every record carries an explicit
  `(pose, camera, feature)` triple. Iteration order: outer = view, inner =
  camera (camera-`None` slots skipped), innermost = feature index in that
  view's `points_3d`. This matches the existing transform chain in
  `compute_rig_reprojection_stats_per_camera`
  (`crates/vision-calibration-core/src/lib.rs:259`).
- Flat Vecs keep JSON small and React-friendly: downstream filters by camera
  or pose with `.filter()` rather than re-flattening nested arrays.
- For single-camera problem types (`PlanarIntrinsics`, `ScheimpflugIntrinsics`,
  `LaserlineDevice`, `SingleCamHandeye`), `camera = 0` everywhere. Histograms
  carry one entry.

### `Option<T>` semantics

- `projected_px = None` and `error_px = None` when projection diverges (point
  behind camera, distortion fixed-point fails). Mirrors the puzzle viewer's
  treatment (`viewer.rs:215-220`).
- `residual_m = None` and `residual_px = None` when the laser ray does not
  intersect the target plane.
- `projected_line_px = None` when the camera cannot synthesize the line
  endpoints in image space.
- The container `PerFeatureResiduals` is always `Default`-constructable; empty
  `Vec`s mean "no observations of this flavor exist for this problem type"
  (e.g., `PlanarIntrinsicsExport.per_feature_residuals.laser` is always empty).
- `target_hist_per_camera` / `laser_hist_per_camera` are `Option` so a problem
  type can deliberately skip the per-camera aggregate (e.g., when the mean is
  trivially uninformative). When `Some`, length matches `num_cameras`.

### Histogram bucket boundaries

- Reprojection bucket edges fixed at **`[1.0, 2.0, 5.0, 10.0]` pixels**, giving
  buckets `[<=1, <=2, <=5, <=10, >10]`. This matches
  `RigHandeyeLaserlinePerCamStats::reproj_histogram_px`
  (`crates/vision-calibration-optim/src/problems/rig_handeye_laserline_bundle.rs:243`)
  so per-camera histograms aggregate identically across the workspace.
- The bucket edges live on the histogram struct as a `[f64; 4]` so the schema
  is self-documenting in JSON.
- Laser-distance histograms reuse the same struct but expose pixel-domain
  errors via `residual_px`; meter-domain laser errors are not exposed in this
  schema slice (point-to-plane meter errors stay in the existing
  `RigHandeyeLaserlinePerCamStats` aggregate). Future ADRs can extend.

### `#[non_exhaustive]` rule

ADR 0011 stated: "Do NOT apply `#[non_exhaustive]`" on ManualInit structs
because cross-crate `..Default::default()` construction breaks under
`#[non_exhaustive]`. The same rule applies here:

- **New core record types** (`TargetFeatureResidual`, `LaserFeatureResidual`,
  `FeatureResidualHistogram`, `PerFeatureResiduals`) **do NOT** apply
  `#[non_exhaustive]`. Tests, examples, and the future React UI's Rust shim
  construct these.
- **Existing `*Export` envelope types** keep their `#[non_exhaustive]` —
  downstream code is expected to read them, not construct them. Adding a new
  field is non-breaking on the wire.

### Where the helpers live

Per the layered crate rules of ADR 0006 (`core` < `linear`/`optim` < `pipeline`):

- `compute_planar_target_residuals(camera, dataset, camera_se3_target) ->
  Vec<TargetFeatureResidual>` lives in `vision-calibration-core/src/lib.rs`.
  Single-camera reprojection only needs `PinholeCamera::project_point_c`
  (`crates/vision-calibration-core/src/models/camera.rs:56`).
- `compute_rig_target_residuals(cameras, dataset, cam_se3_rig, rig_se3_target)
  -> Vec<TargetFeatureResidual>` lives in `vision-calibration-core/src/lib.rs`,
  next to `compute_rig_reprojection_stats_per_camera` and reuses the same
  transform chain.
- `build_feature_histogram(errors_px) -> FeatureResidualHistogram` lives in
  `vision-calibration-core/src/lib.rs` next to the residual helpers.
- `compute_laserline_feature_residuals(dataset, params, residual_type) ->
  Vec<LaserFeatureResidual>` lives in
  `vision-calibration-optim/src/problems/laserline_bundle.rs` because it
  depends on `LaserPlane` and laser-projection geometry that is `optim`-private.
  The record type itself stays in `core`.

### Threading the dataset to `export()`

Computing per-feature residuals requires the input observations (the observed
pixel for each feature) which were not previously available in
`ProblemType::export()`. The trait signature is changed (breaking, pre-1.0):

```diff
-fn export(output: &Self::Output, config: &Self::Config) -> Result<Self::Export, Error>;
+fn export(input: &Self::Input, output: &Self::Output, config: &Self::Config) -> Result<Self::Export, Error>;
```

The single internal caller (`CalibrationSession::export`) already owns
`&self.input`. Each problem type's `export()` is updated mechanically.

### JSON stability promise

The `*Export` JSON is the **public IPC** consumed by external tooling (the
React diagnose UI, Python notebooks, regression-test fixtures).

- Adding a new optional field (with `serde(default)` on the read side) is
  non-breaking. New `*Export` fields land at the same `schema_version`.
- Renaming or removing a field requires bumping `ProblemType::schema_version()`
  per ADR 0007.
- Field ordering inside Vecs is part of the contract (pose-major, then camera,
  then feature). Implementations that rearrange records must bump the schema
  version even though the JSON is technically still parseable.
- New core record types start at the parent export's schema version; they do
  not carry their own.

### Per-problem extension recipe

When extending a new `*Export` type to carry per-feature residuals:

1. Compute the data inside `ProblemType::export()` using the new helpers in
   `core` / `optim`. The required ingredients are:
   - The input dataset (now available via `&Self::Input`).
   - The calibrated camera(s) — already in `Self::Output`.
   - The recovered target / rig / handeye poses — already in `Self::Output`.
2. Populate `target` and (if applicable) `laser` Vecs.
3. Compute per-camera histograms with `build_feature_histogram` if the problem
   has more than one camera or the consumer benefits from the aggregate;
   otherwise `target_hist_per_camera = None`.
4. Add a unit test asserting `target.len() == expected_count` on a synthetic
   perfect-data fixture and a JSON roundtrip test that includes the new field.

## Implementation status (this PR)

- ADR landed.
- Shared record types in `vision-calibration-core::types::residual`.
- Helpers in `vision-calibration-core` (planar, rig) and
  `vision-calibration-optim::problems::laserline_bundle` (laser).
- `ProblemType::export(input, output, config)` signature change applied to
  all nine problem types.
- Three exports populated:
  - `PlanarIntrinsicsExport`
  - `RigExtrinsicsExport`
  - `LaserlineDeviceExport`
- Facade `vision_calibration::core` re-exports the new record types.
- `manual_init_proof` example prints per-camera histograms + head/tail of the
  per-feature error vector and asserts the histogram count sums to
  `target.len()`.

The remaining six exports follow as mechanical PRs in this order:

1. `ScheimpflugIntrinsicsExport`, `RigScheimpflugExtrinsicsExport`.
2. `SingleCamHandeyeExport`, `RigHandeyeExport`, `RigScheimpflugHandeyeExport`
   — adds `compute_handeye_target_residuals` to `core`. **Anchor PR for the
   puzzle 130x130 rig.**
3. `RigLaserlineDeviceExport` + refactor
   `puzzle_130x130_rig/viewer.rs` to consume `Export.per_feature_residuals`
   instead of recomputing.

## Consequences

Positive:

- The future React diagnose UI consumes calibration outputs as static JSON
  without bundling reprojection math.
- The puzzle 130x130 viewer (and any future viewer) becomes a thin renderer
  rather than a parallel implementation of geometry.
- Per-camera histograms unify the workspace's residual reporting; the existing
  ad-hoc `[usize; 5]` arrays in
  `RigHandeyeLaserlinePerCamStats::reproj_histogram_px` are retained for
  backwards-compat but new code uses `FeatureResidualHistogram`.
- Trait change to `ProblemType::export(&Self::Input, …)` makes the input
  available to `export()` for any future export-side computation, not just
  residuals.

Negative:

- `*Export` JSON grows. For a 20-pose 6-camera rig with 130-corner targets
  that is ~15k records per export — order-of-magnitude larger payload, but
  still well under typical JSON-parsing thresholds.
- Computing residuals during `export()` adds work proportional to the dataset
  size. Empirically negligible compared to optimization time; bench under
  budget for stereo_charuco's 8-view dataset is single-digit milliseconds.
- The `ProblemType::export` signature change is a breaking change for anyone
  who has a custom `ProblemType` impl outside the workspace. Pre-1.0; no
  external impls are known.

## Cross-references

- ADR 0006 — Layered Crate Architecture (where helpers and types live).
- ADR 0007 — Session Framework (the `ProblemType` trait, `schema_version`).
- ADR 0011 — Manual Parameter Initialization Workflow (the `Option<T>` and
  `#[non_exhaustive]` conventions reused here).
