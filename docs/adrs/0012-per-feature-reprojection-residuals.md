# ADR 0012: Per-Feature Reprojection Residuals on Export Types

- Status: Accepted
- Date: 2026-04-30

## Context

Aggregate statistics (mean reprojection error, per-camera means) are not enough for diagnosis.
Downstream consumers (the desktop app's Diagnose workspace, ADR 0014; notebooks; regression
fixtures) need per-(view, camera, feature) errors. Re-deriving them from an export would force
every consumer to bundle calibration math (re-project target points, recompute laser
point-to-plane residuals from cameras, extrinsics and target poses). The `*Export` JSON is the
public IPC, so it should carry the data.

## Decision

Every `*Export` type in `vision-calibration-pipeline` carries a
`per_feature_residuals: PerFeatureResiduals` field, populated in `ProblemType::export()`. The
record types live in `vision-calibration-core::types::residual`:

```rust
pub struct TargetFeatureResidual {
    pub pose: usize, pub camera: usize, pub feature: usize,
    pub target_xyz_m: [f64; 3],
    pub observed_px: [f64; 2],
    pub projected_px: Option<[f64; 2]>,
    pub error_px: Option<f64>,
}

pub struct LaserFeatureResidual {
    pub pose: usize, pub camera: usize, pub feature: usize,
    pub observed_px: [f64; 2],
    pub residual_m: Option<f64>,
    pub residual_px: Option<f64>,  // the solver's line distance, px
}

pub struct LaserLineTrace {
    pub pose: usize, pub camera: usize,
    pub points_px: Vec<[f64; 2]>,  // polyline over the view's laser pixels
}

pub struct FeatureResidualHistogram {
    pub bucket_edges_px: [f64; 4],  // [1.0, 2.0, 5.0, 10.0]
    pub counts: [usize; 5],         // [<=1, <=2, <=5, <=10, >10]
    pub count: usize, pub mean: f64, pub max: f64,
}

pub struct PerFeatureResiduals {
    pub target: Vec<TargetFeatureResidual>,
    pub laser: Vec<LaserFeatureResidual>,
    pub laser_lines: Vec<LaserLineTrace>,
    pub target_hist_per_camera: Option<Vec<FeatureResidualHistogram>>,
    pub laser_hist_per_camera: Option<Vec<FeatureResidualHistogram>>,
}
```

### Indexing convention

- **Pose-major flat `Vec`s.** Every record carries an explicit `(pose, camera, feature)` triple.
  Order: outer = view, inner = camera (missing-camera slots skipped), innermost = feature index in
  that view's `points_3d`. Flat Vecs keep the JSON small and let consumers filter by camera or pose
  without re-flattening nested arrays.
- Single-camera problem types use `camera = 0`; histograms carry one entry.

### `Option<T>` semantics

- `projected_px` and `error_px` are `None` when projection diverges (point behind camera,
  distortion inversion fails).
- `residual_m` is `None` when the laser ray does not meet the target plane; `residual_px` is
  `None` when the laser and target planes have no line in common.
- `residual_px` is the line-distance residual the laser solve minimizes (undistorted image plane,
  scaled by `sqrt(fx·fy)`), computed by the same kernel, so the diagnostic and the cost agree.
- The projected laser line is stored once per (view, camera) in `laser_lines`, as a polyline over
  the extent of the view's laser pixels: distortion bends it, and a two-point segment through far
  samples misplaces it by pixels under strong distortion.
- `PerFeatureResiduals` is always `Default`-constructible; an empty `Vec` means "no observations
  of this flavor for this problem type" (e.g. `PlanarIntrinsicsExport` has no laser records).
- The per-camera histograms are `Option`; when `Some`, length equals `num_cameras`.

### Histogram buckets

Reprojection bucket edges are fixed at `[1.0, 2.0, 5.0, 10.0]` px (`REPROJECTION_HISTOGRAM_EDGES_PX`),
giving `[<=1, <=2, <=5, <=10, >10]`, and are stored on the struct so the JSON is self-documenting.
`RigHandeyeLaserlinePerCamStats::reproj_histogram_px` uses the same buckets, so aggregates compare
across the workspace. Laser histograms use the pixel-domain `residual_px`; meter-domain laser
errors are not histogrammed.

### Where the helpers live

Following the layering of ADR 0006:

- `compute_planar_target_residuals`, `compute_rig_target_residuals` and `build_feature_histogram`
  live in `vision-calibration-core` next to `compute_rig_reprojection_stats_per_camera`, sharing
  its transform chain.
- `compute_laserline_feature_residuals` (and the rig variant) live in
  `vision-calibration-optim::problems`, because they depend on the laser-plane geometry that is
  `optim`-private. The record types stay in `core`.

### Export receives the input

`ProblemType::export(input, output, config)` takes the input dataset, since the observed pixels
are needed to build residual records. `CalibrationSession::export` already owns `&self.input`.

### Struct evolution and JSON stability

- The core record types and the `*Export` envelopes are all `#[non_exhaustive]`; consumers read
  them and construct them via `Default` plus field assignment, so adding a field is not a source
  break.
- Adding an optional field (with `serde(default)` on the read side) is non-breaking on the wire
  and keeps the same `schema_version`. Renaming or removing a field bumps
  `ProblemType::schema_version()` (ADR 0007).
- Record ordering (pose-major, then camera, then feature) is part of the contract; reordering
  requires a schema-version bump even though the JSON still parses.
- Core record types do not carry their own version; they follow the parent export's.

### Adding residuals to a new `*Export`

Compute the records in `ProblemType::export()` from the input dataset and the calibrated cameras
and poses in `Self::Output`; fill `target` (and `laser` if applicable); fill per-camera
histograms with `build_feature_histogram` when more than one camera exists or the consumer
benefits; add a test asserting record counts on perfect synthetic data and a JSON round-trip.

## Consequences

Positive:

- Consumers read calibration diagnostics as static JSON, without bundling reprojection math.
- Viewers are thin renderers rather than parallel implementations of the geometry.
- Per-camera histograms unify residual reporting across problem types.
- `export()` has access to the input for any future export-side computation.

Negative:

- `*Export` JSON grows with dataset size (a 20-pose, 6-camera rig with 130-corner targets yields
  on the order of 15k records).
- Computing residuals in `export()` adds work proportional to the dataset size, negligible next
  to optimization.
- Custom `ProblemType` impls outside the workspace must accept the `input` parameter.
