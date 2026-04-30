# Per-feature reprojection residuals

> Onboarding tutorial for [ADR 0012](../adrs/0012-per-feature-reprojection-residuals.md).
> Runnable companion: [`manual_init_proof.rs`](../../crates/vision-calibration/examples/manual_init_proof.rs).

## Why

After a calibration finishes, the only error metric most pipelines surface is
**mean reprojection error in pixels**. That number tells you *something is
wrong* but not *which corners are wrong*. To debug a low-quality calibration
— a board that flexed, a corner detection that drifted, a camera with
borderline FOV — you need the per-corner error broken down by view and
camera.

`calibration-rs` 0.4 attaches per-feature residual records to every
`*Export` type so downstream consumers (a React diagnose UI, a Python
notebook, a regression test) can drill in without re-running geometry.

## Mental model

Every `*Export` carries a `per_feature_residuals: PerFeatureResiduals`
field. The container is always present (`Default` is empty); empty inner
`Vec`s mean "this problem type does not produce observations of that flavor"
(e.g., `PlanarIntrinsicsExport.per_feature_residuals.laser` is always
empty).

```
PerFeatureResiduals
├─ target: Vec<TargetFeatureResidual>     // per-corner reprojection records
├─ laser:  Vec<LaserFeatureResidual>      // per-pixel laser residuals
├─ target_hist_per_camera: Option<Vec<FeatureResidualHistogram>>
└─ laser_hist_per_camera:  Option<Vec<FeatureResidualHistogram>>
```

The records carry an explicit `(pose, camera, feature)` triple and pose-major
ordering. Iteration: outer = view, inner = camera (camera-`None` slots
skipped), innermost = feature index in that view+camera's `points_3d`.

`Option<f64>` semantics:

- `error_px = None` → projection diverged for that record. Use this to
  separate "feature absent" from "feature present but our model is bad".
- `residual_m = None` → the back-projected ray missed the laser plane.

Histogram bucket edges are fixed at `[1, 2, 5, 10]` px (so `<=1`, `<=2`,
`<=5`, `<=10`, `>10`).

## Walkthrough

We will exercise it on the stereo ChArUco dataset shipped in `data/`.

### 1. Run a calibration

Same as any other rig calibration — no configuration knobs are needed for
per-feature residuals; the schema is always populated.

```rust
use vision_calibration::prelude::*;
use vision_calibration::rig_extrinsics::{
    RigExtrinsicsProblem, run_calibration,
};

let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
session.set_input(my_dataset)?;
run_calibration(&mut session)?;
let export = session.export()?;
```

### 2. Sum the histogram counts

Every per-camera histogram aggregates only records that produced a finite
`error_px`. The remaining records are projection failures.

```rust
let pf = &export.per_feature_residuals;
let total = pf.target.len();
let with_value = pf.target.iter().filter(|r| r.error_px.is_some()).count();
println!("{total} target records ({with_value} converged, {} divergent)",
         total - with_value);

if let Some(hists) = pf.target_hist_per_camera.as_ref() {
    for (i, h) in hists.iter().enumerate() {
        println!("cam {i}: count={} mean={:.4}px max={:.4}px buckets {:?}",
                 h.count, h.mean, h.max, h.counts);
    }
}
```

Real output from `manual_init_proof` (8-view stereo ChArUco):

```
4865 target records (4865 converged, 0 divergent)
cam 0: count=2503 mean=0.5969px max=1.8740px buckets [2114, 389, 0, 0, 0]
cam 1: count=2362 mean=0.4869px max=1.6632px buckets [2233, 129, 0, 0, 0]
```

### 3. Drill into the worst corner

Filter to records with the highest `error_px` to triage outliers:

```rust
let mut worst: Vec<_> = pf
    .target
    .iter()
    .filter_map(|r| r.error_px.map(|e| (e, r)))
    .collect();
worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
for (err, r) in worst.iter().take(5) {
    println!(
        "pose={} cam={} feat={} target_xy_m=({:.3},{:.3}) observed=({:.1},{:.1}) err={:.3}px",
        r.pose, r.camera, r.feature,
        r.target_xyz_m[0], r.target_xyz_m[1],
        r.observed_px[0], r.observed_px[1],
        err
    );
}
```

A diagnose UI typically projects this into a heatmap on the source image:
the `(observed_px, projected_px)` pair lets you draw an arrow from observed
to projected for every record.

### 4. Cross-check with the aggregate mean

If the per-camera histogram's `mean` field disagrees with the export's
`per_cam_reproj_errors`, something is inconsistent. The
`manual_init_proof` example asserts this round-trip:

```rust
let mean_from_records: f64 = pf
    .target
    .iter()
    .filter(|r| r.camera == cam_idx)
    .filter_map(|r| r.error_px)
    .sum::<f64>()
    / hist.count as f64;
assert!((mean_from_records - hist.mean).abs() < 1e-9);
```

## Common variations

### Laser residuals

`LaserlineDeviceExport` and `RigLaserlineDeviceExport` populate `pf.laser`
in addition to `pf.target`. Each `LaserFeatureResidual` carries:

- `residual_m` — point-to-plane distance in meters from the back-projected
  ray's intersection with the target plane to the calibrated laser plane.
- `residual_px` — distance in image space from the observed pixel to the
  projected laser line.
- `projected_line_px` — two endpoints of the projected laser line, useful
  for 2D overlays.

### Hand-eye chain

`SingleCamHandeyeExport`, `RigHandeyeExport`, and
`RigScheimpflugHandeyeExport` derive per-view `cam_se3_target` from the
hand-eye chain via
[`handeye_observer_se3_target`](../../crates/vision-calibration-optim/src/problems/handeye.rs).
You don't need to reproduce that math — the export already carries the
ground truth records.

### Scheimpflug projection

`ScheimpflugIntrinsicsExport`, `RigExtrinsicsExport` (when
`SensorMode::Scheimpflug` is configured — `sensors` field populated), and
`RigScheimpflugHandeyeExport` build the full pinhole + Brown-Conrady +
Scheimpflug `Camera` for projection. The records are computed using the
exact same chain the optimizer used.

### JSON consumption

Every record type derives `Serialize, Deserialize`. The export round-trips
through JSON:

```rust
let json = serde_json::to_string(&export)?;
let restored: RigExtrinsicsExport = serde_json::from_str(&json)?;
assert_eq!(restored.per_feature_residuals, export.per_feature_residuals);
```

Empty `Vec`s and `None` histograms are skipped on the wire (`{}` for an
empty container).

### Python

Python bindings deserialize the export via `pythonize`; the records appear
as native dicts/lists.

## What to read next

- [ADR 0012](../adrs/0012-per-feature-reprojection-residuals.md) — schema
  rationale, indexing convention, JSON-stability promise.
- [ADR 0009](../adrs/0009-coordinate-and-pose-conventions.md) — pose
  naming (`frame_se3_frame`).
- [`PerFeatureResiduals`](../../crates/vision-calibration-core/src/types/residual.rs)
  — the struct definitions with full doc comments.
- [`compute_rig_target_residuals`](../../crates/vision-calibration-core/src/lib.rs)
  — the workhorse that produces records for any rig pipeline.
- [`manual_init_proof.rs`](../../crates/vision-calibration/examples/manual_init_proof.rs)
  — the example this tutorial is built on; runs against `data/stereo_charuco/`.
