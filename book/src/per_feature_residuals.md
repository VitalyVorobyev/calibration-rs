# Per-Feature Residuals

After a calibration finishes, the only error metric most pipelines surface is the **mean reprojection error** in pixels. That number says *something is wrong* but not *which corners are wrong*. To debug a low-quality calibration (a board that flexed, a corner detection that drifted, a camera with a borderline field of view) you need the per-corner error broken down by view and camera.

Every `*Export` therefore carries per-feature residual records, so a consumer (the desktop app's Diagnose workspace, a Python notebook, a regression test) can drill in without re-running any geometry.

## Structure

Every export has a `per_feature_residuals: PerFeatureResiduals` field. The container is always present (empty by default); empty inner vectors mean "this problem type does not produce observations of that flavour". For example, `PlanarIntrinsicsExport.per_feature_residuals.laser` is always empty.

Each export also serializes a required `kind` discriminator (`ExportKind`, snake_case: `"planar_intrinsics"`, `"rig_handeye"`, ...) as its first field, so a consumer that loads raw JSON can narrow on the single tag instead of probing which fields are present.

```text
PerFeatureResiduals
├─ target: Vec<TargetFeatureResidual>     // per-corner reprojection records
├─ laser:  Vec<LaserFeatureResidual>      // per-pixel laser residuals
├─ laser_lines: Vec<LaserLineTrace>       // the projected laser line of each view
├─ target_hist_per_camera: Option<Vec<FeatureResidualHistogram>>
└─ laser_hist_per_camera:  Option<Vec<FeatureResidualHistogram>>
```

A target record carries an explicit `(pose, camera, feature)` triple, the 3D target point `target_xyz_m`, the `observed_px` and `projected_px` pixels and the `error_px` between them. Records are pose-major: the outer loop is the view, the inner loop the camera (cameras without an observation in that view are skipped), the innermost the feature index in that view's and camera's `points_3d`.

`Option` fields carry two meanings:

- `error_px = None` (with `projected_px = None`): the projection diverged for that record, for instance a point behind the camera. This separates "feature absent" from "feature present but the model is bad".
- `residual_m = None` on a laser record: the back-projected ray missed the laser plane.

Histogram bucket edges are fixed at `[1, 2, 5, 10]` px, giving the buckets `<=1`, `<=2`, `<=5`, `<=10`, `>10`. A histogram aggregates only records that produced a finite `error_px`.

## Walkthrough

The examples below use the stereo chessboard rig of [Stereo Rig with Real Data](stereo_real_data.md). The `manual_init` example prints the same quantities.

### 1. Run a calibration

No configuration is needed: the records are always populated.

```rust
use vision_calibration::prelude::*;
use vision_calibration::rig_extrinsics::{RigExtrinsicsProblem, run_calibration};

let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
session.set_input(my_dataset)?;
run_calibration(&mut session)?;
let export = session.export()?;
```

### 2. Summarize the histograms

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

Output on the 20-view stereo chessboard rig:

```text
3080 target records (3080 converged, 0 divergent)
cam 0: count=1540 mean=0.2448px max=1.0262px buckets [1539, 1, 0, 0, 0]
cam 1: count=1540 mean=0.2544px max=0.9146px buckets [1540, 0, 0, 0, 0]
```

### 3. Find the worst corners

Sort by `error_px` to triage outliers:

```rust
let mut worst: Vec<_> = pf
    .target
    .iter()
    .filter_map(|r| r.error_px.map(|e| (e, r)))
    .collect();
worst.sort_by(|a, b| b.0.total_cmp(&a.0));
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

A diagnose view typically projects this onto the source image as a heatmap; the `(observed_px, projected_px)` pair lets you draw an arrow from observed to projected for every record.

### 4. Cross-check against the aggregate

The per-camera histogram `mean` must agree with the mean recomputed from the records. The `manual_init` example asserts this:

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

## Laser Residuals

`LaserlineDeviceExport` and `RigLaserlineDeviceExport` fill `laser` and `laser_lines` in addition to `target`. Each `LaserFeatureResidual` carries:

- `residual_m`: the point-to-plane distance in metres from the back-projected ray's intersection with the target plane to the calibrated laser plane;
- `residual_px`: the distance in pixels from the observed pixel to the projected laser line. This is the line-distance residual the laser solve minimizes, measured on the undistorted image plane.

Each `LaserLineTrace` in `laser_lines` is one view's projected laser line (`pose`, `camera`, `points_px`): a polyline over the extent of that view's laser pixels, for 2D overlays. Lens distortion bends the line, so draw all of its points rather than a segment between the first and last.

## Which Camera Chain Produced Them

The records use exactly the camera chain the optimizer used, so they agree with the reported mean error:

- **Hand-eye exports** (`SingleCamHandeyeExport`, `RigHandeyeExport`) derive each view's camera-from-target pose from the hand-eye chain; the export already carries the resulting records.
- **Scheimpflug cameras** (a planar or rig camera whose `CameraParams` carries a tilted sensor) are projected through the full pinhole, distortion and Scheimpflug chain.

## JSON and Python

Every record type derives `Serialize` and `Deserialize`, and the export round-trips through JSON:

```rust
let json = serde_json::to_string(&export)?;
let restored: RigExtrinsicsExport = serde_json::from_str(&json)?;
assert_eq!(restored.per_feature_residuals, export.per_feature_residuals);
```

Empty vectors and `None` histograms are skipped on the wire (`{}` for an empty container).

In Python the typed result classes do not wrap the records; they stay in the raw export that every result keeps in `result.export` (a dict with the JSON layout above):

```python
pf = result.export["per_feature_residuals"]
worst = sorted(
    (r for r in pf["target"] if r["error_px"] is not None),
    key=lambda r: -r["error_px"],
)[:5]
for r in worst:
    print(r["pose"], r["camera"], r["feature"], f'{r["error_px"]:.3f}')
print(pf["target_hist_per_camera"][0]["counts"])
```

## Where Else They Show Up

The desktop app's Diagnose workspace draws these records on the source images, with a histogram, a per-pose table and a cameras-by-poses matrix (see [Inspecting Results](app_inspect.md)).
