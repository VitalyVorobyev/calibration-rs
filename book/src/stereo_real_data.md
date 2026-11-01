# Stereo Rig with Real Data

This chapter walks through the `stereo_session` example, which calibrates a stereo camera rig from synchronized left/right views with `RigExtrinsicsProblem`. It is the quickest way to see a complete multi-camera calibration on real data: the corners are already detected and committed to the repository, so no images are needed.

## Run It First

```bash
cargo run -p vision-calibration --example stereo_session --release
```

The program loads 20 views of a 7x11 chessboard (30 mm squares) seen by both cameras and prints, among other lines:

```text
--- Step 2: Per-Camera Intrinsics Optimization ---
  Camera 0: fx=713.8, fy=724.7, cx=523.4, cy=286.0, reproj_err=0.234px
  Camera 1: fx=724.8, fy=735.6, cx=513.9, cy=293.0, reproj_err=0.244px

--- Step 4: Rig Bundle Adjustment ---
  Rig BA mean reprojection error: 0.2461 px
    Camera 0: 0.2413 px
    Camera 1: 0.2509 px
  Rig baseline (after BA): |c1 - c0| = 0.1898 m (189.84 mm)

--- Uncertainty ---
  Baseline: 189.84 ± 0.489 mm
  Corner noise σ ≈ 0.188 px (6160 residuals)
```

The `±` comes from the export's `uncertainty`: the covariance of the estimated parameters, here of camera 1's `cam_se3_rig`. The rig solve refines both cameras' intrinsics with the rig, so it includes their uncertainty, and it is scaled by the view scatter, about 4 here: the error the views do not share with each other (see [Parameter Uncertainty](uncertainty.md)).

## The Session in Five Calls

A calibration is a session plus a sequence of step functions invoked in order. Each step reads what it needs from the session, writes its own output back, and returns a typed result, so you can inspect the state between any two steps:

```text
Session::new()
  → set_input(observations)
  → step_intrinsics_init_all      ← Zhang's method per camera
  → step_intrinsics_optimize_all  ← per-camera refinement
  → step_rig_init                 ← linear cam_se3_rig + rig_se3_target
  → step_rig_optimize             ← joint bundle adjustment
  → export()                      ← stable, JSON-serializable result
```

The interesting part of the example is ten lines:

```rust
let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
session.set_input(input)?;

step_intrinsics_init_all(&mut session, None)?;
step_intrinsics_optimize_all(&mut session, None)?;
step_rig_init(&mut session, None)?;
step_rig_optimize(&mut session, None)?;

let export = session.export()?;
let mean_reproj = export.mean_reproj_error;
let centre = |i: usize| export.cam_se3_rig[i].inverse().translation.vector;
let baseline = (centre(1) - centre(0)).norm();
```

`RigExtrinsicsProblem` is a marker type: through the `ProblemType` trait it fixes the input, configuration and export types, so the session is fully typed. Each `step_*` takes an `Option<*Options>` for per-call tweaks (`max_iters`, `verbosity`, ...); `None` uses the session's config. `export()` returns a `RigExtrinsicsExport`, the stable view of the result and the handoff to other tools.

The same calibration from Python is one call (`crates/vision-calibration-py/examples/stereo_session.py` loads the same observation file):

```python
import vision_calibration as vc

# `views`: one vc.RigExtrinsicsView per frame, with a vc.Observation (or None)
# per camera; `dataset` wraps them.
dataset = vc.RigExtrinsicsInput(num_cameras=2, views=views)
result = vc.run_rig_extrinsics(dataset)
print(result.mean_reproj_error, result.per_cam_reproj_errors)
```

Other camera setups use other problem types with the same pattern: only the input shape and the step list change. [Step Functions vs Pipeline Functions](step_functions.md) tabulates them, and each problem type's facade module (`vision_calibration::rig_extrinsics`, ...) re-exports its types and steps.

## Building the Rig Dataset

A rig view holds one optional observation per camera. A camera that does not see the target in a given frame contributes `None`; the calibration copes with such views as long as each camera has enough usable views overall (at least 3).

```rust
use vision_calibration::core::{NoMeta, RigDataset, RigView, RigViewObs};

views.push(RigView {
    meta: NoMeta,
    obs: RigViewObs {
        cameras: vec![left, right], // each an Option<CorrespondenceView>
    },
});

let input = RigDataset::new(views, 2)?; // 2 cameras
```

Each `CorrespondenceView` is built from the detected corners with `CorrespondenceView::new(points_3d, points_2d)?`, exactly as in [Planar Intrinsics with Real Data](planar_real_data.md).

## Reading the Step Results

Each step returns its own result struct; keep them to report progress:

```rust
use vision_calibration::rig_extrinsics::{
    RigExtrinsicsProblem, step_intrinsics_init_all, step_intrinsics_optimize_all,
    step_rig_init, step_rig_optimize,
};

step_intrinsics_init_all(&mut session, None)?;
let intr_opt = step_intrinsics_optimize_all(&mut session, None)?;
let rig_init = step_rig_init(&mut session, None)?;
let rig_opt = step_rig_optimize(&mut session, None)?;

println!("Rig BA mean reprojection error: {:.4} px", rig_opt.mean_reproj_error);
for (i, err) in rig_opt.per_cam_reproj_errors.iter().enumerate() {
    println!("  Camera {i}: {err:.4} px");
}
```

`run_calibration(&mut session)` (`vision_calibration::rig_extrinsics::run_calibration`) runs the same four steps with default options in one call.

Every step also appends an entry to the session log, which records what the initialization produced and which values came from manual seeds (see [Seeding with Prior Knowledge](manual_init.md)):

```rust
for entry in session.log() {
    println!("[{}] {:?} {}",
        entry.operation, entry.success, entry.notes.as_deref().unwrap_or(""));
}
```

`CalibrationSession` is JSON-serializable: persist it after `step_rig_init` (`session.to_json()`) and resume later with a fresh `step_rig_optimize` under different solver options (see [CalibrationSession](session.md)).

## The Example Dataset

The example reads `regression/observations/stereo_rig.json`: the corners of a 7×11 chessboard with 30 mm squares, detected in 20 image pairs of a 1024×576 stereo camera. The file is part of the [regression suite](regression_suite.md); its `input` field is a serialized `RigExtrinsicsInput`, so it loads with `serde_json`:

```rust
let text = std::fs::read_to_string("regression/observations/stereo_rig.json")?;
let mut file: serde_json::Value = serde_json::from_str(&text)?;
let input: RigExtrinsicsInput = serde_json::from_value(file["input"].take())?;
```

With your own images, detect the board in each camera's image of a pair and build the `RigView` as above. Pair the images by a shared index or timestamp; a camera whose detection fails contributes `None`. The detection step is the same as in [Planar Intrinsics with Real Data](planar_real_data.md). A ChArUco board identifies its corners by their markers, so a partly visible board still yields correspondences:

```rust
use calib_targets::aruco::builtins;
use calib_targets::charuco::{CharucoBoardSpec, CharucoParams, MarkerLayout};
use calib_targets::detect;

let board = CharucoBoardSpec::new(
    22,                       // rows
    22,                       // columns
    0.00135,                  // cell size, metres
    0.75,                     // marker size relative to a cell
    builtins::DICT_4X4_1000,  // dictionary
)
.with_marker_layout(MarkerLayout::OpenCvCharuco);
let charuco_params = CharucoParams::for_board(board);

let detection = detect::detect_charuco(&img, &charuco_params);
for corner in detection.corners {
    let target = corner.target_position; // board-frame position, metres
    points_3d.push(Pt3::new(target.x as f64, target.y as f64, 0.0));
    points_2d.push(Pt2::new(corner.position.x as f64, corner.position.y as f64));
}
```

## Command-Line Options

```bash
cargo run -p vision-calibration --example stereo_session --release -- --max-views=10
```

`--max-views=N` uses the first N views. The Python counterpart is `crates/vision-calibration-py/examples/stereo_session.py`.

## Interpreting Results

The example prints the number of views each camera sees, per-camera intrinsics with per-camera reprojection errors, the rig BA reprojection error (about 0.25 px on this dataset), and the baseline (about 189 mm).

- **Per-camera intrinsics**: cameras with the same lens should have similar focal lengths. Differences in principal point are normal.
- **Baseline**: the norm of the translation between camera centers should match the physical distance between the cameras. For a horizontal stereo pair, expect a translation primarily along X and a rotation primarily around Y.
- **Reprojection errors** should be within your application tolerance.

### Verification

- **Epipolar constraint**: corresponding points should lie on epipolar lines. Measure the average distance from points to their epipolar lines (see [Epipolar Geometry](epipolar.md)).
- **Rectification**: the pair should rectify cleanly, with horizontal scanlines in the rectified images corresponding.
- **Triangulation**: triangulate known board corners and verify the 3D distances match the board geometry (see [Linear Triangulation](triangulation.md)).

## Data Collection Tips

- **Both cameras should see the board** in most views; views seen by only one camera constrain only that camera's intrinsics.
- **Vary board orientation and distance** — same advice as single-camera calibration.
- **Ensure synchronization** — if images are not captured simultaneously, the board may have moved between left and right captures.
- **Overlap region** — the board should be in the overlapping field of view of both cameras.
