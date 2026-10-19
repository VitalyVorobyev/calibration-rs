# Stereo Rig with Real Data

This chapter walks through the `stereo_session` example, which calibrates a stereo camera rig from synchronized left/right views with `RigExtrinsicsProblem`.

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

## Running the 4-Step Pipeline

```rust
use vision_calibration::rig_extrinsics::{
    RigExtrinsicsProblem, step_intrinsics_init_all, step_intrinsics_optimize_all,
    step_rig_init, step_rig_optimize,
};

let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
session.set_input(input)?;

step_intrinsics_init_all(&mut session, None)?;
let intr_opt = step_intrinsics_optimize_all(&mut session, None)?;
let rig_init = step_rig_init(&mut session)?;
let rig_opt = step_rig_optimize(&mut session, None)?;

println!("Rig BA mean reprojection error: {:.4} px", rig_opt.mean_reproj_error);
for (i, err) in rig_opt.per_cam_reproj_errors.iter().enumerate() {
    println!("  Camera {i}: {err:.4} px");
}
```

`run_calibration(&mut session)` runs the same four steps in one call.

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

## Running

```bash
cargo run -p vision-calibration --example stereo_session --release
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
