# Stereo Rig with Real Data

This chapter walks through two examples that calibrate a stereo camera rig from synchronized left/right image pairs: `stereo_session` (chessboard) and `stereo_charuco_session` (ChArUco). Both feed `RigExtrinsicsProblem` and differ only in how the target is detected.

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

## Chessboard Example

### Dataset

Located at `data/stereo/imgs/`:

- **Left camera**: `leftcamera/Im_L_<index>.png`
- **Right camera**: `rightcamera/Im_R_<index>.png`
- **Pattern**: 7×11 chessboard, 30 mm square size

Left and right images are paired by their shared index (the sorted intersection of the two folders). The detector is the same `calib_targets` chessboard detector used in the planar example:

```rust
use calib_targets::chessboard::ChessboardParams;
use calib_targets::detect::{self, default_chess_config};

let board_params = ChessboardParams::default();
let img = image::ImageReader::open(path)?.decode()?.to_luma8();
let detection = detect::detect_chessboard(&img, &default_chess_config(), &board_params);
```

### Running

```bash
cargo run -p vision-calibration --example stereo_session
cargo run -p vision-calibration --example stereo_session -- --max-views=15
```

`--max-views=N` limits the number of image pairs processed.

## ChArUco Example

### Dataset

The example uses `data/stereo_charuco/`:

- `cam1/Cam1_*.png` for camera 0
- `cam2/Cam2_*.png` for camera 1
- Pairing by shared filename suffix (deterministic sorted intersection)

Non-PNG files (for example `Thumbs.db`) are ignored.

### Detector

A ChArUco board provides corner identities from its markers, so partially visible boards still yield correspondences. The board is described once:

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
```

Each detected corner carries its board-frame position, which becomes the 3D point on the $Z = 0$ plane:

```rust
for corner in detection.corners {
    let target = corner.target_position;
    points_3d.push(Pt3::new(target.x as f64, target.y as f64, 0.0));
    points_2d.push(Pt2::new(corner.position.x as f64, corner.position.y as f64));
}
```

Detections with fewer than 4 corners are discarded.

### Running

```bash
cargo run -p vision-calibration --example stereo_charuco_session
cargo run -p vision-calibration --example stereo_charuco_session -- --max-views=8
```

Default: all detected stereo pairs are used. `--max-views` caps the number of pairs (a deterministic prefix of the sorted pairs).

## Interpreting Results

Both examples print a dataset summary (`total_pairs`, `used_views`, `skipped_views`, `usable_left`, `usable_right`), per-camera intrinsics with per-camera reprojection errors, the rig BA reprojection error, and the baseline.

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
