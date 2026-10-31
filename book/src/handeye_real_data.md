# Hand-Eye with KUKA Robot

This chapter walks through the `handeye_session` example, which performs eye-in-hand calibration using a real KUKA robot dataset.

## Dataset

Located at `data/kuka_1/`:

- **Images**: 30 PNG files (`01.png` through `30.png`)
- **Robot poses**: `RobotPosesVec.txt` — one row-major $4 \times 4$ homogeneous matrix per line (16 whitespace-separated values), each the pose $T_{B,G}$ of the gripper in the robot base frame, in metres. Image `NN.png` pairs with pose row `NN`.
- **Board square size**: `squaresize.txt` (`20mm`)
- **Calibration board**: 17×28 chessboard, 20 mm squares

## Data Loading

The example's support module (`examples/support/handeye_io.rs`) reads the square size and the robot poses, detects chessboard corners in each image with the `calib_targets` chessboard detector, and pairs each successful detection with its robot pose. Images where detection fails are skipped.

Each pose line is converted to an `Iso3`:

```rust
let r = Matrix3::new(
    values[0], values[1], values[2],
    values[4], values[5], values[6],
    values[8], values[9], values[10],
);
let t = Vector3::new(values[3], values[7], values[11]);
let rot = Rotation3::from_matrix_unchecked(r);
let pose = Iso3::from_parts(Translation3::from(t), UnitQuaternion::from_rotation_matrix(&rot));
```

The detected views become the calibration input:

```rust
use vision_calibration::single_cam_handeye::{
    HandeyeMeta, SingleCamHandeyeInput, SingleCamHandeyeProblem, SingleCamHandeyeView,
};

let views: Vec<SingleCamHandeyeView> = samples
    .into_iter()
    .map(|s| SingleCamHandeyeView {
        obs: s.view,                                          // CorrespondenceView
        meta: HandeyeMeta { base_se3_gripper: s.robot_pose }, // T_B_G
    })
    .collect();

let input = SingleCamHandeyeInput::new(views)?;
```

## Calibration

```rust
use vision_calibration::single_cam_handeye::{
    step_handeye_init, step_handeye_optimize, step_intrinsics_init, step_intrinsics_optimize,
};

let mut session = CalibrationSession::<SingleCamHandeyeProblem>::new();
session.set_input(input)?;

step_intrinsics_init(&mut session, None)?;
let intr_opt = step_intrinsics_optimize(&mut session, None)?;
step_handeye_init(&mut session, None)?;
let he_opt = step_handeye_optimize(&mut session, None)?;

let export = session.export()?;
```

## Running the Example

```bash
cargo run -p vision-calibration --example handeye_session
```

The example reports:

1. Dataset summary (total images, used views, skipped views)
2. Per-step results (initialization, optimization)
3. Final calibrated parameters (intrinsics, distortion, hand-eye transform, target pose)

Output on `data/kuka_1/` (30 chessboard views, eye-in-hand, the default mode), abridged:

```text
--- Step 2: Intrinsics Optimization ---
  Intrinsics: fx=2057.8, fy=2058.4, cx=962.1, cy=609.9
  Reprojection error: 0.1346 px

--- Step 3: Hand-Eye Initialization (Tsai-Lenz) ---
  Hand-eye |t|: 0.2806m

--- Step 4: Hand-Eye Optimization ---
  Final reprojection error: 0.1346 px

Robot pose refinements (30 views):
  Max rotation delta: 0.0030 rad (0.17°)
  Max translation delta: 0.0015m (1.53mm)
```

The joint hand-eye error matches the intrinsics-only error: every view is explained through one shared `gripper_se3_camera` and one shared `base_se3_target`, and the small per-view robot-pose corrections (0.17° and 1.5 mm at most) absorb the kinematic noise. A correction acts in the gripper frame, so its rotation and translation are the tool's orientation and position errors. A hand-eye error far above the intrinsics-only error points at robot-pose noise, mechanical play or a wrong pose convention.

## Interpreting Results

Key outputs of the export (eye-in-hand):

- **Hand-eye transform** (`gripper_se3_camera`): translation magnitude should match the physical camera-to-gripper distance. Rotation should reflect the mounting orientation.
- **Target in base** (`base_se3_target`): the calibration board's position in the robot base frame. Verify against the known physical setup.
- **Reprojection error** (`mean_reproj_error`): under 1 px indicates a good calibration; above 3 px suggests problems.

## Common Failure Modes

1. **Insufficient rotation diversity**: All robot poses rotate around the same axis (e.g., only wrist rotation). The Tsai-Lenz initialization will fail or produce a poor estimate.

2. **Incorrect pose convention**: Robot poses must be $T_{B,G}$ (`base_se3_gripper`). If the convention is inverted, the calibration will diverge.

3. **Mismatched corner ordering**: If the chessboard detector assigns corners in a different order than expected, the 2D-3D correspondences are wrong.

4. **Robot pose timestamps**: If images and robot poses are not synchronized, the calibration will fail.

## Data Collection Recommendations

- **Rotation diversity**: Include poses with significant roll, pitch, and yaw. Avoid pure translations or rotations around a single axis.
- **Target visibility**: Ensure the calibration board is fully visible in all images. Partial visibility causes corner detection failure.
- **Stable poses**: Take images when the robot is stationary. Motion blur degrades corner detection.
- **Coverage**: Vary the distance to the board and the position within the image.
