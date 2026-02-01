# Hand-Eye with KUKA Robot

> **[COLLAB]** This chapter requires user collaboration for KUKA dataset details, data collection protocol, and practical tips.

This chapter walks through the `handeye_session` example, which performs hand-eye calibration using a real KUKA robot dataset.

## Dataset

Located at `data/kuka_1/`:

- **Images**: 30 PNG files (`01.png` through `30.png`)
- **Robot poses**: `RobotPosesVec.txt` — 30 poses in $4 \times 4$ matrix format
- **Calibration board**: 17×28 chessboard, 20 mm square size

<!-- [COLLAB]: Describe the KUKA robot model, camera specifications, mounting configuration (eye-in-hand), and data collection procedure -->

## Data Loading

The example loads images and robot poses, detects chessboard corners, and constructs the input:

```rust
// Load robot poses from 4×4 matrix format
let robot_poses: Vec<Iso3> = load_robot_poses("data/kuka_1/RobotPosesVec.txt")?;

// Detect corners per image
let views: Vec<SingleCamHandeyeView> = images
    .iter()
    .zip(&robot_poses)
    .filter_map(|(img, pose)| {
        let corners = detect_chessboard(img, &config)?;
        Some(SingleCamHandeyeView {
            obs: CorrespondenceView::new(board_3d.clone(), corners),
            meta: HandeyeMeta { base_se3_gripper: *pose },
        })
    })
    .collect();
```

Typically 20-25 of the 30 images yield successful corner detections.

## Running the Example

```bash
cargo run -p vision-calibration --example handeye_session
```

The example reports:

1. Dataset summary (total images, used views, skipped views)
2. Per-step results (initialization, optimization)
3. Final calibrated parameters (intrinsics, distortion, hand-eye transform, target pose)
4. Robot pose refinement deltas (if enabled)

## Interpreting Results

<!-- [COLLAB]: Add actual calibration results from the KUKA dataset -->

Key outputs:

- **Hand-eye transform** ($T_{G,C}$): Translation magnitude should match the physical camera-to-gripper distance. Rotation should reflect the mounting orientation.
- **Target-in-base** ($T_{B,T}$): The calibration board's position in the robot base frame. Verify against the known physical setup.
- **Reprojection error**: <1 px indicates a good calibration. >3 px suggests problems.
- **Robot pose deltas**: If enabled, large deltas (>1° rotation, >5mm translation) suggest robot kinematic inaccuracies.

## Common Failure Modes

<!-- [COLLAB]: Expand with real-world experience -->

1. **Insufficient rotation diversity**: All robot poses rotate around the same axis (e.g., only wrist rotation). The Tsai-Lenz initialization will fail or produce a poor estimate.

2. **Incorrect pose convention**: Robot poses must be $T_{B,G}$ (base-to-gripper). If the convention is inverted, the calibration will diverge.

3. **Mismatched corner ordering**: If the chessboard detector assigns corners in a different order than expected, the 2D-3D correspondences are wrong.

4. **Robot pose timestamps**: If images and robot poses are not synchronized, the calibration will fail.

## Data Collection Recommendations

<!-- [COLLAB]: Provide detailed data collection advice based on KUKA experience -->

- **Rotation diversity**: Include poses with significant roll, pitch, and yaw. Avoid pure translations or rotations around a single axis.
- **Target visibility**: Ensure the calibration board is fully visible in all images. Partial visibility causes corner detection failure.
- **Stable poses**: Take images when the robot is stationary. Motion blur degrades corner detection.
- **Coverage**: Vary the distance to the board and the position within the image.
