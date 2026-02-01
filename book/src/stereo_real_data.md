# Stereo Rig with Real Data

> **[COLLAB]** This chapter requires user collaboration for dataset-specific details and verification strategies.

This chapter walks through the `stereo_session` example, which calibrates a stereo camera rig from synchronized left/right image pairs.

## Dataset

Located at `data/stereo/imgs/`:

- **Left camera**: `leftcamera/Im_L_*.png`
- **Right camera**: `rightcamera/Im_R_*.png`
- **Pattern**: 7×11 chessboard, 30 mm square size
- **Synchronized**: Left and right images are captured simultaneously

<!-- [COLLAB]: Describe camera hardware, baseline, lens specifications, and data collection setup -->

## Workflow

### 1. Load Stereo Pairs

The example loads images from both camera directories and pairs them by filename:

```rust
let board = ChessboardParams { rows: 7, cols: 11, square_size: 0.03 };
let views: Vec<RigView<NoMeta>> = image_pairs
    .iter()
    .filter_map(|(left_img, right_img)| {
        let left_corners = detect_chessboard(left_img, &config)?;
        let right_corners = detect_chessboard(right_img, &config)?;
        // Both cameras must detect corners for a valid view
        Some(RigView::new(vec![
            Some(CorrespondenceView::new(board_3d.clone(), left_corners)),
            Some(CorrespondenceView::new(board_3d.clone(), right_corners)),
        ]))
    })
    .collect();
```

### 2. Run 4-Step Pipeline

```rust
let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
session.set_input(RigDataset::new(views, 2)?)?;

step_intrinsics_init_all(&mut session, None)?;
step_intrinsics_optimize_all(&mut session, None)?;
step_rig_init(&mut session)?;
step_rig_optimize(&mut session, None)?;
```

## Interpreting Results

<!-- [COLLAB]: Add actual calibration results from the stereo dataset -->

### Per-Camera Intrinsics

Both cameras should have similar focal lengths (if they use the same lens). Differences in principal point are normal.

### Stereo Baseline

The extrinsics give the relative pose between cameras. For a horizontal stereo pair, expect:

- Rotation: primarily around the Y axis (converging or parallel configuration)
- Translation: primarily along X (horizontal baseline)
- Baseline magnitude: should match the physical distance between cameras

### Verification

<!-- [COLLAB]: Describe verification strategies -->

- **Epipolar constraint**: For correctly calibrated stereo, corresponding points should lie on epipolar lines. Measure the average distance from points to their epipolar lines.
- **Rectification**: The stereo pair should rectify cleanly — horizontal scanlines in the rectified images should correspond.
- **Triangulation**: Triangulate known board corners and verify the 3D distances match the board geometry.

## Running the Example

```bash
cargo run -p vision-calibration --example stereo_session
cargo run -p vision-calibration --example stereo_session -- --max-views 15
```

The `--max-views` flag limits the number of image pairs processed, useful for faster iteration during development.

## Data Collection Tips

<!-- [COLLAB]: Expand with practical advice -->

- **Both cameras must see the board** in each view. Discard pairs where only one camera detects corners.
- **Vary board orientation and distance** — same advice as single-camera calibration.
- **Ensure synchronization** — if images are not captured simultaneously, the board may have moved between left and right captures.
- **Overlap region** — the board should be in the overlapping field of view of both cameras.
