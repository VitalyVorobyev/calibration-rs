# Planar Intrinsics with Real Data

> **[COLLAB]** This chapter requires user collaboration for dataset-specific details, data collection advice, and practical tips.

This chapter walks through the `planar_real` example, which calibrates a camera from real chessboard images.

## Dataset

The example uses images from `data/stereo/imgs/leftcamera/`:

- **Pattern**: 7×11 chessboard, 30 mm square size
- **Images**: Multiple PNG files (`Im_L_*.png`)
- **Camera**: Standard industrial camera (specific model TBD)

<!-- [COLLAB]: Describe the camera hardware, lens, resolution, and data collection setup -->

## Workflow

### 1. Corner Detection

The example uses the `chess-corners` crate for chessboard detection:

```rust
use chess_corners::{ChessboardParams, ChessConfig, detect_chessboard};

let board_params = ChessboardParams {
    rows: 7,
    cols: 11,
    square_size: 0.03, // 30 mm
};
```

Each image is processed independently. Failed detections are skipped — the workflow continues with the views that succeed.

<!-- [COLLAB]: Discuss corner detection reliability, common failure modes, lighting conditions, and recommended practices for data collection -->

### 2. Dataset Construction

Detected corners are assembled into `CorrespondenceView` structures with matched 3D board points:

```rust
let views: Vec<View<NoMeta>> = images
    .iter()
    .filter_map(|img| {
        let corners = detect_chessboard(img, &config)?;
        let obs = CorrespondenceView::new(board_3d.clone(), corners);
        Some(View::without_meta(obs))
    })
    .collect();
let dataset = PlanarDataset::new(views)?;
```

### 3. Calibration

The pipeline is identical to the synthetic case:

```rust
let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
session.set_input(dataset)?;
step_init(&mut session, None)?;
step_optimize(&mut session, None)?;
let export = session.export()?;
```

## Interpreting Results

Key outputs to examine:

- **Focal lengths** ($f_x$, $f_y$): Should match the expected value based on sensor size and lens focal length. For a 1/2" sensor with 8mm lens: $f \approx 800$ pixels.
- **Principal point** ($c_x$, $c_y$): Should be near the image center. Large offsets may indicate a decentered lens.
- **Distortion** ($k_1$, $k_2$): Negative $k_1$ indicates barrel distortion (common). $|k_1| > 0.3$ suggests a wide-angle lens.
- **Reprojection error**: <1 px is good. >2 px suggests problems with corner detection or insufficient view diversity.

<!-- [COLLAB]: Add actual results from the stereo dataset, discuss expected values, and show example images with detected corners -->

## Comparison with Synthetic Data

| Aspect | Synthetic | Real |
|--------|-----------|------|
| Corner accuracy | Exact (no detection noise) | ~0.1-0.5 px (detector dependent) |
| Distortion | Known ground truth | Unknown, estimated |
| View coverage | Controlled | May have gaps |
| Typical reprojection error | <0.01 px | 0.1-1.0 px |

## Practical Tips

<!-- [COLLAB]: Expand with real-world advice -->

- **Use 15-30 images** with diverse viewpoints
- **Cover the full image area** — points only near the center poorly constrain distortion
- **Include tilted views** — not just frontal. Rotation around both axes constrains all intrinsic parameters
- **Check corner ordering** — mismatched 2D-3D correspondences cause calibration failure
- **Start with `fix_k3: true`** — only estimate $k_3$ if reprojection error is high and you suspect higher-order radial distortion

## Running the Example

```bash
cargo run -p vision-calibration --example planar_real
```

The example prints per-step results including initialization accuracy, optimization convergence, and final calibrated parameters.
