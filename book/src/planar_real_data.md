# Planar Intrinsics with Real Data

This chapter walks through the `planar_real` example, which calibrates a camera from real chessboard images.

## Dataset

The example uses the images in `data/kuka_1/`:

- **Pattern**: 17×28 chessboard, 20 mm squares (the size is read from `squaresize.txt`)
- **Images**: 30 PNG files `01.png` … `30.png`, 1928×1208 px

The images are stored with Git LFS. If they are small text pointer files in your checkout, run `git lfs pull` first.

## Workflow

### 1. Corner Detection

The example detects corners with the `calib_targets` crate (a dev-dependency of the examples; the calibration library itself takes plain point correspondences). Each image is processed independently; failed detections are skipped and the workflow continues with the views that succeed.

```rust
use calib_targets::chessboard::{ChessboardDetection, ChessboardParams};
use calib_targets::detect::{self, default_chess_config};

const SQUARE_SIZE_M: f64 = 0.02; // 20 mm squares

let board_params = ChessboardParams::default();

let img = image::ImageReader::open(path)?.decode()?.to_luma8();
let Ok(detection) = detect::detect_chessboard(&img, &default_chess_config(), &board_params)
else {
    return Ok(None); // no board found in this image
};
```

### 2. Correspondence Construction

Each detected corner carries its grid index (`corner.grid.u`, `corner.grid.v`) and its pixel position. The board lies in the plane $Z = 0$, so the 3D point follows directly from the grid index and the square size:

```rust
fn detection_to_view(detection: ChessboardDetection) -> Result<CorrespondenceView> {
    let mut points_3d = Vec::new();
    let mut points_2d = Vec::new();

    for corner in detection.corners {
        let grid = corner.grid;
        points_3d.push(Pt3::new(
            grid.u as f64 * SQUARE_SIZE_M,
            grid.v as f64 * SQUARE_SIZE_M,
            0.0,
        ));
        points_2d.push(Pt2::new(corner.position.x as f64, corner.position.y as f64));
    }

    Ok(CorrespondenceView::new(points_3d, points_2d)?)
}
```

`CorrespondenceView::new` validates that both point lists have the same length.

### 3. Dataset and Calibration

The remaining pipeline is identical to the synthetic case:

```rust
let dataset = PlanarDataset::new(views.into_iter().map(View::without_meta).collect())?;

let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
session.set_input(dataset)?;
let init = step_init(&mut session, None)?;
let opt = step_optimize(&mut session, None)?;
let export = session.export()?;

let k = export.params.intrinsics();
println!("fx={:.2} fy={:.2} cx={:.2} cy={:.2}", k.fx, k.fy, k.cx, k.cy);
println!("Mean reprojection error: {:.4} px", export.mean_reproj_error);
```

### 4. Parameter Uncertainty

The export carries the covariance of the estimated parameters (see [Parameter Uncertainty](uncertainty.md)):

```rust
use vision_calibration::core::Uncertainty;

if let Uncertainty::Available(cov) = &export.uncertainty {
    println!("fx = {:.2} ± {:.2} px", k.fx, cov.std_dev("fx").unwrap());
    println!("corr(fx, cx) = {:+.3}", cov.correlation("fx", "cx").unwrap());
}
```

On this dataset the example reports a mean reprojection error of about 0.13 px and `fx` ≈ 2058 px with a standard deviation of about 0.3 px.

### 5. Outlier Filtering

Real data usually contains a few poorly detected corners. `run_calibration_with_filtering` solves, removes observations above a reprojection-error threshold, and solves again:

```rust
use vision_calibration::planar_intrinsics::{IntrinsicsFilterOptions, run_calibration_with_filtering};

let mut filter_opts = IntrinsicsFilterOptions::default();
filter_opts.max_reproj_error = 1.0;
filter_opts.min_points_per_view = 10;
filter_opts.remove_sparse_views = true;
run_calibration_with_filtering(&mut session, Some(filter_opts))?;
```

## Interpreting Results

Key outputs to examine:

- **Focal lengths** ($f_x$, $f_y$): Should match the expected value based on sensor size and lens focal length.
- **Principal point** ($c_x$, $c_y$): Should be near the image center. Large offsets may indicate a decentered lens.
- **Distortion** ($k_1$, $k_2$): Negative $k_1$ indicates barrel distortion (common). $|k_1| > 0.3$ suggests a wide-angle lens.
- **Reprojection error**: <1 px is good. >2 px suggests problems with corner detection or insufficient view diversity.

## Comparison with Synthetic Data

| Aspect | Synthetic | Real |
|--------|-----------|------|
| Corner accuracy | Exact (no detection noise) | ~0.1-0.5 px (detector dependent) |
| Distortion | Known ground truth | Unknown, estimated |
| View coverage | Controlled | May have gaps |
| Typical reprojection error | <0.01 px | 0.1-1.0 px |

## Practical Tips

- **Use 15-30 images** with diverse viewpoints
- **Cover the full image area** — points only near the center poorly constrain distortion
- **Include tilted views** — not just frontal. Rotation around both axes constrains all intrinsic parameters
- **Check corner ordering** — mismatched 2D-3D correspondences cause calibration failure
- **Keep `fix_k3` on (the default)** — only estimate $k_3$ if reprojection error is high and you suspect higher-order radial distortion

## Running the Example

```bash
cargo run -p vision-calibration --example planar_real --release
```

The example prints per-step results including initialization accuracy, optimization convergence, the final calibrated parameters and their standard deviations. Its Python counterpart is `crates/vision-calibration-py/examples/planar_real.py`.
