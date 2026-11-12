# Dense Stereo

Once a pair is rectified, `mvg::dense::match_block` produces a per-pixel disparity map. It is a pure-Rust matcher: no OpenCV and no GPU.

## Block Matching

The matcher scores each pixel and each candidate disparity with the zero-mean normalized cross-correlation (ZNCC) over a square window, which tolerates a per-camera gain and bias. It selects the winner per pixel, refines it to sub-pixel precision with a parabola through the correlation peak, and filters with left-right consistency, a uniqueness check and a minimum correlation. Pixels that fail a check are `NaN` in the result.

```rust
use vision_calibration::mvg::dense::{match_block, BlockMatchOptions, GrayImage};

// `left` / `right` are rectified grayscale images: `rect.rectify_image(0, ..)?`
// and `rect.rectify_image(1, ..)?` of the raw pair (see Stereo from a Calibration).
let mut options = BlockMatchOptions::default();
options.min_disparity = 0;
options.num_disparities = 64; // search [min, min + num)
options.block_size = 9;       // odd; larger = smoother, fewer matches near edges
let disp = match_block(&left, &right, &options)?;
// disp.get(x, y) is x_left − x_right in pixels, or NaN where the matcher abstains.
```

```python
from vision_calibration import mvg

left = rect.rectify_image(0, raw_left, 640, 480)     # (H, W) uint8 arrays
right = rect.rectify_image(1, raw_right, 640, 480)
disp = mvg.match_block(left, right, mvg.BlockMatchOptions(num_disparities=64, block_size=9))
# disp is a (H, W) float32 array: x_left − x_right in pixels, NaN where the matcher abstains.
```

`GrayImage::new(width, height, data)` wraps row-major 8-bit pixels, and returns an error for an empty image or a buffer of the wrong length. The options and their defaults:

| Option | Default | Meaning |
|---|---|---|
| `min_disparity`, `num_disparities` | 0, 64 | the searched disparity range `[min, min + num)` |
| `block_size` | 7 | square window side in pixels; odd |
| `min_correlation` | 0.5 | reject a pixel whose best ZNCC is below this |
| `uniqueness_ratio` | 0.03 | reject a pixel unless its best ZNCC beats the best non-adjacent alternative by this margin |
| `lr_consistency`, `lr_max_diff` | on, 1.0 px | the left-right cross-check and its tolerance |
| `subpixel` | on | parabolic refinement of the peak |
| `semi_global`, `sgm_p1`, `sgm_p2` | off, 0.1, 1.5 | semi-global aggregation (below) |

`match_block` returns an `mvg::Error::InvalidInput` for images of different sizes, an even or zero `block_size`, a non-positive `num_disparities`, or images smaller than the window. The result is a `DisparityMap` with `get`, `is_valid`, `valid_count` and `valid_range`.

## Semi-Global Matching

For low-texture surfaces (the flat interiors of chessboard squares, for instance) set `semi_global: true`. The matching cost is then aggregated along 8 paths with the `sgm_p1` penalty for a one-pixel disparity step (small, so slanted surfaces survive) and the `sgm_p2` penalty for larger jumps (large, so depth discontinuities survive; `sgm_p2` must be at least `sgm_p1`). Disparity then propagates into regions block matching leaves blank: on a real chessboard rig coverage rose from about 21 % to about 49 %.

Semi-global matching is strictly additive: block matching stays the default. In SGM mode `min_correlation` and `uniqueness_ratio` (which gate on the raw per-pixel correlation) are bypassed; the left-right consistency check and the disparity window are the filters. It costs extra time and memory.

## Disparity to Depth

For a rectified pair with shared focal length $f$ (pixels) and baseline $b$, a disparity $d$ gives depth $Z = f\,b / d$. `StereoRectification` converts for you (`point`, the matrix `disparity_to_depth`, and the disparity's covariance; see [Stereo from a Calibration](mvg_rectification.md)). The desktop app's Depth workspace uses it and renders depth and a reprojected point cloud (see [Inspecting Results](app_inspect.md#depth)).

## Point Clouds with Uncertainty

`StereoRectification::dense_cloud` turns a whole disparity map into a `DenseCloud`: one point per pixel with a finite, positive disparity, in the **rig frame**, with the covariance of any point computed when you ask for it. The covariance has two sources: the matcher's disparity noise `sigma_d`, and the calibration's uncertainty, which every point of the cloud shares.

```rust
use vision_calibration::mvg::metrology::{PointSet, distance};

let cloud = rect.dense_cloud(&rig, &disp, 0.2)?;     // σ_d = 0.2 px
let p = &cloud.points()[i];                          // p.point, p.pixel = [x, y], p.disparity
let cov = cloud.point_covariance(i);                 // Option<3×3>, m²
// A measurement between cloud points takes their joint covariance:
let set = PointSet::from_dense_cloud(&cloud, &[i, j])?;
let length = distance(&set, 0, 1)?;                  // length.value, length.sigma
```

```python
cloud = rect.dense_cloud(result, disp, sigma_d=0.2)  # disp from mvg.match_block
cloud.points                                         # (N, 3), rig frame, metres
cloud.pixels                                         # (N, 2) rectified left pixels (x, y)
cov = cloud.point_covariance(i)                      # (3, 3), m², or None
points = mvg.PointSet.from_dense_cloud(cloud, [i, j])
length = mvg.distance(points, 0, 1)
```

**The measurement.** The rectified images are resampled with the calibration's values, so the left pixel of a match is a grid point of a fixed raw pixel, and the disparity says where along the row of the rectified right image the matcher found it. If the calibration is off, the true point lies on the true left ray through that raw pixel, at the depth where its true right projection falls on that rectified column; the row it falls on there (a vertical disparity) is not measured. The covariance is the first-order propagation of this model, with the cameras' exact derivatives through their own models (distortion, fisheye and tilted sensors included):

$$\Sigma_i = G_i\,\Sigma_c\,G_i^\top + \sigma_d^2\,g_i g_i^\top,$$

with $G_i$ the derivative of point $i$ by the calibration's parameters (both cameras' intrinsics, distortion and sensor, and their extrinsics), $\Sigma_c$ their covariance from the calibration and $g_i$ the derivative by the disparity (along the left viewing ray). Parameters the calibration held fixed count as exact. Two points share $\Sigma_c$, so they are correlated by $G_i \Sigma_c G_j^\top$; the disparity errors are taken as independent.

**On demand.** Building the cloud stores only the points (a 640 × 480 map in a few milliseconds). Each covariance evaluates both cameras' derivatives at the point, about 15 µs per point on one laptop core, and `covariance(&indices)` (Python: `cloud.covariance(indices)`) gives the joint $3k \times 3k$ covariance of $k$ chosen points. Storing every $G_i$ of a full map would take hundreds of megabytes; choose the points a measurement needs instead.

**The calibration's share matters.** On a stereo rig calibrated 200 times from noisy board views, with ten pixels of a held-out scene 0.6 to 0.8 m away measured at $\sigma_d = 0.2$ px, the depth error over the stated σ is 0.91 (2.6 mm). The disparity's share alone (1.0 mm) is 2.5 times too small: on a short baseline at short range, the calibration is most of the error.

For the same matches, the depth σ agrees with `triangulate`'s within 1 %; it differs across the rows, where `triangulate` takes both raw pixels as measured and shares a vertical disparity between the two cameras, while the matcher measures none.

A rig without a covariance gives a cloud whose points have none: `point_covariance` is `None`, `covariance` and `PointSet::from_dense_cloud` return an `InvalidInput` error with the reason, and `covariance_unavailable()` says `CalibrationWithoutCovariance`. The disparity's share alone, if that is what you want, is `disparity_covariance`. `dense_cloud` returns `InvalidInput` for a rig other than the one rectified, a `sigma_d` that is negative or not finite, or a covariance that lists part of a camera's extrinsic.

## Try It

A runnable demo writes inspectable PNGs (left, right, ground truth, estimate and error) to `target/fixtures/`:

```bash
cargo run -p vision-calibration-bench --example dense_synth --features tier-b --release
```
