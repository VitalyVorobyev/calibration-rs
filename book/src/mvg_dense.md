# Dense Stereo

Once a pair is rectified, `mvg::dense::match_block` produces a per-pixel disparity map. It is a pure-Rust matcher: no OpenCV and no GPU.

## Block Matching

The matcher scores each pixel and each candidate disparity with the zero-mean normalized cross-correlation (ZNCC) over a square window, which tolerates a per-camera gain and bias. It selects the winner per pixel, refines it to sub-pixel precision with a parabola through the correlation peak, and filters with left-right consistency, a uniqueness check and a minimum correlation. Pixels that fail a check are `NaN` in the result.

```rust
use vision_calibration::mvg::dense::{match_block, BlockMatchOptions, GrayImage};

// `left` / `right` are rectified grayscale images: resample each source image
// through its rectifying homography first.
let disp = match_block(&left, &right, &BlockMatchOptions {
    min_disparity: 0,
    num_disparities: 64,   // search [min, min + num)
    block_size: 9,         // odd; larger = smoother, fewer matches near edges
    ..Default::default()
})?;
// disp.get(x, y) is x_left − x_right in pixels, or NaN where the matcher abstains.
```

`GrayImage::new(width, height, data)` wraps row-major 8-bit pixels. The options and their defaults:

| Option | Default | Meaning |
|---|---|---|
| `min_disparity`, `num_disparities` | 0, 64 | the searched disparity range `[min, min + num)` |
| `block_size` | 7 | square window side in pixels; odd |
| `min_correlation` | 0.5 | reject a pixel whose best ZNCC is below this |
| `uniqueness_ratio` | 0.03 | reject a pixel unless its best ZNCC beats the best non-adjacent alternative by this margin |
| `lr_consistency`, `lr_max_diff` | on, 1.0 px | the left-right cross-check and its tolerance |
| `subpixel` | on | parabolic refinement of the peak |
| `semi_global`, `sgm_p1`, `sgm_p2` | off, 0.1, 1.5 | semi-global aggregation (below) |

`match_block` returns an `MvgError::InvalidInput` for images of different sizes, an even or zero `block_size`, a non-positive `num_disparities`, or images smaller than the window. The result is a `DisparityMap` with `get`, `is_valid`, `valid_count` and `valid_range`.

## Semi-Global Matching

For low-texture surfaces (the flat interiors of chessboard squares, for instance) set `semi_global: true`. The matching cost is then aggregated along 8 paths with the `sgm_p1` penalty for a one-pixel disparity step (small, so slanted surfaces survive) and the `sgm_p2` penalty for larger jumps (large, so depth discontinuities survive; `sgm_p2` must be at least `sgm_p1`). Disparity then propagates into regions block matching leaves blank: on a real chessboard rig coverage rose from about 21 % to about 49 %.

Semi-global matching is strictly additive: block matching stays the default. In SGM mode `min_correlation` and `uniqueness_ratio` (which gate on the raw per-pixel correlation) are bypassed; the left-right consistency check and the disparity window are the filters. It costs extra time and memory.

## Disparity to Depth

For a rectified pair with shared focal length $f$ (pixels) and baseline $b$, a disparity $d$ gives depth $Z = f\,b / d$. `StereoRectification` carries `k_rect` (with $f$) and `baseline`. The desktop app's Depth workspace performs this conversion and renders depth and a reprojected point cloud (see [Inspecting Results](app_inspect.md#depth)).

## Try It

A runnable demo writes inspectable PNGs (left, right, ground truth, estimate and error) to `target/fixtures/`:

```bash
cargo run -p vision-calibration-bench --example dense_synth --features tier-b --release
```
