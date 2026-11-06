# Stereo from a Calibration

`mvg::rectification::rectify` turns two cameras of a `CalibratedRig` into a **rectified pair**: two virtual pinhole cameras that share one set of intrinsics and an orientation whose x axis is the baseline, so corresponding points lie on the **same image row**. That is the precondition for one-dimensional disparity search ([Dense Stereo](mvg_dense.md)), and the pair then turns disparities into 3D points and states how uncertain they are.

The maps go through each camera's own unprojection and projection, so **raw pixels go in** and every camera model works: lens distortion, a fisheye, a tilted (Scheimpflug) sensor. There is no separate undistortion step and no model-specific case.

## Usage

```rust
use vision_calibration::mvg::rectification::{RectifyOptions, rectify};

let rect = rectify(&rig, [0, 1], &RectifyOptions::default())?;

// A raw pixel of each camera, and the rectified pixels they map to:
let left = rect.rectified_pixel(0, &raw_left).unwrap();
let right = rect.rectified_pixel(1, &raw_right).unwrap();
// left.y == right.y for a corresponding pair, and
let disparity = left.x - right.x; // positive for a point ahead of the pair
```

```python
from vision_calibration import mvg

rect = mvg.rectify(result, (0, 1))               # k_rect=..., scale=... as RectifyOptions
left = rect.rectified_pixel(0, raw_left)         # (x, y), or None
right = rect.rectified_pixel(1, raw_right)
disparity = left[0] - right[0]                   # positive for a point ahead of the pair
point = rect.point(left, disparity)              # (3,) in the left camera's frame
cov = rect.disparity_covariance(left, disparity, sigma_d=0.2)   # (3, 3), m²
q = rect.disparity_to_depth()                    # (4, 4)
rectified = rect.rectify_image(0, raw_image, 640, 480)   # (480, 640) uint8
found = rect.triangulate(result, [(left, disparity)], noise)  # with the calibration's σ
```

`StereoRectification` is an opaque object in Python: it is rebuilt from a calibration with `mvg.rectify`, not saved. Images are `(H, W)` uint8 numpy arrays.

`cameras = [left, right]` names two different cameras of the rig (any pair: the reference camera need not be one of them). `RectifyOptions` has two settings. By default the rectified intrinsics are the mean of the two cameras' `fx`, `fy`, `cx`, `cy`, multiplied by `scale` (1 by default; 0.5 gives a half-size rectified image); `k_rect` sets them explicitly. Both rectified cameras share them, which is what aligns the rows.

| Method | Meaning |
|---|---|
| `rectified_pixel(side, raw)` | the rectified pixel of a raw pixel (`side` 0 is the left camera, 1 the right); `None` when its ray points behind the rectified image plane |
| `raw_pixel(side, rectified)` | the raw pixel a rectified pixel comes from; `None` outside the camera's projection domain |
| `rectify_image(side, src, width, height)` | the rectified image, sampled bilinearly from the raw one; 0 where the source has no pixel |
| `k_rect()`, `baseline()`, `rect_r_cam(side)` | the rectified intrinsics, the distance between the camera centres (metres for a calibration in metres) and the rotation into the rectified frame |

`rectify_image` takes the output size freely, so a rectification with `scale: 0.5` of a full-size source fills a half-size image. It samples the source rather than filtering it: reduce a large source first, or keep the reduction small, to avoid aliasing.

## Disparity to Depth

On the rectified pair a match is a left pixel $(x, y)$ and a disparity $d = x_\text{left} - x_\text{right} > 0$. With baseline $B$ and rectified focal length $f_x$, the depth is $Z = f_x B / d$. `disparity_to_depth()` returns the $4\times4$ matrix $Q$ (OpenCV's convention) that does this for a whole map: $Q\,[x, y, d, 1]^\top$ is the homogeneous point in the rectified left frame.

`point(&left, d)` gives the same point in the **left camera's frame** (`cameras[0]`), or `None` for $d \le 0$:

```rust
let point = rect.point(&Pt2::new(x, y), disparity).unwrap();
```

## Disparity Uncertainty

A disparity error $\delta d$ moves the point along its viewing ray, $\delta P = -P\,\delta d / d$, with the rectified row and column exact. `disparity_covariance(&left, d, sigma_d)` returns the covariance of `point` that a disparity noise of `sigma_d` pixels alone gives it, in the left camera's frame:

$$\Sigma = \left(\frac{\sigma_d}{d}\right)^2 P P^\top,\qquad \sigma_Z = \frac{Z^2\,\sigma_d}{f_x B}.$$

It has rank 1, along the ray, and it grows with the square of the depth. Over 2000 draws of the disparity noise the scatter of the depth over this $\sigma_Z$ is 0.97 to 1.06 at 0.7, 1.5 and 3 m, which is the sampling error of the draws. It does **not** include the calibration's uncertainty; a dense point cloud with it is `dense_cloud` ([Dense Stereo](mvg_dense.md#point-clouds-with-uncertainty)).

## Measurements with the Calibration's Uncertainty

The calibration is itself uncertain, and every point measured with it shares that error. For matches you select (corners, a few points to measure), `triangulate` includes it:

```rust
use vision_calibration::control::RunControl;
use vision_calibration::core::PointCovariance;
use vision_calibration::mvg::rectification::RectifiedMatch;

let matches = [RectifiedMatch::new(Pt2::new(x, y), disparity)];
let noise = PointCovariance::isotropic(0.3)?; // each raw pixel's σ
let found = rect.triangulate(&rig, &matches, noise, &RunControl::default())?;
let covariance = found.point_covariance(0); // 3×3, m², the calibration's share included
```

Each match is mapped back to the two cameras' raw pixels and triangulated as one stereo capture with [`triangulate`](mvg_pose.md): refined on the reprojection error of the stated noise, the rig's calibration entering as consider parameters. The points are in the **rig frame** (the left camera's frame only when it is the reference camera), and the covariance is joint over the points, so `cross_covariance(i, j)` gives the correlation the shared calibration causes. `rig` must be the rig the rectification was built from.

Measured on a stereo rig calibrated 200 times from noisy board views spread over the image, with ten held-out points measured from freshly noisy pixels, the scatter of the depth over the stated depth σ is **0.97** with the calibration's share (3.0 mm). The σ from the observations alone (2.1 mm) is 1.46 times too small, and so is the disparity covariance of the previous section, because the calibration is a large part of the error of a short baseline at a short range (with centred, barely tilted calibration views, 3.2 times). A disparity covariance is the right σ for the matcher's noise alone; `triangulate`, and `dense_cloud` for a whole disparity map, are the ones for a measurement.

## Failure Modes

`rectify` fails with a typed error before any work: the same camera twice, a camera out of range, or a `scale` or `k_rect` that is not finite and positive is `MvgError::InvalidInput`; coincident camera centres (a zero baseline) is `MvgError::Degenerate`. `triangulate` returns `InvalidInput` for a rig other than the one rectified, or for a match whose rectified pixel has no raw pixel in a camera.

A baseline along the optical axis (forward stereo) is handled, but the rectified field of view degrades as the baseline leaves the image plane.
