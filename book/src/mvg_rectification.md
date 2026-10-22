# Stereo Rectification

`rectify_stereo_pair` returns rectifying homographies that put corresponding points on the **same image row**: the precondition for one-dimensional disparity search. It handles Scheimpflug (tilted-sensor) cameras as well as plain pinholes.

## Usage

```rust
use vision_calibration::mvg::rectification::{rectify_stereo_pair, RectifyCamera, RectifyOptions};

let rect = rectify_stereo_pair(
    &RectifyCamera::pinhole(k0),     // or RectifyCamera::scheimpflug(k0, tilt0)
    &RectifyCamera::pinhole(k1),
    &cam1_se3_cam0,                  // the calibrated relative pose T_C1_C0
    &RectifyOptions::default(),
)?;
let p_left  = rect.rectify_left(&undistorted_pixel0);
let p_right = rect.rectify_right(&undistorted_pixel1);
// p_left.y == p_right.y for a corresponding pair
```

The inputs are each camera's intrinsic matrix `k` (and, for a tilted sensor, its `ScheimpflugParams`) and the relative pose `cam1_se3_cam0`. Compose the pose from a rig calibration as shown in [From a Calibration to MVG Inputs](mvg.md#from-a-calibration-to-mvg-inputs).

The result is a `StereoRectification`:

| Field | Meaning |
|---|---|
| `h_left`, `h_right` | homographies mapping **undistorted** left and right pixels to rectified pixels |
| `k_rect` | the shared rectified intrinsics |
| `r_rect` | the common rectified orientation |
| `baseline` | the distance between the optical centres, in the units of the pose translation |

`rectify_left` and `rectify_right` apply the homographies to single pixels. Both rectified cameras share `k_rect` so that rows align; by default it averages the two cameras' focal lengths and principal points with zero skew, or you can pass your own through `RectifyOptions::k_rect`.

## Undistorted Pixels

The inputs to the homographies are **undistorted** pixels. Remove the lens distortion first: the rectifying homography is the rotation part of rectification, exactly as OpenCV splits `initUndistortRectifyMap`. To rectify a whole image, resample it through the homography after undistorting it, which is also the input of [dense matching](mvg_dense.md).

## Scheimpflug Rigs

For a tilted sensor, pass `RectifyCamera::scheimpflug(k, tilt)` for that camera. The tilt is absorbed into the rectifying homography, so a rig of tilted cameras rectifies the same way as a pinhole rig. The tilt (`tilt_x`, `tilt_y`) is the one the calibration estimated for that camera's sensor.

## Failure Modes

`rectify_stereo_pair` fails with a typed error rather than returning a meaningless homography: a zero baseline (coincident optical centres) or a tilt beyond about 80 degrees, where the sensor homography degenerates, is `MvgError::Degenerate`; a non-invertible intrinsic matrix is `MvgError::Numerical`.
