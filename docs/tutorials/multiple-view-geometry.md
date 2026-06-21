# Multiple-view geometry

> Onboarding tutorial. Runnable companion:
> [`mvg_two_view.rs`](../../crates/vision-calibration/examples/mvg_two_view.rs).
> Run it with `cargo run -p vision-calibration --example mvg_two_view --features refine`.

## Why

Calibration tells you *what each camera is* (intrinsics, distortion, pose).
**Multiple-view geometry (MVG)** is what you do *with* a calibrated rig:
recover the relative pose between two views, triangulate 3-D structure, refine
both with bundle adjustment, and rectify a stereo pair for dense matching. This
tutorial is the tour of that surface, now reachable through the facade as
`vision_calibration::mvg`.

It is aimed at someone who already has calibrated cameras (e.g. from the
[five-minute calibration](./five-minute-calibration.md)) and wants to reconstruct
or rectify.

## Mental model

Everything in `mvg` operates on **calibrated** observations. Two coordinate
conventions matter:

- **Normalized image coordinates** — pixels after applying `K⁻¹`. Relative-pose
  recovery and triangulation work here, so geometry is independent of the
  specific intrinsics.
- **Pixel coordinates** — what rectification consumes (it folds `K` back in).

Poses follow the project convention (ADR 0009): `T_C1_C0` maps a point from
camera 0's frame into camera 1's frame (`p_c1 = R · p_c0 + t`).

The surface, in dependency order:

```
mvg::pose_recovery   recover_relative_pose  ← 5-point + cheirality (also triangulates)
mvg::triangulation   triangulate_nview      ← N-view DLT + refinement
mvg::bundle_adjust   bundle_adjust          ← joint poses + structure (feature `refine`)
mvg::rectification   rectify_stereo_pair    ← rectifying homographies (Scheimpflug-aware)
mvg::robust          *_robust               ← RANSAC wrappers for outlier-laden data
```

## Walkthrough

The companion example builds a synthetic calibrated stereo pair with known
ground truth and runs the whole pipeline. The key steps:

### 1. Recover the relative pose (and triangulate)

Given calibrated correspondences in **normalized** coordinates,
`recover_relative_pose` runs the 5-point algorithm, disambiguates the four
essential-matrix decompositions by cheirality (positive depth), and triangulates
the inliers:

```rust
use vision_calibration::mvg::pose_recovery::recover_relative_pose;
use vision_calibration::mvg::types::Correspondence2D;

let corrs: Vec<Correspondence2D> = /* (normalized pt in cam0, normalized pt in cam1) */;
let rel = recover_relative_pose(&corrs)?;
// rel.r : rotation cam0 → cam1
// rel.t : unit translation direction (absolute scale is unobservable from two views)
// rel.points : triangulated 3-D points with reprojection + parallax diagnostics
```

Two-view translation is recovered only **up to scale** — there is no metric
anchor in two bare views. Fix scale with a known baseline or a known 3-D point.

For outlier-contaminated correspondences, reach for
`mvg::robust::recover_relative_pose_robust` (RANSAC) instead.

### 2. Bundle adjustment (feature `refine`)

`bundle_adjust` jointly refines all camera **poses** and the 3-D **structure**
to minimize reprojection error, holding per-camera intrinsics frozen. It is
behind the `refine` feature (it pulls in `tiny-solver`):

```rust
use vision_calibration::mvg::bundle_adjust::{bundle_adjust, BundleAdjustmentOptions, BundleObservation};

let res = bundle_adjust(&intrinsics, &observations, &init_poses, &init_points,
                        &BundleAdjustmentOptions::default())?;
println!("RMS {:.3} → {:.3} px", res.initial_rms, res.final_rms);
```

By default the **first observed camera** is held fixed to remove the rigid gauge
freedom; global scale remains a gauge that reprojection cannot observe with free
structure (documented on `BundleAdjustmentOptions::fix_first_camera`). In the
example a ~18 px initial reprojection error collapses to ~0.03 px.

### 3. Stereo rectification

`rectify_stereo_pair` returns rectifying homographies that put corresponding
points on the **same image row** — the precondition for 1-D disparity search.
It handles **Scheimpflug** (tilted-sensor) cameras as well as plain pinholes:

```rust
use vision_calibration::mvg::rectification::{rectify_stereo_pair, RectifyCamera, RectifyOptions};

let rect = rectify_stereo_pair(
    &RectifyCamera::pinhole(k0),            // or RectifyCamera::scheimpflug(k0, tilt0)
    &RectifyCamera::pinhole(k1),
    &cam1_se3_cam0,                          // the calibrated relative pose
    &RectifyOptions::default(),
)?;
let p_left  = rect.rectify_left(&undistorted_pixel0);
let p_right = rect.rectify_right(&undistorted_pixel1);
// p_left.y == p_right.y for a corresponding pair
```

Inputs are **undistorted** pixels — remove lens distortion first (the rectifying
homography is the rotation part of rectification, exactly as OpenCV splits
`initUndistortRectifyMap`). See [ADR 0015](../adrs/0015-mvg-ceiling.md) for the
crate's scope boundary.

### 4. Dense matching

Once a pair is rectified, `mvg::dense::match_block` produces a per-pixel
disparity map by block matching — ZNCC over a square window (so it tolerates
per-camera gain/bias), winner-take-all with parabolic sub-pixel refinement, and
left-right-consistency / uniqueness / min-correlation filtering (rejected pixels
are `NaN`):

```rust
use vision_calibration::mvg::dense::{match_block, BlockMatchOptions, GrayImage};

// `left` / `right` are rectified grayscale images (resample the source through
// the rectifying homography first; see the dense_stereo_real example).
let disp = match_block(&left, &right, &BlockMatchOptions {
    min_disparity: 0,
    num_disparities: 64,   // search [min, min + num)
    block_size: 9,         // odd; larger = smoother, fewer matches near edges
    ..Default::default()
})?;
// disp.get(x, y) is x_left − x_right in pixels, or NaN where the matcher abstains.
```

This is the block-matching MVP (semi-global aggregation is a follow-up). Two
runnable demos write inspectable PNGs to `target/fixtures/`:

```bash
# synthetic ground truth: left | right | GT | estimate | error
cargo run -p vision-calibration-bench --example dense_synth --features tier-b --release
# real chessboard rig: undistort → rectify → match → disparity overlay
cargo run -p vision-calibration --example dense_stereo_real --release
```

## Common variations

- **Outliers in the matches** → `mvg::robust::recover_relative_pose_robust` /
  `estimate_essential` / `estimate_homography` (RANSAC).
- **More than two views** → `mvg::triangulation::triangulate_nview` for a single
  point across N cameras; `bundle_adjust` already takes any number of cameras.
- **A Scheimpflug rig** → pass `RectifyCamera::scheimpflug(k, tilt)` per camera;
  the tilt is absorbed into the rectifying homography automatically.
- **Getting the inputs from a calibration** → from a `RigExtrinsicsExport`,
  `cameras[i].k.k_matrix()` and `sensors[i]` build each `RectifyCamera`. The
  relative pose is **composed** from the two cameras' extrinsics:

  ```rust
  // cam_se3_rig[i] is T_Ci_R (camera i ← rig), so:
  let cam1_se3_cam0 = export.cam_se3_rig[right] * export.cam_se3_rig[left].inverse();
  ```

  A single `cam_se3_rig[i]` equals the relative pose only in the special case
  where the left camera is the reference frame (`cam_se3_rig[left] = Identity`);
  compose explicitly for any other pair or `reference_camera_idx`.

## What to read next

- [ADR 0015](../adrs/0015-mvg-ceiling.md) — the MVG crate's scope ceiling
  (no SfM / pose-graph / loop closure).
- [ADR 0009](../adrs/0009-coordinate-and-pose-conventions.md) — pose and frame
  conventions (`frame_se3_frame`, `T_C_W`).
- Crate docs for `vision_calibration::mvg::{pose_recovery, triangulation,
  bundle_adjust, rectification, dense, robust}`.
- The low-level solvers underneath: `vision_calibration::geometry`
  (epipolar, homography, triangulation, camera-matrix).
