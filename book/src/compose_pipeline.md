# Compose Your Own Pipeline

A calibration workflow runs four stages:
1. it takes the observations;
2. it computes starting values in closed form;
3. it refines them with a non-linear solve;
4. it reports the covariance of the result.

The piece behind each stage is public. When your setup is not one of the
workflows, build it from the same pieces. This chapter builds a measurement
this way: the pose of a machined part, seen by a camera calibrated from
board images. The σ of the measurement includes the uncertainty of the
calibration.

This is Rust only, as is [Your Own Residual](custom_residual.md), which
this chapter builds on.

## The pieces

| Stage | Public piece |
|---|---|
| Observations | `core::CorrespondenceView` (points and their pixels), `core::PlanarDataset` |
| Starting values | [`init`](https://docs.rs/vision-calibration/latest/vision_calibration/init/): a camera from views of a plane, a plane's pose from its homography (`mvg::dlt_homography`), a pose from 3D–2D points (`pnp`), hand-eye, a plane through points |
| Solve | [`optim::problem`](https://docs.rs/vision-calibration/latest/vision_calibration/optim/problem/): blocks, a camera, reprojection residuals, residuals of your own, `solve` |
| Uncertainty | `Solution::uncertainty`, labelled as a workflow's; [`projection_uncertainty`](uncertainty_projection.md) for a camera |
| Testing | `synthetic` (feature `synthetic`): boards, poses and noise with a known truth |

## Starting values

The camera comes from Zhang's method with a distortion fit, over the board
views. Each board pose comes from its homography, and the part's pose from
EPnP on its eight fiducials, whose positions on the part are known:

```rust
use vision_calibration::core::{CorrespondenceView, PlanarDataset, Pt2, View};
use vision_calibration::init::{
    IterativeIntrinsicsOptions, estimate_intrinsics_iterative, estimate_planar_pose_from_h, pnp,
};
use vision_calibration::mvg::dlt_homography;

let mut views = Vec::new();
for pixels in &board_pixels {
    views.push(View::without_meta(CorrespondenceView::new(board.clone(), pixels.clone())?));
}
let start = estimate_intrinsics_iterative(
    &PlanarDataset::new(views)?,
    IterativeIntrinsicsOptions::default(),
)?;

let board_2d: Vec<Pt2> = board.iter().map(|p| Pt2::new(p.x, p.y)).collect();
let mut board_poses = Vec::new();
for pixels in &board_pixels {
    let h = dlt_homography(&board_2d, pixels)?;
    board_poses.push(estimate_planar_pose_from_h(&start.k.k_matrix(), &h)?);
}
let part_start = pnp::epnp(&fiducials, &part_pixels, &start.k)?;
```

EPnP uses the pinhole intrinsics and ignores distortion. That is enough
for a starting value, because the solve refines it with the full camera
model.

## One problem for the calibration and the measurement

The board views and the part enter one problem, through one camera. Each
has its own pose block:

```rust
use vision_calibration::core::{CameraFix, pinhole_camera_params};
use vision_calibration::optim::RobustLoss;
use vision_calibration::optim::problem::{
    PixelNoise, PoseChain, Problem, SolveOptions, TargetPoint, solve,
};

let mut problem = Problem::new();
let cam = problem.add_camera(&pinhole_camera_params(&start), &CameraFix::default())?;
for (v, (pose, pixels)) in board_poses.iter().zip(&board_pixels).enumerate() {
    let pose = problem.add_pose(format!("view{v}"), pose)?;
    for (point, pixel) in board.iter().zip(pixels) {
        problem.add_reprojection(
            cam,
            PoseChain::Pose(pose),
            TargetPoint::Known(*point),
            *pixel,
            PixelNoise::Estimated,
            RobustLoss::None,
        )?;
    }
}
let part = problem.add_pose("part", &part_start)?;
for (point, pixel) in fiducials.iter().zip(&part_pixels) {
    problem.add_reprojection(
        cam,
        PoseChain::Pose(part),
        TargetPoint::Known(*point),
        *pixel,
        PixelNoise::Estimated,
        RobustLoss::None,
    )?;
}

let solution = solve(&problem, &SolveOptions::default())?;
let camera_se3_part = solution.pose(part).unwrap();
```

Each reprojection names how its point reaches the camera (`PoseChain`:
one pose, or a camera of a rig), the point (`TargetPoint`: a known point
such as a board corner, or a block being estimated) and the pixel's noise
(`PixelNoise`: estimated from the scatter of the problem's reprojections,
or a stated covariance).

The board pixels determine the camera. The part's pixels determine its
pose through that camera. Its covariance block then holds both sources:
the noise of the part's own pixels, and the uncertainty of the camera,
which comes from the board pixels.

## A calibration you have as a result

A calibration's result carries its cameras and their covariance, without
the board pixels it came from. `add_calibrated_rig` adds it to a problem
with that covariance:

```rust
use vision_calibration::core::CalibratedRig;
use vision_calibration::optim::problem::Use;

let rig: CalibratedRig = export.calibrated_rig()?;
let mut problem = Problem::new();
let blocks = problem.add_calibrated_rig(&rig, Use::Consider)?;
let part = problem.add_pose("part", &part_start)?;
for (point, pixel) in fiducials.iter().zip(&part_pixels) {
    problem.add_reprojection(
        blocks.camera(0).unwrap(),
        PoseChain::Pose(part),
        TargetPoint::Known(*point),
        *pixel,
        PixelNoise::Estimated,
        RobustLoss::None,
    )?;
}
let solution = solve(&problem, &SolveOptions::default())?;
```

Every calibration result has `calibrated_rig()`. The calibration enters in
one of two ways:
- **`Use::Consider`**: the cameras stay fixed at their calibrated values,
  so the estimate is the one a fixed camera gives. The covariance adds the
  calibration's share, `S Σ_c Sᵀ`, with `Σ_c` the calibration's covariance
  and `S` how the estimate moves with the calibration. This is the use for
  a measurement.
- **`Use::Prior`**: the cameras are free, held by a prior with the
  calibration's covariance, so the new observations refine them too. The
  prior is weighed by its covariance, so every pixel's noise is stated
  (`PixelNoise::Stated`).

For a rig of several cameras, `blocks.chain(camera, rig_se3_target)` is the
chain from a target posed in the rig frame to one camera.

## The measurement's σ

The part's covariance is labelled `part.rx … part.tz`. As for any pose,
`part.tx … part.tz` are steps in the part's own frame
(see [Reading the result](custom_residual.md#reading-the-result)). The
variance of its depth in the camera frame is `r₃ᵀ Σ_t r₃`, with `r₃` the
third row of the pose's rotation.

To check that σ against the actual scatter, the test setup is:
- the calibration: 8 views of a 9×6 board with 30 mm squares, at about
  0.45 m;
- the part: the corners of an 80 × 60 × 30 mm block, at 0.5 m;
- noise: 0.3 px in every image.

Over 120 repetitions, each with new noise in the calibration images and in
the part's image, the depth's σ was:

| | stated σ(depth) | actual scatter / stated σ |
|---|---|---|
| one problem, as above | 2.53 mm | 0.91 |
| calibrate first, then measure with `Use::Consider` | 2.54 mm | 0.91 |
| calibrate first, then measure with the camera fixed | 0.91 mm | 2.52 |

The depth estimates are nearly the same every way and scatter by 2.3 mm.
Only the stated σ differs.

A camera held fixed is treated as exact, so the σ counts only the part's
16 pixel coordinates. At half a metre, the calibration's uncertainty in the
focal length dominates: the depth scales with it. Fixing the camera states
σ about 2.5 times too small.

## Measuring well

- **Several measurements with one calibration.** Put them all in one
  problem. Their covariance then includes the correlation that the shared
  calibration induces between them; two separate solves lose it.
- **A calibration as a result.** Add it with `Use::Consider`, not as a
  fixed camera: fixed, it gives the understated σ above.
- **Outliers.** A measurement image with false detections takes a robust
  loss on its reprojections (see [Robust Loss Functions](robust_loss.md)).

## Testing a pipeline of your own

The `synthetic` feature generates boards, poses and noisy pixels from a
known truth. Two checks are worth running:
- run your pipeline on that data and compare its result with the truth;
- repeat the run over many noise draws and compare the stated σ with the
  scatter of the estimates.

The table above came from such a run.
