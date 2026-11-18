# Laser Plane Initialization

In laser triangulation, a laser line is projected onto the scene and a camera observes the line. The laser plane — defined by a normal vector and distance from the camera origin — must be calibrated. This chapter describes the linear initialization of the laser plane from observations of the laser line on a calibration target.

## Problem Statement

**Given**:
- Camera intrinsics $K$ and distortion
- Per-view camera poses $T_{C,T}^{(v)}$ (target to camera)
- Per-view laser pixel observations $\{\mathbf{p}_{vi}\}$ (pixels where the laser line intersects the target)

**Find**: Laser plane $(\hat{\mathbf{n}}, d)$ in the camera frame, where $\hat{\mathbf{n}}$ is the unit normal and $d$ is the signed distance from the camera origin:

$$\hat{\mathbf{n}}^T \mathbf{P} + d = 0$$

for all 3D points $\mathbf{P}$ on the laser plane (in camera coordinates).

## Algorithm

### Step 1: Back-Project Laser Pixels to 3D

For each laser pixel in each view:

1. **Undistort** the pixel to get normalized coordinates: $\mathbf{n} = D^{-1}(K^{-1} [\mathbf{p}, 1]^T)$
2. **Form a ray** in the camera frame: $\mathbf{r}(t) = t \cdot [\mathbf{n}, 1]^T$
3. **Intersect with the target plane**: The target at $Z_{\text{target}} = 0$ in target coordinates, transformed to camera frame by $T_{C,T}$, defines a plane. Solve for $t$ that places the ray on this plane.
4. **Compute 3D point**: $\mathbf{P} = \mathbf{r}(t)$ in camera coordinates

This gives a set of 3D points on the laser plane.

### Step 2: Fit a Plane via SVD

Given $N$ 3D points $\{\mathbf{P}_i\}$ on the laser plane:

1. **Centroid**: $\bar{\mathbf{P}} = \frac{1}{N} \sum \mathbf{P}_i$
2. **Center the points**: $\mathbf{Q}_i = \mathbf{P}_i - \bar{\mathbf{P}}$
3. **Covariance matrix**: $C = \sum \mathbf{Q}_i \mathbf{Q}_i^T$
4. **SVD**: $C = U \Sigma V^T$
5. **Normal**: $\hat{\mathbf{n}}$ is the eigenvector corresponding to the smallest singular value (last column of $V$)
6. **Distance**: $d = -\hat{\mathbf{n}}^T \bar{\mathbf{P}}$

### Degeneracy Detection

If all 3D points are nearly collinear (e.g., because all views have the same target pose), the plane is underdetermined. The algorithm checks the ratio of the two smallest singular values:

$$\frac{\sigma_2}{\sigma_1} < \epsilon$$

If the points are too close to collinear, the plane fit fails with an error.

### Robust Fit

The SVD fit weighs every point: a stray reflection or a pixel picked off the
stripe tilts the plane. `from_views_ransac` and `from_points_3d_ransac` run the
same fit inside a seeded [RANSAC](ransac.md) instead: each iteration fits a
plane to 3 sampled points, the inliers are the points within `thresh` of it
(the point-to-plane distance, in the unit of the points), and the plane with
the largest consensus wins. With `refit_on_inliers` each candidate is refit on
its inliers by the SVD fit above and rescored; without it the plane is the one
through the best 3-point sample. A plane is accepted only when its inliers
reach both `min_inliers` and `min_inlier_fraction`, the share of **all** the
points (one per laser pixel for a fit from views); otherwise the fit returns
an error rather than a plane fitted to the outliers.

## Plane Representation for Optimization

In the non-linear optimization, the plane is parameterized as:

- **Normal**: A unit vector $\hat{\mathbf{n}} \in S^2$ (2 degrees of freedom on the unit sphere)
- **Distance**: A scalar $d \in \mathbb{R}$ (1 degree of freedom)

The $S^2$ manifold is used for the normal to maintain the unit-norm constraint during optimization (see [Manifold Optimization](manifolds.md)).

## Accuracy

The linear plane initialization accuracy depends on:

- **View diversity**: Different target orientations provide points at different locations on the laser plane, improving the fit
- **Camera calibration accuracy**: Errors in intrinsics, distortion, or poses propagate to the 3D point estimates
- **Number of laser pixels**: More pixels (from more views or more detected pixels per view) improve the SVD fit

Typical accuracy: 1-5° normal direction error, 5-20% distance error. Refined in the [Laserline Device Calibration](laserline.md) optimization.

## API

The closed-form fit from views:

```rust,no_run
use vision_calibration::core::{Iso3, PinholeCamera, Pt2};
use vision_calibration::init::{LaserlinePlaneSolver, PlaneFitView};
# fn main() -> anyhow::Result<()> {
# let poses: Vec<Iso3> = unimplemented!("the board's pose per view");
# let laser_pixels_per_view: Vec<Vec<Pt2>> = unimplemented!("the laser pixels per view");
# let camera: PinholeCamera = unimplemented!("the calibrated camera");

// Each view carries laser pixels and the camera-to-target pose
let views: Vec<PlaneFitView> = poses.iter().zip(laser_pixels_per_view.iter())
    .map(|(pose, pixels)| PlaneFitView {
        cam_se3_target: *pose,
        laser_pixels: pixels.clone(),
    })
    .collect();

let plane = LaserlinePlaneSolver::from_views(&views, &camera)?;
// plane.normal: UnitVector3<f64> (unit normal in camera frame)
// plane.distance: f64 (signed distance from camera origin)
// plane.rmse: f64 (fit residual)
# Ok(())
# }
```

The robust fit takes `RansacOptions`: the threshold (meters here), the
iteration cap, the confidence, the minimum inlier count and fraction, the
refit and the seed. The same points and options give the same plane, and the
estimate lists the inliers by index. Here 70 points lie on the plane
$z = 0.5 + 0.3x$ and 30 lie 10 mm off it:

```rust
use vision_calibration::core::Pt3;
use vision_calibration::init::LaserlinePlaneSolver;
use vision_calibration::mvg::RansacOptions;
# fn main() -> anyhow::Result<()> {
let mut points: Vec<Pt3> = (0..70)
    .map(|i| {
        let (x, y) = ((i % 10) as f64 * 0.02, (i / 10) as f64 * 0.02);
        Pt3::new(x, y, 0.5 + 0.3 * x)
    })
    .collect();
points.extend((0..30).map(|i| {
    let (x, y) = ((i % 10) as f64 * 0.02, (i / 10) as f64 * 0.03);
    Pt3::new(x, y, 0.51 + 0.3 * x)
}));

let opts = RansacOptions::new(1e-3) // inlier within 1 mm of the plane
    .with_max_iters(500)
    .with_confidence(0.999)
    .with_min_inliers(20)
    .with_min_inlier_fraction(0.6) // of all 100 points
    .with_refit_on_inliers(true)
    .with_seed(7);
let plane = LaserlinePlaneSolver::from_points_3d_ransac(&points, &opts)?;
assert_eq!(plane.inliers, (0..70).collect::<Vec<_>>());
assert!(plane.rmse < 1e-9);

// A policy the data cannot meet is an error, not a plane through outliers.
let strict = opts.clone().with_min_inlier_fraction(0.8);
assert!(LaserlinePlaneSolver::from_points_3d_ransac(&points, &strict).is_err());
# Ok(())
# }
```

`LaserlinePlaneSolver::from_views_ransac(&views, &camera, &opts)` is the same
fit on the points of the views; its `inliers` index the laser pixels taken
in order, view by view.

### Seeding a Workflow

The laser workflows initialize the plane with the closed-form fit. To start
one from the robust fit instead, fit each camera's plane and pass it as the
manual seed; for a rig:

```rust,no_run
use vision_calibration::core::LaserPlane;
use vision_calibration::init::{LaserlinePlaneSolver, PlaneFitView};
use vision_calibration::mvg::RansacOptions;
use vision_calibration::rig_laserline_device::{
    PlaneManualInit, RigLaserlineDeviceProblem, step_init_with_seed, step_optimize,
};
use vision_calibration::session::CalibrationSession;
# fn main() -> anyhow::Result<()> {
# let mut session: CalibrationSession<RigLaserlineDeviceProblem> =
#     unimplemented!("a session with its input set");
let opts = RansacOptions::new(0.5e-3).with_min_inlier_fraction(0.8).with_seed(1);
let input = session.require_input()?.clone();
let mut planes = Vec::new();
for cam in 0..input.dataset.num_cameras {
    let camera = input.upstream.cameras[cam].build()?;
    let views: Vec<PlaneFitView> = input
        .dataset
        .views
        .iter()
        .zip(&input.upstream.rig_se3_target)
        .filter_map(|(view, rig_se3_target)| {
            let pixels = view.laser_pixels[cam].clone()?;
            Some(PlaneFitView {
                laser_pixels: pixels,
                cam_se3_target: input.upstream.cam_se3_rig[cam] * rig_se3_target,
            })
        })
        .collect();
    let fit = LaserlinePlaneSolver::from_views_ransac(&views, &camera, &opts)?;
    planes.push(LaserPlane::new(fit.normal.into_inner(), fit.distance)?);
}
let mut seed = PlaneManualInit::default();
seed.laser_planes_cam = Some(planes);
step_init_with_seed(&mut session, seed, None)?;
step_optimize(&mut session, None)?;
# Ok(())
# }
```
