# Laserline Device Calibration

Laserline calibration jointly estimates camera parameters and a laser plane from observations of both a calibration board and laser line projections on the board. This is used in laser triangulation systems where a camera and laser are rigidly mounted together.

## Problem Formulation

### Parameters

- Camera intrinsics: $K = (f_x, f_y, c_x, c_y)$
- Distortion: the coefficients of the configured model, e.g.
  $\mathbf{d} = (k_1, k_2, k_3, p_1, p_2)$ for Brown-Conrady
- Sensor tilt: $(\tau_x, \tau_y)$ (Scheimpflug cameras only)
- Per-view poses: $\{T_v\}$ (camera-to-target SE(3))
- Laser plane: normal $\hat{\mathbf{n}} \in S^2$, distance $d \in \mathbb{R}$

### Observations

Each view provides two types of observations:

1. **Calibration points**: 2D-3D correspondences from the chessboard (same as planar intrinsics)
2. **Laser pixels**: 2D pixel positions where the laser line appears on the target

### Objective

The cost function has two terms:

$$F = w_c \sum_{\text{calib}} \rho_c\left(\| \pi(K, \mathbf{d}, S, T_v, \mathbf{P}_j) - \mathbf{p}_{vj} \|\right) + w_l \sum_{\text{laser}} \rho_l\left( r_{\text{laser}}(\cdot) \right)$$

where $w_c, w_l$ are weights, $\rho_c, \rho_l$ are (possibly different) robust loss functions, and $r_{\text{laser}}$ is the laser residual.

## Laser Residual Types

Two approaches are implemented for the laser residual, selectable via configuration.

### PointToPlane (3D Distance)

**Algorithm**:
1. Undistort laser pixel to normalized coordinates
2. Back-project to a ray in camera frame
3. Intersect the ray with the target plane (at $Z_{\text{target}} = 0$, transformed by pose $T_v$) to get a 3D point $\mathbf{P}_c$
4. Compute signed distance from $\mathbf{P}_c$ to the laser plane:

$$r = \sqrt{w} \cdot (\hat{\mathbf{n}}^T \mathbf{P}_c - d)$$

**Residual dimension**: 1 (meters)

### LineDistNormalized (2D Line Distance) — Default

**Algorithm**:
1. Compute the 3D intersection line of the laser plane and the target plane (in camera frame)
2. Project this 3D line onto the $Z = 1$ normalized camera plane
3. Undistort the laser pixel to normalized coordinates (done once, not per-iteration)
4. Measure the perpendicular distance from the undistorted pixel to the projected line
5. Scale by $\sqrt{f_x \cdot f_y}$ for pixel-comparable units:

$$r = \sqrt{w} \cdot d_\perp \cdot \sqrt{f_x f_y}$$

**Residual dimension**: 1 (effective pixels)

### Comparison

| Property | PointToPlane | LineDistNormalized |
|----------|--------------|--------------------|
| Residual units | meters | pixels |
| Undistortion | Per-iteration | Once per pixel |
| Geometry | Ray-plane intersection | 2D line distance |
| Speed | Slower | Faster |
| Recommended | Alternative | **Default** |

Both approaches yield similar accuracy in practice (<6% intrinsics error, <5° plane normal error in synthetic tests).

## Derivation: Line-Distance Residual

### Plane-Plane Intersection Line

The laser plane $\hat{\mathbf{n}}^T \mathbf{P} = d$ and the target plane (normal $\hat{\mathbf{m}}$, distance $e$ from camera, derived from pose $T_v$) intersect in a 3D line with:

$$\text{direction: } \hat{\mathbf{l}} = \hat{\mathbf{n}} \times \hat{\mathbf{m}}$$

$$\text{origin: solved from } \begin{bmatrix} \hat{\mathbf{n}}^T \\ \hat{\mathbf{m}}^T \end{bmatrix} \mathbf{P}_0 = \begin{bmatrix} d \\ e \end{bmatrix}$$

### Projection onto Normalized Plane

Project the 3D line to the $Z = 1$ plane:

$$\mathbf{p}_0 = \text{point on line at } Z = 1$$
$$\hat{\mathbf{d}}_{\text{2D}} = \text{direction projected to } Z = 1 \text{ plane}$$

### Perpendicular Distance

For an undistorted pixel $\mathbf{q}$ in normalized coordinates, the perpendicular distance to the 2D line is:

$$d_\perp = \frac{|(\mathbf{q} - \mathbf{p}_0) \times \hat{\mathbf{d}}_{\text{2D}}|}{|\hat{\mathbf{d}}_{\text{2D}}|}$$

## Configuration

`init` and `solver` are the shared sub-structs, `camera` the camera model
and its fixed parameters (as in [Planar Intrinsics](planar_intrinsics.md)),
and `optimize` groups the laser-specific bundle-adjustment settings:

```rust
pub struct LaserlineDeviceConfig {
    // Per-camera linear-initialization stage settings.
    pub init: IntrinsicsInitConfig,
    // { init_iterations: usize = 2, fix_k3: bool = true,
    //   fix_tangential: Option<bool> = None }  (None: true for a Scheimpflug
    //   sensor, false otherwise)

    // The camera model (any distortion kind, an identity or Scheimpflug
    // sensor) and the parameters the bundle adjustment holds fixed, named per
    // slot; a slot left out keeps the model's default (`k3` for Brown-Conrady;
    // everything but the low-order radial terms under a Scheimpflug sensor).
    pub camera: CameraConfig, // { model: CameraModelSpec, fix: CameraFix }

    // Non-linear solve stage settings. `robust_loss` is not consulted here —
    // laser-carrying stages use `optimize.calib_loss`/`optimize.laser_loss`.
    pub solver: SolverConfig, // { max_iters: usize = 50, verbosity: usize = 0, .. }

    // Bundle-adjustment options.
    pub optimize: LaserlineDeviceOptimizeConfig,
}

pub struct LaserlineDeviceOptimizeConfig {
    pub calib_loss: RobustLoss,        // Default: Huber { scale: 1.0 }
    pub laser_loss: RobustLoss,        // Default: Huber { scale: 1.0 }, in the residual's unit
    pub calib_weight: f64,             // Weight for calibration residuals (default: 1.0)
    pub laser_weight: f64,             // Weight for laser residuals (default: 1.0)
    pub fix_poses: Vec<usize>,         // Default: empty (each pose is set by its own corners)
    pub fix_plane: bool,
    pub laser_residual_type: LaserlineResidualType, // Default: LineDistNormalized
}
```

The line-distance residual needs a projection that keeps lines straight
(the pinhole); the configuration is checked before the solve.

### Weight Balancing

Since calibration residuals (in pixels) and laser residuals may have different scales, the weights allow balancing their relative influence. A common starting point is `calib_weight = 1.0` and `laser_weight = 1.0`, adjusting if one term dominates.

## Complete Example

```rust
use vision_calibration::prelude::*;
use vision_calibration::laserline_device::*;

let mut session = CalibrationSession::<LaserlineDeviceProblem>::new();
session.set_input(laserline_input)?;

run_calibration(&mut session, None)?;

let export = session.export()?;
println!("Plane normal: {:?}", export.estimate.params.plane.normal);
println!("Plane distance: {:.4}", export.estimate.params.plane.distance);
println!("Reprojection error: {:.4} px", export.mean_reproj_error);
println!("Laser error: {:.4}", export.stats.mean_laser_error);

// The plane's uncertainty, from corner and laser residuals together.
if let Some(cov) = export.estimate.uncertainty.covariance() {
    println!("Plane distance σ: {:.2e} m", cov.std_dev("plane.d").unwrap());
}
```

`export.estimate.params.camera` is the calibrated camera in its model
(`CameraParams`). The estimate's `uncertainty` covers the camera's free
parameters (intrinsics, distortion and, for a tilted sensor, the tilt) and the
laser plane (`plane.nx`, `plane.ny`, `plane.nz`, `plane.d`, in the camera
frame), with each view's target pose as a per-view block; see
[Parameter Uncertainty](uncertainty.md). The rig laserline result labels
each camera's plane `cam<i>.plane.nx` …, and the joint rig hand-eye
laserline result adds the planes to the rig hand-eye labels.

## Scheimpflug Support

Laser profilers usually tilt their sensor. Select it in the camera model:

```rust
let mut config = LaserlineDeviceConfig::default();
config.camera = CameraConfig::new(CameraModelSpec::pinhole(
    DistortionKind::BrownConrady5,
    SensorKind::Scheimpflug,
));
```

A laser profiler's mount tilt is a hardware property, measured separately,
so the device holds it by default: give it in the manual-init `sensor` seed
(`step_init_with_seed`; zero when unseeded), and the camera initializes as an
untilted one with the tilt attached. To estimate the tilt instead, list the
sensor parameters to hold explicitly: `camera.fix.sensor = Some(vec![])`
refines both angles. Without a seed, `step_init` then estimates the tilt with
the linear tilt sweep and refines the camera alone through the staged tilt
multi-start before it fits the laser plane, because a cold `tilt = 0` start
falls into the tilt / principal-point / distortion degeneracy. A tilt seed for
an untilted camera must be zero.
