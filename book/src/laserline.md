# Laserline Device Calibration

Laserline calibration jointly estimates camera parameters and a laser plane from observations of both a calibration board and laser line projections on the board. This is used in laser triangulation systems where a camera and laser are rigidly mounted together.

## Problem Formulation

### Parameters

- Camera intrinsics: $K = (f_x, f_y, c_x, c_y)$
- Distortion: $\mathbf{d} = (k_1, k_2, k_3, p_1, p_2)$
- Sensor tilt: $(\tau_x, \tau_y)$ (optional, for Scheimpflug cameras)
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

```rust
pub struct LaserlineDeviceConfig {
    // Camera
    pub fix_k3: bool,
    pub fix_sensor: bool,
    pub fix_poses: Vec<usize>,

    // Residual weighting
    pub calib_weight: f64,           // Weight for calibration residuals
    pub laser_weight: f64,           // Weight for laser residuals
    pub calib_loss: Option<RobustLoss>,
    pub laser_loss: Option<RobustLoss>,

    // Laser residual type
    pub laser_residual_type: LaserlineResidualType, // Default: LineDistNormalized

    // Optimization
    pub max_iters: usize,
}
```

### Weight Balancing

Since calibration residuals (in pixels) and laser residuals may have different scales, the weights allow balancing their relative influence. A common starting point is `calib_weight = 1.0` and `laser_weight = 1.0`, adjusting if one term dominates.

## Complete Example

```rust
use vision_calibration::prelude::*;
use vision_calibration::laserline_device::*;

let mut session = CalibrationSession::<LaserlineDeviceProblem>::new();
session.set_input(laserline_input)?;

run_calibration(&mut session)?;

let export = session.export()?;
println!("Plane normal: {:?}", export.plane_normal);
println!("Plane distance: {:.4}", export.plane_distance);
println!("Reprojection error: {:.4} px", export.mean_reproj_error);
println!("Laser error: {:.4}", export.mean_laser_error);
```

## Scheimpflug Support

For laser profilers with tilted sensors, the sensor model parameters $(\tau_x, \tau_y)$ are jointly optimized. The `ReprojPointPinhole4Dist5Scheimpflug2` factor handles the extended camera model.
