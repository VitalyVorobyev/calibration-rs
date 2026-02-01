# Laser Plane Initialization

In laser triangulation, a laser line is projected onto the scene and a camera observes the line. The laser plane — defined by a normal vector and distance from the camera origin — must be calibrated. This chapter describes the linear initialization of the laser plane from observations of the laser line on a calibration target.

## Problem Statement

**Given**:
- Camera intrinsics $K$ and distortion
- Per-view camera poses $T_{C,T}^{(v)}$ (target to camera)
- Per-view laser pixel observations $\{\mathbf{p}_{vi}\}$ (pixels where the laser line intersects the target)

**Find**: Laser plane $(\hat{\mathbf{n}}, d)$ in the camera frame, where $\hat{\mathbf{n}}$ is the unit normal and $d$ is the signed distance from the camera origin:

$$\hat{\mathbf{n}}^T \mathbf{P} = d$$

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
6. **Distance**: $d = \hat{\mathbf{n}}^T \bar{\mathbf{P}}$

### Degeneracy Detection

If all 3D points are nearly collinear (e.g., because all views have the same target pose), the plane is underdetermined. The algorithm checks the ratio of the two smallest singular values:

$$\frac{\sigma_2}{\sigma_1} < \epsilon$$

If the points are too close to collinear, the plane fit fails with an error.

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

```rust
let plane = LaserlinePlaneSolver::from_views(
    &camera, &poses, &laser_pixels
)?;
// plane.normal: Vec3 (unit normal in camera frame)
// plane.distance: f64 (signed distance from camera origin)
```
