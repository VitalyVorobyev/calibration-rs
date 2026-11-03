# Linear Triangulation

Triangulation reconstructs a 3D point from its projections in two or more calibrated views. It is the inverse of the projection operation: given pixel observations and camera poses, recover the world point.

## Problem Statement

**Given**: $n \geq 2$ views with poses $\{(R_i, \mathbf{t}_i)\}$ (world to camera $i$) and, in each, the *bearing* $\mathbf{b}_i$ along which the point is seen: the unit ray from the camera's own unprojection of the observed pixel.

**Find**: 3D point $\mathbf{X} = [X, Y, Z]^T$ in the world frame such that $\mathbf{b}_i \parallel R_i \mathbf{X} + \mathbf{t}_i$.

**Assumptions**:
- Camera poses and models are known
- Correspondences are correct
- The point is visible from all views (not occluded)

Working on bearings rather than on image points serves every camera model: a pinhole's bearing is $(x, y, 1)$ of its normalized coordinates, and a fisheye's may point more than 90° from its axis, where normalized coordinates do not exist.

## Derivation (DLT)

With $P_i = [R_i \mid \mathbf{t}_i]$ and the point $\mathbf{X}_h = [X, Y, Z, 1]^T$, the constraint says $P_i \mathbf{X}_h$ is parallel to $\mathbf{b}_i$, i.e. their cross product vanishes:

$$[\mathbf{b}_i]_\times\, P_i\, \mathbf{X}_h = 0$$

This gives three equations per view, of rank two. Stacking all views gives a $3n \times 4$ system:

$$A \mathbf{X}_h = 0$$

Solve via the smallest eigenvector of $A^T A$ (the smallest-singular direction): $\mathbf{X}_h$ is the right singular vector corresponding to the smallest singular value. Dehomogenize: $\mathbf{X} = [X_h / w, Y_h / w, Z_h / w]^T$. A near-zero second-smallest singular value (identical cameras, coincident rays) is reported as a degenerate system.

## Limitations

- **Algebraic error**: The DLT minimizes algebraic error, not geometric (reprojection) error, and carries no covariance
- **Baseline sensitivity**: For small baselines (nearly parallel views), the triangulation is poorly conditioned — small pixel errors lead to large depth errors
- **Outlier sensitivity**: A single incorrect correspondence corrupts the result; no robustness mechanism

For accurate points, refine them on the reprojection error with their noise stated: the calibrated entry point, `mvg::triangulation::triangulate`, does so, and returns the points' covariance with the calibration's share (see [Pose Recovery and Triangulation](mvg_pose.md)). The linear estimate is its starting point.

## API

```rust
let X = triangulate_point_linear(&cam_se3_world, &bearings)?;
```

Where `cam_se3_world` holds each view's `Iso3` pose and `bearings` the matching rays (any nonzero length). The calibrated entry point, `mvg::triangulation::triangulate`, builds the bearings from pixels through a `CalibratedRig`, refines the points and adds diagnostics and covariance; see [Pose Recovery and Triangulation](mvg_pose.md).

In Python, `vision_calibration.mvg.triangulate` is the calibrated entry point; the raw linear solver is Rust only (numpy and OpenCV cover linear algebra on plain arrays).

Deterministic two-view solvers, all free functions in per-topic modules:

| Module | Functions |
|--------|-----------|
| `homography` | `dlt_homography`, `dlt_homography_ransac` |
| `epipolar` | `fundamental_8point`, `essential_5point` |
| `triangulation` | `triangulate_point_linear` |
| `camera_matrix` | `dlt_camera_matrix` |

The facade re-exports the main solvers at the root of its `mvg` module (`vision_calibration::mvg::dlt_homography`, …); the full set is in the `vision-geometry` crate.
