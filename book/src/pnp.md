# Perspective-n-Point Solvers

The Perspective-n-Point (PnP) problem estimates the camera pose from $n$ known 3D-2D point correspondences. Unlike homography-based pose estimation, PnP does not require coplanar points.

## Problem Statement

**Given**: $n$ 3D world points $\{\mathbf{P}_i\}$ and their corresponding 2D image points $\{\mathbf{p}_i\}$, plus camera intrinsics $K$.

**Find**: Camera pose $T_{C,W} = [R | \mathbf{t}] \in \text{SE}(3)$ such that $\mathbf{p}_i \sim K (R \mathbf{P}_i + \mathbf{t})$.

**Assumptions**:
- Camera intrinsics $K$ are known
- Correspondences are correct (or RANSAC is used)
- Points are not degenerate (e.g., not all collinear)

## P3P: Kneip's Minimal Solver

The P3P solver uses exactly 3 correspondences — the minimum for a finite number of solutions. It returns up to 4 candidate poses.

### Algorithm

**Input**: 3 world points $\{\mathbf{P}_0, \mathbf{P}_1, \mathbf{P}_2\}$, 3 pixel points $\{\mathbf{p}_0, \mathbf{p}_1, \mathbf{p}_2\}$, intrinsics $K$.

1. **Bearing vectors**: Convert pixels to unit bearing vectors in the camera frame:

$$\hat{\mathbf{b}}_i = \frac{K^{-1} [\mathbf{p}_i, 1]^T}{\| K^{-1} [\mathbf{p}_i, 1]^T \|}$$

2. **Inter-point distances** in world frame:

$$a = \|\mathbf{P}_1 - \mathbf{P}_2\|, \quad b = \|\mathbf{P}_0 - \mathbf{P}_2\|, \quad c = \|\mathbf{P}_0 - \mathbf{P}_1\|$$

3. **Bearing vector cosines**:

$$\cos\alpha = \hat{\mathbf{b}}_1 \cdot \hat{\mathbf{b}}_2, \quad \cos\beta = \hat{\mathbf{b}}_0 \cdot \hat{\mathbf{b}}_2, \quad \cos\gamma = \hat{\mathbf{b}}_0 \cdot \hat{\mathbf{b}}_1$$

4. **Quartic polynomial**: Using the ratios $d = (b^2 - a^2)/c^2$ and $e = b^2/c^2$, Kneip derives a quartic polynomial in a distance ratio $u$. The coefficients are functions of $d$, $e$, $\cos\alpha$, $\cos\beta$, $\cos\gamma$.

5. **Solve quartic** for up to 4 real roots $u_k$.

6. **For each root**:
   - Compute the second distance ratio $v$
   - Compute the three camera-frame distances $x, y, z$ (depths of the three points)
   - Back-project to 3D points in camera frame: $\mathbf{Q}_i = x_i \hat{\mathbf{b}}_i$
   - Recover pose from the 3D-3D correspondence $\{\mathbf{P}_i\} \leftrightarrow \{\mathbf{Q}_i\}$ using SVD-based rigid alignment

### Disambiguation

P3P returns up to 4 poses. To select the correct one, use a fourth point (or more points with RANSAC) and pick the pose with the smallest reprojection error.

## DLT PnP: Linear Solver for $n \geq 6$

The DLT (Direct Linear Transform) PnP uses an overdetermined system for $n \geq 6$ points.

### Derivation

The projection equation in normalized coordinates is:

$$\begin{bmatrix} u \\ v \end{bmatrix} = \frac{1}{[\mathbf{r}_3^T \; t_z] [\mathbf{P}, 1]^T} \begin{bmatrix} [\mathbf{r}_1^T \; t_x] [\mathbf{P}, 1]^T \\ [\mathbf{r}_2^T \; t_y] [\mathbf{P}, 1]^T \end{bmatrix}$$

where $[u, v]$ are normalized coordinates (after applying $K^{-1}$ to pixels) and $\mathbf{r}_i^T$ are rows of $R$.

Cross-multiplying gives two equations per point:

$$u (\mathbf{r}_3^T \mathbf{P} + t_z) - (\mathbf{r}_1^T \mathbf{P} + t_x) = 0$$
$$v (\mathbf{r}_3^T \mathbf{P} + t_z) - (\mathbf{r}_2^T \mathbf{P} + t_y) = 0$$

The 12 unknowns are the entries of the $3 \times 4$ matrix $[R | \mathbf{t}]$.

### The Linear System

For each point $(\mathbf{P}_i, u_i, v_i)$:

$$\begin{bmatrix} X & Y & Z & 1 & 0 & 0 & 0 & 0 & -uX & -uY & -uZ & -u \\ 0 & 0 & 0 & 0 & X & Y & Z & 1 & -vX & -vY & -vZ & -v \end{bmatrix}$$

Stacking gives $2n \times 12$ matrix $A$. Solve $A\mathbf{p} = 0$ via SVD.

### Post-Processing

1. Reshape the 12-vector into a $3 \times 4$ matrix
2. Normalize the scale using the row norms of the $3 \times 3$ block
3. Project the rotation block onto SO(3) via SVD (same as in [Pose from Homography](planar_pose.md))
4. Extract translation from the fourth column

### Hartley Normalization

The 3D world points are normalized before building the system (center at origin, scale mean distance to $\sqrt{3}$). The image points are normalized via $K^{-1}$. The result is denormalized after solving.

## RANSAC Wrappers

Both solvers have RANSAC variants for handling outliers:

```rust
// P3P + RANSAC (preferred: minimal solver)
let (pose, inliers) = pnp_ransac(
    &world_pts, &image_pts, &K,
    &RansacOptions { thresh: 5.0, ..Default::default() }
)?;

// DLT + RANSAC (less common)
let (pose, inliers) = dlt_ransac(
    &world_pts, &image_pts, &K,
    &RansacOptions { thresh: 5.0, ..Default::default() }
)?;
```

P3P with RANSAC is preferred because the minimal 3-point sample gives the highest inlier efficiency per iteration.

## Comparison

| Solver | Min. points | Solutions | Strengths |
|--------|------------|-----------|-----------|
| P3P | 3 | Up to 4 | Best for RANSAC (minimal sample) |
| DLT PnP | 6 | 1 | Simple, no polynomial solving |

> **OpenCV equivalence**: `cv::solvePnP` with `SOLVEPNP_P3P` or `SOLVEPNP_DLS`; `cv::solvePnPRansac` for robust estimation.

## References

- Kneip, L., Scaramuzza, D., & Siegwart, R. (2011). "A Novel Parametrization of the Perspective-Three-Point Problem for a Direct Computation of Absolute Camera Position and Orientation." *CVPR*.
