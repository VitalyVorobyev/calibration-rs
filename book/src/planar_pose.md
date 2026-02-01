# Pose from Homography

Given camera intrinsics $K$ and a homography $H$ from a planar calibration board to the image, we can decompose $H$ to recover the camera pose (rotation $R$ and translation $\mathbf{t}$) relative to the board.

## Problem Statement

**Given**: Intrinsics $K$ and homography $H$ such that $\mathbf{p} \sim H \cdot [\mathbf{P}_{xy}, 1]^T$ for board points at $Z = 0$.

**Find**: Rigid transform $T_{C,B} = [R | \mathbf{t}]$ (board to camera).

**Assumptions**:
- The board lies at $Z = 0$ in world coordinates
- $K$ is known (or has been estimated)
- The homography $H$ was computed from correct correspondences

## Derivation

### Extracting Rotation and Translation

Recall from the [Homography](homography.md) chapter:

$$H \sim K [\mathbf{r}_1 \; \mathbf{r}_2 \; \mathbf{t}]$$

where $\mathbf{r}_1, \mathbf{r}_2$ are the first two columns of the rotation matrix and $\mathbf{t}$ is the translation.

Removing the intrinsics:

$$K^{-1} H = \lambda [\mathbf{r}_1 \; \mathbf{r}_2 \; \mathbf{t}]$$

Let $[\mathbf{a}_1 \; \mathbf{a}_2 \; \mathbf{a}_3] = K^{-1} H$. The scale factor $\lambda$ is recovered from the constraint that $\mathbf{r}_1$ and $\mathbf{r}_2$ have unit norm:

$$\lambda = \frac{2}{\|\mathbf{a}_1\| + \|\mathbf{a}_2\|}$$

Then:

$$\mathbf{r}_1 = \lambda \mathbf{a}_1, \quad \mathbf{r}_2 = \lambda \mathbf{a}_2, \quad \mathbf{t} = \lambda \mathbf{a}_3$$

The third rotation column is:

$$\mathbf{r}_3 = \mathbf{r}_1 \times \mathbf{r}_2$$

### Projecting onto SO(3)

Due to noise, the matrix $R_{\text{approx}} = [\mathbf{r}_1 \; \mathbf{r}_2 \; \mathbf{r}_3]$ is not exactly orthonormal. We project it onto $\text{SO}(3)$ using SVD:

$$R_{\text{approx}} = U \Sigma V^T$$

$$R = U V^T$$

If $\det(R) = -1$, flip the sign of the third column of $U$ and recompute.

### Ensuring Forward-Facing

If $t_z < 0$, the board is behind the camera. In this case, flip the sign of both $R$ and $\mathbf{t}$:

$$R \leftarrow -R, \quad \mathbf{t} \leftarrow -\mathbf{t}$$

This resolves the sign ambiguity inherent in the scale factor $\lambda$.

## Accuracy

The pose from homography is an approximate estimate because:

1. The homography itself is subject to noise (DLT algebraic error minimization, not geometric)
2. The SVD projection onto SO(3) corrects non-orthogonality but introduces additional error
3. Distortion (if not corrected) biases the homography

Typical rotation error: 1-5°. Typical translation direction error: 5-15%. These estimates are refined in non-linear optimization.

> **OpenCV equivalence**: `cv::decomposeHomographyMat` provides a similar decomposition, returning up to 4 pose candidates. calibration-rs returns a single pose by resolving ambiguities via the forward-facing constraint.

## API

```rust
let pose = estimate_planar_pose_from_h(&K, &H)?;
// pose: Iso3 (T_C_B: board to camera transform)
```

## Usage in Calibration Pipeline

In the planar intrinsics calibration pipeline, pose estimation is applied to every view after $K$ has been estimated:

1. Compute homography $H_k$ per view
2. Estimate $K$ from all homographies (Zhang's method)
3. **Decompose each $H_k$** to get pose $T_{C,B}^{(k)}$
4. Use poses as initial values for non-linear optimization
