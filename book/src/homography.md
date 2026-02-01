# Homography Estimation (DLT)

A homography is a $3 \times 3$ projective transformation between two planes. In calibration, the most common case is the mapping from a planar calibration board to the image plane. Homography estimation is the first step in Zhang's calibration method.

## Problem Statement

**Given**: $N \geq 4$ point correspondences $\{(\mathbf{x}_i, \mathbf{x}'_i)\}$ where $\mathbf{x}_i$ are points on the world plane and $\mathbf{x}'_i$ are the corresponding image points.

**Find**: $H \in \mathbb{R}^{3 \times 3}$ such that $\mathbf{x}'_i \sim H \mathbf{x}_i$ (equality up to scale) for all $i$.

**Assumptions**:
- The world points are coplanar (the calibration board is flat)
- Correspondences are correct (or RANSAC is used to handle outliers)
- At least 4 non-collinear point correspondences

## Derivation

### From Projective Relation to Linear System

The relation $\mathbf{x}' \sim H\mathbf{x}$ in homogeneous coordinates means:

$$\begin{bmatrix} u' \\ v' \\ 1 \end{bmatrix} \sim \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

Expanding and eliminating the scale factor using cross-product (the constraint $\mathbf{x}' \times H\mathbf{x} = 0$), we get two independent equations per correspondence:

$$-x \cdot h_{11} - y \cdot h_{12} - h_{13} + u'(x \cdot h_{31} + y \cdot h_{32} + h_{33}) = 0$$

$$-x \cdot h_{21} - y \cdot h_{22} - h_{23} + v'(x \cdot h_{31} + y \cdot h_{32} + h_{33}) = 0$$

### The DLT System

Stacking all $N$ correspondences into a $2N \times 9$ matrix $A$:

$$A = \begin{bmatrix} -x_1 & -y_1 & -1 & 0 & 0 & 0 & u'_1 x_1 & u'_1 y_1 & u'_1 \\ 0 & 0 & 0 & -x_1 & -y_1 & -1 & v'_1 x_1 & v'_1 y_1 & v'_1 \\ \vdots & & & & & & & & \vdots \end{bmatrix}$$

where $\mathbf{h} = [h_{11}, h_{12}, h_{13}, h_{21}, h_{22}, h_{23}, h_{31}, h_{32}, h_{33}]^T$.

### Solving via SVD

We seek $\mathbf{h}$ minimizing $\|A\mathbf{h}\|^2$ subject to $\|\mathbf{h}\| = 1$ (to avoid the trivial solution $\mathbf{h} = 0$). This is the standard homogeneous least squares problem, solved by the SVD of $A$:

$$A = U \Sigma V^T$$

The solution is the last column of $V$ (corresponding to the smallest singular value). Reshaping this 9-vector into a $3 \times 3$ matrix gives $H$.

### Normalization and Denormalization

The complete algorithm with Hartley normalization:

1. Normalize world and image points: $(\tilde{\mathbf{x}}_i, T_w)$ and $(\tilde{\mathbf{x}}'_i, T_i)$
2. Build the $2N \times 9$ matrix $A$ from normalized points
3. Solve via SVD to get $H_{\text{norm}}$
4. Denormalize: $H = T_i^{-1} \cdot H_{\text{norm}} \cdot T_w$
5. Scale: $H \leftarrow H / H_{33}$

## Degrees of Freedom

$H$ is a $3 \times 3$ matrix with 9 entries, but it is defined up to scale, giving **8 degrees of freedom**. Each correspondence provides 2 equations, so the minimum is $N = 4$ correspondences (giving $2 \times 4 = 8$ equations for 8 unknowns).

With $N > 4$, the system is overdetermined and the SVD solution minimizes the algebraic error in a least-squares sense.

## RANSAC Wrapper

For real data with potential mismatched correspondences, `dlt_homography_ransac` wraps the DLT solver in RANSAC:

```rust
let (H, inliers) = dlt_homography_ransac(&world_pts, &image_pts, &opts)?;
```

- `MIN_SAMPLES = 4`
- **Residual**: Euclidean reprojection error in pixels: $\| \text{dehomogenize}(H \mathbf{x}) - \mathbf{x}' \|$
- **Degeneracy check**: Tests if the first 3 world points are collinear (a degenerate configuration for homography estimation)

> **OpenCV equivalence**: `cv::findHomography` with `RANSAC` method.

## API

```rust
// Direct DLT (all points assumed inliers)
let H = dlt_homography(&world_2d, &image_2d)?;

// With RANSAC
let opts = RansacOptions {
    max_iters: 1000,
    thresh: 3.0,   // pixels
    confidence: 0.99,
    ..Default::default()
};
let (H, inliers) = dlt_homography_ransac(&world_2d, &image_2d, &opts)?;
```

## Geometric Interpretation

For a planar calibration board at $Z = 0$ in world coordinates, the $3 \times 4$ camera projection matrix $P = K[R | \mathbf{t}]$ reduces to a $3 \times 3$ homography:

$$H \sim K [\mathbf{r}_1 \; \mathbf{r}_2 \; \mathbf{t}]$$

where $\mathbf{r}_1, \mathbf{r}_2$ are the first two columns of $R$. This relationship is the foundation of Zhang's calibration method (next chapter).
