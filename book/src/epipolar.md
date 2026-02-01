# Epipolar Geometry

Epipolar geometry describes the geometric relationship between two views of the same scene. It is encoded in the **fundamental matrix** $F$ (for pixel coordinates) or the **essential matrix** $E$ (for normalized coordinates). These matrices are central to stereo vision, visual odometry, and structure from motion.

## Fundamental Matrix

### Problem Statement

**Given**: $N$ point correspondences $\{(\mathbf{p}_i, \mathbf{p}'_i)\}$ in pixel coordinates from two views.

**Find**: Fundamental matrix $F \in \mathbb{R}^{3 \times 3}$ satisfying $\mathbf{p}'^T F \mathbf{p} = 0$ for all correspondences.

**Properties of $F$**:
- Rank 2 (by the epipolar constraint)
- 7 degrees of freedom (9 entries, scale ambiguity, rank constraint)
- Maps a point in one image to its epipolar line in the other: $\mathbf{l}' = F \mathbf{p}$

### 8-Point Algorithm

The simplest method, requiring $N \geq 8$ correspondences.

**Derivation**: The constraint $\mathbf{p}'^T F \mathbf{p} = 0$ expands to:

$$u' u f_{11} + u' v f_{12} + u' f_{13} + v' u f_{21} + v' v f_{22} + v' f_{23} + u f_{31} + v f_{32} + f_{33} = 0$$

Each correspondence gives one linear equation in the 9 entries of $F$.

**Algorithm**:

1. **Normalize** both point sets using Hartley normalization
2. **Build** $N \times 9$ design matrix $A$ where each row is:
$$[u'u, \; u'v, \; u', \; v'u, \; v'v, \; v', \; u, \; v, \; 1]$$
3. **Solve** $A\mathbf{f} = 0$ via SVD; $\mathbf{f}$ is the last column of $V$
4. **Enforce rank 2**: Decompose $F = U \Sigma V^T$ and set $\sigma_3 = 0$:
$$F = U \cdot \text{diag}(\sigma_1, \sigma_2, 0) \cdot V^T$$
5. **Denormalize**: $F = T_2^T \cdot F_{\text{norm}} \cdot T_1$

The rank-2 enforcement is critical: without it, the epipolar lines do not intersect at a common epipole.

### 7-Point Algorithm

A minimal solver using exactly 7 correspondences, producing 1 or 3 candidate matrices.

**Derivation**: With 7 equations for 9 unknowns (modulo scale), the null space of $A$ is 2-dimensional. Let the null space basis be $F_1$ and $F_2$. The fundamental matrix is:

$$F = F_1 + t \cdot F_2$$

for some scalar $t$. The rank-2 constraint $\det(F) = 0$ gives a cubic equation in $t$:

$$\det(F_1 + t F_2) = 0$$

This cubic has 1 or 3 real roots, each giving a candidate $F$.

## Essential Matrix

### Problem Statement

**Given**: $N$ point correspondences in **normalized coordinates** $\{(\hat{\mathbf{p}}_i, \hat{\mathbf{p}}'_i)\}$ where $\hat{\mathbf{p}} = K^{-1} [\mathbf{p}, 1]^T$.

**Find**: Essential matrix $E$ satisfying $\hat{\mathbf{p}}'^T E \hat{\mathbf{p}} = 0$.

**Properties of $E$**:
- $E = [t]_\times R$ where $R$ is the relative rotation and $\mathbf{t}$ is the translation direction
- Has exactly two equal non-zero singular values: $\sigma_1 = \sigma_2$, $\sigma_3 = 0$
- 5 degrees of freedom (3 rotation + 2 translation direction)

### 5-Point Algorithm (Nister)

The minimal solver, producing up to 10 candidate matrices.

**Algorithm**:

1. **Normalize** both point sets using Hartley normalization
2. **Build** $5 \times 9$ matrix $A$ (same form as 8-point)
3. **Null space**: SVD of $A$ gives a 4-dimensional null space. The essential matrix is:
$$E = x \mathbf{e}_1 + y \mathbf{e}_2 + z \mathbf{e}_3 + \mathbf{e}_4$$
for unknowns $(x, y, z)$, where $\{\mathbf{e}_i\}$ are the null space basis vectors
4. **Polynomial constraints**: The essential matrix constraint $E E^T E - \frac{1}{2} \text{tr}(E E^T) E = 0$ (9 equations) and $\det(E) = 0$ (1 equation) generate polynomial equations in $(x, y, z)$
5. **Action matrix**: These constraints are assembled into a polynomial system solved via the eigenvalues of a $10 \times 10$ action matrix
6. **Extract real solutions**: Each real eigenvalue $z$ determines $(x, y)$ and thus $E$

Up to 10 candidate essential matrices may be returned.

### Decomposing $E$ into $R, \mathbf{t}$

Given $E = U \text{diag}(\sigma, \sigma, 0) V^T$, there are 4 possible decompositions:

$$R_1 = U W V^T, \quad R_2 = U W^T V^T$$
$$\mathbf{t}_1 = +U_3, \quad \mathbf{t}_2 = -U_3$$

where $W = \begin{bmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{bmatrix}$ and $U_3$ is the third column of $U$.

**Chirality check**: Of the 4 candidates $(R_i, \mathbf{t}_j)$, only one places all triangulated points in front of both cameras. This is verified by triangulating a test point and checking that $Z > 0$ in both views.

## Coordinate Conventions

| Matrix | Input coordinates | Usage |
|--------|------------------|-------|
| Fundamental $F$ | Pixel coordinates | When intrinsics are unknown |
| Essential $E$ | Normalized coordinates ($K^{-1}\mathbf{p}$) | When intrinsics are known |

The relationship is: $E = K'^T F K$.

## RANSAC

Both solvers have RANSAC wrappers for outlier rejection:

```rust
let (F, inliers) = fundamental_8point_ransac(&pts1, &pts2, &opts)?;
let (E, inliers) = essential_5point_ransac(&pts1_norm, &pts2_norm, &opts)?;
```

The 5-point solver is preferred with RANSAC because its minimal sample size (5 vs. 8) gives better inlier efficiency.

> **OpenCV equivalence**: `cv::findFundamentalMat` (8-point and 7-point), `cv::findEssentialMat` (5-point with RANSAC).

## References

- Nister, D. (2004). "An Efficient Solution to the Five-Point Relative Pose Problem." *IEEE TPAMI*, 26(6), 756-770.
- Hartley, R.I. (1997). "In Defense of the Eight-Point Algorithm." *IEEE TPAMI*, 19(6), 580-593.
