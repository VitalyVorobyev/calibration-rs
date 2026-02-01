# Linear Triangulation

Triangulation reconstructs a 3D point from its projections in two or more calibrated views. It is the inverse of the projection operation: given pixel observations and camera poses, recover the world point.

## Problem Statement

**Given**: $n \geq 2$ camera projection matrices $\{P_i\}$ (each $3 \times 4$) and corresponding pixel observations $\{\mathbf{p}_i = (u_i, v_i)\}$.

**Find**: 3D point $\mathbf{X} = [X, Y, Z]^T$ such that $\mathbf{p}_i \sim P_i [\mathbf{X}, 1]^T$.

**Assumptions**:
- Camera poses and intrinsics are known
- Correspondences are correct
- The point is visible from all views (not occluded)

## Derivation (DLT)

For each view $i$, the projection constraint $\mathbf{p}_i \sim P_i \mathbf{X}_h$ gives (by cross-multiplication):

$$u_i (\mathbf{p}_3^{(i)T} \mathbf{X}_h) - \mathbf{p}_1^{(i)T} \mathbf{X}_h = 0$$
$$v_i (\mathbf{p}_3^{(i)T} \mathbf{X}_h) - \mathbf{p}_2^{(i)T} \mathbf{X}_h = 0$$

where $\mathbf{p}_k^{(i)T}$ is the $k$-th row of $P_i$ and $\mathbf{X}_h = [X, Y, Z, 1]^T$.

Stacking all views gives a $2n \times 4$ system:

$$A \mathbf{X}_h = 0$$

Solve via SVD: $\mathbf{X}_h$ is the right singular vector corresponding to the smallest singular value. Dehomogenize: $\mathbf{X} = [X_h / w, Y_h / w, Z_h / w]^T$.

## Limitations

- **No uncertainty modeling**: The DLT minimizes algebraic error, not geometric (reprojection) error
- **Baseline sensitivity**: For small baselines (nearly parallel views), the triangulation is poorly conditioned — small pixel errors lead to large depth errors
- **Outlier sensitivity**: A single incorrect correspondence corrupts the result; no robustness mechanism

For high-accuracy 3D reconstruction, triangulation should be followed by bundle adjustment that jointly optimizes points and cameras to minimize reprojection error.

## API

```rust
let X = triangulate_point_linear(&projection_matrices, &pixel_points)?;
```

Where each projection matrix $P_i = K_i [R_i | \mathbf{t}_i]$ is precomputed from the camera intrinsics and pose.
