# Manifold Optimization

Camera poses are elements of the Lie group $\text{SE}(3)$, which is a 6-dimensional smooth manifold — not a Euclidean space. Laser plane normals live on the unit sphere $S^2$. Optimizing these parameters with standard Euclidean updates (adding $\boldsymbol{\delta}$ to the parameter vector) would violate the manifold constraints: rotations would become non-orthogonal, quaternions would lose unit norm, and normals would lose unit length.

**Manifold optimization** solves this by computing updates in the **tangent space** and then retracting back to the manifold.

## The Retract-Update Pattern

For a parameter $\mathbf{x}$ on a manifold $\mathcal{M}$:

1. **Compute update** $\boldsymbol{\delta}$ in the tangent space $T_\mathbf{x}\mathcal{M}$ (a Euclidean space of dimension equal to the manifold's degrees of freedom)
2. **Retract**: $\mathbf{x} \leftarrow \mathbf{x} \oplus \boldsymbol{\delta}$ where $\oplus$ maps from tangent space back to the manifold

This is also called **local parameterization**: the optimizer sees a Euclidean space of the correct dimension, and the manifold structure handles the constraint.

## SE(3): Rigid Transforms

**Ambient dimension**: 7 (quaternion $[q_x, q_y, q_z, q_w]$ + translation $[t_x, t_y, t_z]$)

**Tangent dimension**: 6 ($[\boldsymbol{\omega}, \mathbf{v}] \in \mathbb{R}^6$)

**Plus operation** ($\oplus$): Given current pose $T$ and tangent vector $\boldsymbol{\xi} = [\boldsymbol{\omega}, \mathbf{v}]$:

$$T' = \exp(\boldsymbol{\xi}^\wedge) \cdot T$$

where $\exp$ is the SE(3) exponential map (see [Rigid Transformations](rigid_transforms.md)):

$$\exp(\boldsymbol{\xi}^\wedge) = \begin{bmatrix} \exp([\boldsymbol{\omega}]_\times) & V\mathbf{v} \\ \mathbf{0}^T & 1 \end{bmatrix}$$

**Minus operation** ($\ominus$): The inverse — compute the tangent vector between two poses — uses the SE(3) logarithm.

### Implementation in calibration-rs

The `se3_exp` function in the factor module computes the exponential map generically over `RealField` for autodiff compatibility:

```rust
fn se3_exp<T: RealField>(xi: &[T; 6]) -> [[T; 4]; 4]
```

This uses Rodrigues' formula with special handling for the small-angle case ($\|\boldsymbol{\omega}\| < \epsilon$) to avoid division by zero.

## SO(3): Rotations

**Ambient dimension**: 4 (unit quaternion)

**Tangent dimension**: 3 (axis-angle $\boldsymbol{\omega} \in \mathbb{R}^3$)

**Plus operation**: $R' = \exp([\boldsymbol{\omega}]_\times) \cdot R$

SO(3) is used for rotation-only parameters. In calibration-rs, most poses use the full SE(3), but SO(3) is available for specialized problems.

## $S^2$: Unit Sphere

**Ambient dimension**: 3 (unit vector $\hat{\mathbf{n}} \in \mathbb{R}^3$, $\|\hat{\mathbf{n}}\| = 1$)

**Tangent dimension**: 2 ($\boldsymbol{\delta} \in \mathbb{R}^2$)

The $S^2$ manifold is used for **laser plane normals** — 3D unit vectors with 2 degrees of freedom.

**Plus operation**: Given current normal $\hat{\mathbf{n}}$ and tangent vector $\boldsymbol{\delta} \in \mathbb{R}^2$:

1. Compute orthonormal basis $(\mathbf{b}_1, \mathbf{b}_2)$ of the tangent plane at $\hat{\mathbf{n}}$
2. Tangent vector in 3D: $\mathbf{v} = \delta_1 \mathbf{b}_1 + \delta_2 \mathbf{b}_2$
3. Retract: $\hat{\mathbf{n}}' = \frac{\hat{\mathbf{n}} + \mathbf{v}}{\|\hat{\mathbf{n}} + \mathbf{v}\|}$

For small $\|\boldsymbol{\delta}\|$, this is equivalent to the exponential map on $S^2$.

### Basis Construction

The tangent plane basis is constructed deterministically:

1. Choose a reference vector not parallel to $\hat{\mathbf{n}}$ (e.g., $[1,0,0]$ or $[0,1,0]$)
2. $\mathbf{b}_1 = \hat{\mathbf{n}} \times \text{ref}$, normalized
3. $\mathbf{b}_2 = \hat{\mathbf{n}} \times \mathbf{b}_1$

## Euclidean Parameters

For intrinsics ($f_x, f_y, c_x, c_y$), distortion ($k_1, k_2, k_3, p_1, p_2$), and other unconstrained parameters, the manifold is simply $\mathbb{R}^n$ with:

$$\mathbf{x}' = \mathbf{x} + \boldsymbol{\delta}$$

No special treatment needed.

## ManifoldKind in the IR

The optimization IR specifies the manifold for each parameter block:

```rust
pub enum ManifoldKind {
    Euclidean,  // R^n, standard addition
    SE3,        // 7D ambient, 6D tangent
    SO3,        // 4D ambient, 3D tangent
    S2,         // 3D ambient, 2D tangent
}
```

The backend maps these to solver-specific manifold implementations.

## Fixed Parameters on Manifolds

For Euclidean parameters, individual indices can be fixed (e.g., fix $c_x$ but optimize $f_x$). For manifold parameters (SE3, SO3, S2), fixing is **all-or-nothing**: either the entire parameter block is optimized or the entire block is fixed. This is because the tangent space decomposition does not naturally support partial fixing on non-Euclidean manifolds.
