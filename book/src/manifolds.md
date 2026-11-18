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

**Plus operation** ($\oplus$): Given current pose $T$ and tangent vector $\boldsymbol{\xi} = [\boldsymbol{\omega}, \mathbf{v}]$, the step is applied on the right, in the frame of the pose:

$$T' = T \cdot \exp(\boldsymbol{\xi}^\wedge)$$

where $\exp$ is the SE(3) exponential map (see [Rigid Transformations](rigid_transforms.md)):

$$\exp(\boldsymbol{\xi}^\wedge) = \begin{bmatrix} \exp([\boldsymbol{\omega}]_\times) & V\mathbf{v} \\ \mathbf{0}^T & 1 \end{bmatrix}$$

The factrs backend applies this map as written. The tiny-solver backend composes $T$ with $(\exp([\boldsymbol{\omega}]_\times), \mathbf{v})$, taking the translation step without $V$; the two retractions agree to first order, so a step of zero is a step of zero in both and they reach the same minimizer. A pose's covariance is expressed in this right-perturbation tangent.

**Minus operation** ($\ominus$): The inverse — compute the tangent vector between two poses — uses the SE(3) logarithm.

## SO(3): Rotations

**Ambient dimension**: 4 (unit quaternion)

**Tangent dimension**: 3 (axis-angle $\boldsymbol{\omega} \in \mathbb{R}^3$)

**Plus operation**: $R' = R \cdot \exp([\boldsymbol{\omega}]_\times)$

SO(3) is used for rotation-only parameters. The calibration workflows estimate full SE(3) poses; an SO(3) block is available to a problem of your own ([Your Own Residual](custom_residual.md)).

## $S^2$: Unit Sphere

**Ambient dimension**: 3 (unit vector $\hat{\mathbf{n}} \in \mathbb{R}^3$, $\|\hat{\mathbf{n}}\| = 1$)

**Tangent dimension**: 2 ($\boldsymbol{\delta} \in \mathbb{R}^2$)

The $S^2$ manifold is used for **laser plane normals** — 3D unit vectors with 2 degrees of freedom. Both backends use the same retraction.

**Plus operation**: Given current normal $\hat{\mathbf{n}}$ and tangent vector $\boldsymbol{\delta} \in \mathbb{R}^2$:

1. Compute the orthonormal basis $(\mathbf{b}_1, \mathbf{b}_2)$ of the tangent plane at $\hat{\mathbf{n}}$ (below)
2. Tangent vector in 3D: $\mathbf{v} = \delta_1 \mathbf{b}_1 + \delta_2 \mathbf{b}_2$, with $\theta = \|\mathbf{v}\|$
3. Exponential map: $\hat{\mathbf{n}}' = \hat{\mathbf{n}} \cos\theta + \mathbf{v}\, \dfrac{\sin\theta}{\theta}$, renormalized to absorb rounding

Below $\theta = 10^{-6}$, $\cos\theta$ and $\sin\theta/\theta$ are replaced by their Taylor series $1 - \theta^2/2$ and $1 - \theta^2/6$, which keeps the map differentiable at $\boldsymbol{\delta} = 0$.

**Minus operation**: the logarithm at $\hat{\mathbf{n}}$ of a unit vector $\hat{\mathbf{m}}$ is $\boldsymbol{\delta} = (\mathbf{b}_1^T\mathbf{w}, \mathbf{b}_2^T\mathbf{w})$ with

$$\mathbf{w} = \frac{\theta}{\sin\theta}\left(\hat{\mathbf{m}} - \hat{\mathbf{n}}\,(\hat{\mathbf{n}}^T\hat{\mathbf{m}})\right), \qquad \theta = \operatorname{atan2}\left(\|\hat{\mathbf{n}} \times \hat{\mathbf{m}}\|,\ \hat{\mathbf{n}}^T\hat{\mathbf{m}}\right).$$

### Basis Construction

The tangent plane basis is constructed deterministically:

1. Choose the reference axis $\mathbf{a} = [0, 0, 1]$, or $\mathbf{a} = [1, 0, 0]$ when $|\hat{n}_z| \ge 0.9$
2. $\mathbf{b}_1 = \operatorname{normalize}\left(\mathbf{a} - \hat{\mathbf{n}}\,(\hat{\mathbf{n}}^T\mathbf{a})\right)$ (one Gram–Schmidt step)
3. $\mathbf{b}_2 = \hat{\mathbf{n}} \times \mathbf{b}_1$

## Euclidean Parameters

For intrinsics ($f_x, f_y, c_x, c_y$), distortion ($k_1, k_2, k_3, p_1, p_2$), and other unconstrained parameters, the manifold is simply $\mathbb{R}^n$ with:

$$\mathbf{x}' = \mathbf{x} + \boldsymbol{\delta}$$

No special treatment needed.

## Choosing a Manifold

A problem of your own names the manifold of each parameter block with `optim::problem::Manifold` ([Your Own Residual](custom_residual.md)):

```rust,ignore
pub enum Manifold {
    Euclidean,  // R^n, updated by addition
    Se3,        // [qx, qy, qz, qw, tx, ty, tz]: 7D ambient, 6D tangent
    UnitVector, // [x, y, z] on S²: 3D ambient, 2D tangent
    So3,        // [qx, qy, qz, qw]: 4D ambient, 3D tangent
}
```

The calibration workflows choose the manifolds of their own blocks, and each backend maps them to its own manifold implementation.

## Fixed Parameters on Manifolds

For Euclidean parameters, individual indices can be fixed (e.g., fix $c_x$ but optimize $f_x$). A manifold block (SE(3), SO(3), $S^2$) is fixed **as a whole or not at all**: no tangent coordinate corresponds to one stored component, so both backends reject a partially fixed manifold block with an `InvalidInput` error.
