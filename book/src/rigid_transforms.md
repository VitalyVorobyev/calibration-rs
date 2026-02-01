# Rigid Transformations and SE(3)

Camera calibration is fundamentally about estimating rigid transformations — rotations and translations that relate different coordinate frames: camera, world, robot gripper, calibration target, rig, etc. This chapter covers the rotation representations and transformation algebra used throughout calibration-rs.

## Rotation Representations

### Rotation Matrix $R \in \text{SO}(3)$

A $3 \times 3$ orthogonal matrix with determinant $+1$:

$$R^T R = I, \quad \det(R) = 1$$

The set of all such matrices forms the **special orthogonal group** $\text{SO}(3)$. It has 3 degrees of freedom (9 entries minus 6 constraints).

### Unit Quaternion

A 4-element vector $\mathbf{q} = [q_x, q_y, q_z, q_w]^T$ with unit norm $\|\mathbf{q}\| = 1$. The rotation of a vector $\mathbf{v}$ is:

$$\mathbf{v}' = \mathbf{q} \otimes \mathbf{v} \otimes \mathbf{q}^*$$

where $\otimes$ is quaternion multiplication and $\mathbf{q}^*$ is the conjugate.

Quaternions are preferred for optimization because they avoid gimbal lock and have simpler interpolation properties than Euler angles. calibration-rs (via nalgebra) stores quaternions as `[i, j, k, w]` internally, which maps to $[q_x, q_y, q_z, q_w]$.

### Axis-Angle

A rotation of angle $\theta$ around unit axis $\hat{\mathbf{a}}$ is represented as:

$$\boldsymbol{\omega} = \theta \hat{\mathbf{a}} \in \mathbb{R}^3$$

with $\theta = \|\boldsymbol{\omega}\|$ and $\hat{\mathbf{a}} = \boldsymbol{\omega} / \theta$.

The **Rodrigues formula** converts axis-angle to rotation matrix:

$$R = I + \frac{\sin\theta}{\theta} [\boldsymbol{\omega}]_\times + \frac{1 - \cos\theta}{\theta^2} [\boldsymbol{\omega}]_\times^2$$

where $[\boldsymbol{\omega}]_\times$ is the skew-symmetric matrix of $\boldsymbol{\omega}$:

$$[\boldsymbol{\omega}]_\times = \begin{bmatrix} 0 & -\omega_z & \omega_y \\ \omega_z & 0 & -\omega_x \\ -\omega_y & \omega_x & 0 \end{bmatrix}$$

## Rigid Transformations: SE(3)

A rigid body transformation combines a rotation $R$ and translation $\mathbf{t}$:

$$T = \begin{bmatrix} R & \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix} \in \text{SE}(3)$$

The **special Euclidean group** $\text{SE}(3)$ has 6 degrees of freedom (3 rotation + 3 translation).

### Applying a Transform

A point $\mathbf{P}$ transformed by $T$:

$$\mathbf{P}' = R \mathbf{P} + \mathbf{t}$$

or in homogeneous coordinates:

$$\begin{bmatrix} \mathbf{P}' \\ 1 \end{bmatrix} = T \begin{bmatrix} \mathbf{P} \\ 1 \end{bmatrix}$$

### Composition

Transforms compose by matrix multiplication:

$$T_{A,C} = T_{A,B} \cdot T_{B,C}$$

### Inverse

$$T^{-1} = \begin{bmatrix} R^T & -R^T \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix}$$

## Naming Convention

calibration-rs uses the notation $T_{A,B}$ to mean "the transform **from** frame $B$ **to** frame $A$":

$$\mathbf{P}_A = T_{A,B} \cdot \mathbf{P}_B$$

Common transforms:

| Symbol | Meaning |
|--------|---------|
| $T_{C,W}$ | World to camera (camera extrinsics) |
| $T_{C,B}$ | Calibration board to camera |
| $T_{R,C}$ | Camera to rig (rig extrinsics) |
| $T_{G,C}$ | Camera to gripper (hand-eye) |
| $T_{B,T}$ | Target to base (hand-eye target pose) |

## nalgebra Type: `Iso3`

calibration-rs uses nalgebra's `Isometry3<f64>` (aliased as `Iso3`) for SE(3) transforms:

```rust
pub type Iso3 = nalgebra::Isometry3<f64>;
```

This stores a `UnitQuaternion` (rotation) and a `Translation3` (translation) separately, avoiding the redundancy of a full $4 \times 4$ matrix.

## SE(3) Storage for Optimization

In the optimization IR, SE(3) parameters are stored as a 7-element vector:

$$[q_x, q_y, q_z, q_w, t_x, t_y, t_z]$$

The quaternion $[q_x, q_y, q_z, q_w]$ is unit-constrained. The optimizer uses the SE(3) manifold (see [Manifold Optimization](manifolds.md)) to maintain this constraint during parameter updates.

## The Exponential Map

The **Lie algebra** $\mathfrak{se}(3)$ provides a 6-dimensional tangent space at the identity. A tangent vector $\boldsymbol{\xi} = [\boldsymbol{\omega}, \mathbf{v}]^T \in \mathbb{R}^6$ maps to an SE(3) element via the exponential map:

$$T = \exp(\boldsymbol{\xi}^\wedge) = \begin{bmatrix} \exp([\boldsymbol{\omega}]_\times) & V\mathbf{v} \\ \mathbf{0}^T & 1 \end{bmatrix}$$

where $V$ is the left Jacobian of SO(3):

$$V = I + \frac{1 - \cos\theta}{\theta^2} [\boldsymbol{\omega}]_\times + \frac{\theta - \sin\theta}{\theta^3} [\boldsymbol{\omega}]_\times^2$$

The exponential map is central to manifold optimization: parameter updates are computed in the tangent space and then retracted to the manifold (see [Manifold Optimization](manifolds.md)).
