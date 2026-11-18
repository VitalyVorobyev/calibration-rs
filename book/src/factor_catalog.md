# Factor Catalog Reference

This chapter is a complete reference for the factor kinds (residual
computations) supported by the optimization IR. A factor is described along
three independent axes carried **as data**, not baked into variant names:

1. **Camera model** (`CameraModelSpec`) — which projection / distortion /
   sensor kinds apply and which leading parameter blocks exist.
2. **Pose chain** (`ReprojChain` / `LaserChain`) — how a target-frame point
   reaches the camera frame and which trailing parameter blocks exist.
3. **Residual family** (the `FactorKind` variant) — what is measured.

There are six residual families:

```rust,ignore
pub enum FactorKind {
    ReprojPoint       { model: CameraModelSpec, chain: ReprojChain, pw: [f64; 3], uv: [f64; 2], sqrt_info: ReprojSqrtInfo },
    ReprojFreePoint   { model: CameraModelSpec, chain: ReprojChain, uv: [f64; 2], sqrt_info: ReprojSqrtInfo },
    LaserPointToPlane { model: CameraModelSpec, chain: LaserChain, laser_pixel: [f64; 2], w: f64 },
    LaserLineDistance { model: CameraModelSpec, chain: LaserChain, laser_pixel: [f64; 2], w: f64 },
    Se3TangentPrior   { sqrt_info: [f64; 6], residual_sigma: f64 },
    Custom(CustomResidual),
}
```

## Camera-Model Spec

```rust,ignore
pub struct CameraModelSpec {
    pub projection: ProjectionKind, // Pinhole | Equidistant
    pub distortion: DistortionKind, // None | BrownConrady5 | Rational8 | ThinPrism9 | Division1 | KannalaBrandt4
    pub sensor: SensorKind,         // Identity | Scheimpflug
}
```

A slot whose kind has no parameters contributes **no parameter block**. The
expected layout of every factor is therefore derived, never hand-listed:

```text
[intrinsics, projection?, distortion?, sensor?, <chain blocks…>]
```

`FactorKind::param_layout()` returns this layout as
`Vec<ParamSlotSpec { dim, manifold, role }>`, and `ProblemIR::validate`
checks each referenced block against it.

Evaluation dispatches per slot through the kind's own methods in
`vision-calibration-core` (`ProjectionKind::project`,
`DistortionKind::distort`, `SensorKind::to_sensor`, generic over the autodiff
scalar `T: RealField`), so the optimizer holds no camera math: a new camera
model is a new kind variant in core, not a new factor.

## Reprojection Chains

`ReprojPoint` computes the weighted pixel residual

$$\mathbf{r} = L \left( \mathbf{p}_{\text{obs}} - \pi(\boldsymbol{\theta}, \mathbf{P}_c) \right) \in \mathbb{R}^2$$

where $L$ (`ReprojSqrtInfo`) is $\sqrt{w}\,I$ for a point without a stated
covariance, or the upper-triangular square root of $\Sigma^{-1}$ ($L^\top L =
\Sigma^{-1}$) for a point stating the covariance $\Sigma$, times $\sqrt{w}$ for a
family weight $w$; one problem holds one kind or the other. $\pi$ is the camera
model selected by the spec and
$\mathbf{P}_c$ comes from the chain:

### `ReprojChain::SinglePose`

**Chain blocks**: `[cam_se3_target(7)]`

$$\mathbf{P}_c = T_{C,T} \cdot \mathbf{P}_w$$

**Use case**: single-camera calibration (planar / Scheimpflug intrinsics,
laserline device bundles).

### `ReprojChain::TwoSe3`

**Chain blocks**: `[extrinsics(7), pose(7)]`

$$\mathbf{P}_c = T_{\text{extr}}^{-1} \cdot T_{\text{pose}} \cdot \mathbf{P}_w$$

where `extrinsics` maps camera → rig and `pose` maps target → rig.

**Use case**: multi-camera rig extrinsics.

### `ReprojChain::Rig`

**Chain blocks**: `[cam_se3_rig(7), rig_se3_target(7)]`

$$\mathbf{P}_c = T_{C,R} \cdot T_{R,T} \cdot \mathbf{P}_w$$

**Use case**: a camera of a rig whose pose is given as `cam_se3_rig`, as the
public problem builder adds it for a calibrated rig.

### `ReprojChain::Handeye { base_se3_gripper, mode }`

**Chain blocks**: `[extrinsics(7), handeye(7), target(7)]`

**Chain data**: the measured robot pose $T_{B,G}$ and the `HandeyeMode`.

Eye-in-hand:
$$T_{C,T} = T_{\text{extr}}^{-1} \cdot T_{\text{handeye}}^{-1} \cdot T_{B,G}^{-1} \cdot T_{\text{target}}$$

Eye-to-hand:
$$T_{C,T} = T_{\text{extr}}^{-1} \cdot T_{\text{handeye}} \cdot T_{B,G} \cdot T_{\text{target}}$$

For single-camera hand-eye, $T_{\text{extr}} = I$.

**Use case**: hand-eye calibration (camera on robot arm, or rig observing a
robot-mounted target).

### `ReprojChain::HandeyeRobotDelta { base_se3_gripper, mode }`

**Chain blocks**: `[extrinsics(7), handeye(7), target(7), robot_delta(6)]`

Same as `Handeye`, but with a per-view se(3) tangent correction applied to
the robot pose in the gripper frame:

$$T_{B,G}' = T_{B,G} \cdot \exp(\boldsymbol{\xi}_{\text{delta}})$$

The correction is regularized by an `Se3TangentPrior` factor.

**Use case**: hand-eye calibration with imprecise robot kinematics.

## ReprojFreePoint

The same residual as `ReprojPoint`, for a point that is **estimated**
rather than known: the target-frame point is the last parameter block, a
Euclidean 3-vector, instead of factor data.

**Chain blocks**: those of `ReprojChain::SinglePose` or `ReprojChain::Rig`
(the only chains it accepts), then `[point(3)]`.

**Use case**: triangulation and bundle adjustment over calibrated cameras,
which the public problem builder lowers to this factor for a point given as
a block.

## Laser Factors

Both laser families have residual dimension 1. Their chains mirror the
reprojection chains but end with the laser-plane blocks
`[plane_normal(3, S²), plane_distance(1)]`; a robot-delta block, when
present, always comes last:

- `LaserChain::SinglePose` — blocks `[cam_se3_target, plane_normal, plane_distance]`.
- `LaserChain::RigHandeye { base_se3_gripper, mode }` — blocks
  `[cam_se3_rig, handeye, target_ref, plane_normal, plane_distance]`; the
  per-view pose is composed exactly like the hand-eye reprojection chain.
- `LaserChain::RigHandeyeRobotDelta { … }` — adds `robot_delta(6)` last.

### LaserPointToPlane

**Computation**:
1. Undistort the laser pixel to normalized coordinates (inverting sensor and
   distortion kernels)
2. Back-project to a ray in camera frame
3. Intersect the ray with the target plane (Z = 0 in the target frame)
4. Compute the 3D point in camera frame
5. Measure the signed distance from the 3D point to the laser plane

**Residual**: $r = \sqrt{w} \cdot (\hat{\mathbf{n}}^T \mathbf{P}_c + d)$ — distance in meters.

### LaserLineDistance

**Computation**:
1. Compute the 3D intersection line of laser plane and target plane (camera frame)
2. Project this line onto the $Z = 1$ normalized camera plane
3. Undistort the laser pixel to normalized coordinates
4. Measure the perpendicular distance from pixel to projected line (2D geometry)
5. Scale by $\sqrt{f_x \cdot f_y}$ for pixel-comparable units

**Residual**: $r = \sqrt{w} \cdot d_{\perp} \cdot \sqrt{f_x f_y}$ — distance in effective pixels.

**Advantages over LaserPointToPlane**: residuals in pixel units (directly
comparable to reprojection error) and simpler 2D geometry.

**Default**: this is the recommended laser residual type
(`LaserlineResidualType::LineDistNormalized`).

## Prior Factors

### Se3TangentPrior

**Parameters**: `[robot_delta(6)]` (Euclidean se(3) tangent block)

**Computation**: element-wise scaling of the tangent block by the diagonal
square-root information:

$$\mathbf{r} = \operatorname{diag}(\mathbf{s}) \cdot \boldsymbol{\xi} \in \mathbb{R}^6$$

**Use case**: zero-mean Gaussian prior on robot pose corrections
(`robot_delta`) to penalize deviations from the nominal robot kinematics.

## Custom Residuals

### Custom

**Parameters**: the blocks the residual declares, of the dimensions it
declares, on any manifold (it reads their stored values).

**Computation**: the user's own residual, written through the public
`optim::problem` builder (see [Your Own Residual](custom_residual.md)).
Custom residuals run on the tiny-solver backend only: factrs fixes each
variable's size at compile time, so it rejects an IR that holds one.

## Summary Table

| Factor | Blocks | Res. dim | Units |
|--------|--------|---------|-------|
| `ReprojPoint` | camera blocks + 1–4 chain blocks | 2 | pixels (whitened: dimensionless) |
| `ReprojFreePoint` | camera blocks + 1–2 chain blocks + point | 2 | pixels (whitened: dimensionless) |
| `LaserPointToPlane` | camera blocks + 3–6 chain blocks | 1 | meters |
| `LaserLineDistance` | camera blocks + 3–6 chain blocks | 1 | pixels |
| `Se3TangentPrior` | 1 | 6 | normalized |
| `Custom` | as declared | as declared | the residual's own |

Camera blocks = 1 (intrinsics) + 1 for each of projection, distortion and
sensor whose kind has parameters. For example, `ReprojPoint` with a pinhole,
Brown-Conrady, Scheimpflug camera and `HandeyeRobotDelta` connects 7 blocks:
`[intrinsics, distortion, sensor, extrinsics, handeye, target, robot_delta]`.
