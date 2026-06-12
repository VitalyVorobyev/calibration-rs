# Factor Catalog Reference

This chapter is a complete reference for the factor kinds (residual
computations) supported by the optimization IR. A factor is described along
three independent axes carried **as data**, not baked into variant names:

1. **Camera model** (`CameraModelDesc`) — which projection / distortion /
   sensor kernels apply and which leading parameter blocks exist.
2. **Pose chain** (`ReprojChain` / `LaserChain`) — how a target-frame point
   reaches the camera frame and which trailing parameter blocks exist.
3. **Residual family** (the `FactorKind` variant) — what is measured.

```rust,ignore
pub enum FactorKind {
    ReprojPoint       { model: CameraModelDesc, chain: ReprojChain, pw: [f64; 3], uv: [f64; 2], w: f64 },
    LaserPointToPlane { model: CameraModelDesc, chain: LaserChain, laser_pixel: [f64; 2], w: f64 },
    LaserLineDistance { model: CameraModelDesc, chain: LaserChain, laser_pixel: [f64; 2], w: f64 },
    Se3TangentPrior   { sqrt_info: [f64; 6] },
}
```

## Camera-Model Descriptor

```rust,ignore
pub struct CameraModelDesc {
    pub projection: ProjectionKind, // Pinhole               → intrinsics block (dim 4)
    pub distortion: DistortionKind, // None | BrownConrady5  → distortion block (dim 0 | 5)
    pub sensor: SensorKind,         // None | Scheimpflug2   → sensor block (dim 0 | 2)
}
```

A slot with dimension 0 contributes **no parameter block**. The expected
layout of every factor is therefore derived, never hand-listed:

```text
[intrinsics, distortion?, sensor?, <chain blocks…>]
```

`FactorKind::param_layout()` returns this layout as
`Vec<ParamSlotSpec { dim, manifold, role }>`, and `ProblemIR::validate`
checks each referenced block against it. Common descriptors are available as
constants: `CameraModelDesc::PINHOLE4`, `::PINHOLE4_DIST5`,
`::PINHOLE4_DIST5_SCHEIMPFLUG2`.

Each descriptor maps to zero-sized **kernel types** (one per slot) whose
static methods are generic over the autodiff scalar `T: RealField`. The
backend matches the descriptor once per factor and monomorphizes the residual
over the kernels — there is no per-evaluation dispatch on the camera-model
axis. Adding a camera model means one descriptor variant, one kernel type,
and one row in the backend dispatch table (see
[Adding a New Solver Backend](new_backend.md)).

## Reprojection Chains

`ReprojPoint` computes the weighted pixel residual

$$\mathbf{r} = \sqrt{w} \cdot \left( \mathbf{p}_{\text{obs}} - \pi(\boldsymbol{\theta}, \mathbf{P}_c) \right) \in \mathbb{R}^2$$

where $\pi$ is the camera model selected by the descriptor and
$\mathbf{P}_c$ comes from the chain:

### `ReprojChain::SinglePose`

**Chain blocks**: `[camera_se3_target(7)]`

$$\mathbf{P}_c = T_{C,T} \cdot \mathbf{P}_w$$

**Use case**: single-camera calibration (planar / Scheimpflug intrinsics,
laserline device bundles).

### `ReprojChain::TwoSe3`

**Chain blocks**: `[extrinsics(7), pose(7)]`

$$\mathbf{P}_c = T_{\text{extr}}^{-1} \cdot T_{\text{pose}} \cdot \mathbf{P}_w$$

where `extrinsics` maps camera → rig and `pose` maps target → rig.

**Use case**: multi-camera rig extrinsics.

### `ReprojChain::HandEye { base_se3_gripper, mode }`

**Chain blocks**: `[extrinsics(7), handeye(7), target(7)]`

**Chain data**: the measured robot pose $T_{B,G}$ and the [`HandEyeMode`].

Eye-in-hand:
$$T_{C,T} = T_{\text{extr}}^{-1} \cdot T_{\text{handeye}}^{-1} \cdot T_{B,G}^{-1} \cdot T_{\text{target}}$$

Eye-to-hand:
$$T_{C,T} = T_{\text{extr}}^{-1} \cdot T_{\text{handeye}} \cdot T_{B,G} \cdot T_{\text{target}}$$

For single-camera hand-eye, $T_{\text{extr}} = I$.

**Use case**: hand-eye calibration (camera on robot arm, or rig observing a
robot-mounted target).

### `ReprojChain::HandEyeRobotDelta { base_se3_gripper, mode }`

**Chain blocks**: `[extrinsics(7), handeye(7), target(7), robot_delta(6)]`

Same as `HandEye`, but with a per-view se(3) tangent correction applied to
the robot pose:

$$T_{B,G}' = \exp(\boldsymbol{\xi}_{\text{delta}}) \cdot T_{B,G}$$

The correction is regularized by an `Se3TangentPrior` factor.

**Use case**: hand-eye calibration with imprecise robot kinematics.

## Laser Factors

Both laser families have residual dimension 1. Their chains mirror the
reprojection chains but end with the laser-plane blocks
`[plane_normal(3, S²), plane_distance(1)]`; a robot-delta block, when
present, always comes last:

- `LaserChain::SinglePose` — blocks `[camera_se3_target, plane_normal, plane_distance]`.
- `LaserChain::RigHandEye { base_se3_gripper, mode }` — blocks
  `[cam_se3_rig, handeye, target_ref, plane_normal, plane_distance]`; the
  per-view pose is composed exactly like the hand-eye reprojection chain.
- `LaserChain::RigHandEyeRobotDelta { … }` — adds `robot_delta(6)` last.

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

## Summary Table

| Factor | Blocks | Res. dim | Units |
|--------|--------|---------|-------|
| `ReprojPoint` | camera blocks + 1–4 chain blocks | 2 | pixels |
| `LaserPointToPlane` | camera blocks + 3–6 chain blocks | 1 | meters |
| `LaserLineDistance` | camera blocks + 3–6 chain blocks | 1 | pixels |
| `Se3TangentPrior` | 1 | 6 | normalized |

Camera blocks = 1 (intrinsics) + 1 if the descriptor has distortion + 1 if it
has a sensor. For example, `ReprojPoint` with `PINHOLE4_DIST5_SCHEIMPFLUG2`
and `HandEyeRobotDelta` connects 7 blocks:
`[intrinsics, distortion, sensor, extrinsics, handeye, target, robot_delta]`.
