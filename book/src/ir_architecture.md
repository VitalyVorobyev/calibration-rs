# Backend-Agnostic IR Architecture

calibration-rs separates the **definition** of optimization problems from their **execution** by a specific solver. Problems are expressed as an Intermediate Representation (IR) that is compiled to a solver-specific form. This design allows swapping backends without changing problem definitions and makes it straightforward to add new problem types.

## Three-Stage Pipeline

```text
Problem Builder  →  ProblemIR  →  Backend.compile()  →  Backend.solve()
 (domain code)      (generic)     (solver-specific)     (optimization)
```

1. **Problem Builder**: Domain code in `vision-calibration-optim`'s `problems/` (one builder per calibration problem) constructs a `ProblemIR` from calibration data and initial parameter estimates
2. **Backend Compilation**: The backend the solver config selects (tiny-solver or factrs) translates the IR into its own variables and residuals
3. **Solving**: The shared Levenberg–Marquardt loop runs on that backend's linearization and returns a `BackendSolution` (see [Solver Backends](solver_backends.md))

These types are internal to `vision-calibration-optim`. A problem of your own goes through the public builder of [Your Own Residual](custom_residual.md), which lowers to the same IR.

## ProblemIR

The central type that describes a complete optimization problem:

```rust,ignore
pub struct ProblemIR {
    pub params: Vec<ParamBlock>,
    pub residuals: Vec<ResidualBlock>,
}
```

`ProblemIR::new()` creates an empty problem; builders add blocks with `add_param_block` and `add_residual_block`, then call `validate()` (see [Validation](#validation)). Both backends validate the IR again before compiling it.

## ParamBlock

Each parameter block represents a group of optimization variables:

```rust,ignore
pub struct ParamBlock {
    pub id: ParamId,
    pub name: String,
    pub dim: usize,
    pub manifold: ManifoldKind,
    pub fixed: FixedMask,
    pub bounds: Option<Vec<Bound>>,  // per-index box constraints
}
```

| Field | Description |
|-------|-------------|
| `id` | Index of the block in `params` |
| `name` | Name, unique within the problem (e.g., `"cam"`, `"pose/0"`, `"plane_n"`) |
| `dim` | Ambient dimension (e.g., 4 for intrinsics, 7 for SE3) |
| `manifold` | Geometry of the parameter space: `Euclidean`, `SE3`, `SO3` or `S2` |
| `fixed` | Which indices (or the whole block) are held constant |
| `bounds` | Optional box constraints on parameter values |

### Block Names

The problem builders name their blocks as follows; `{i}` is a camera index, `{v}` a view index:

| Name | Dim | Manifold | Holds |
|------|-----|----------|-------|
| `cam` | 4 | Euclidean | intrinsics $f_x, f_y, c_x, c_y$ |
| `proj` | model | Euclidean | projection parameters, when the projection has any |
| `dist` | model | Euclidean | distortion coefficients (5 for Brown-Conrady), absent without distortion |
| `sensor` | 2 | Euclidean | Scheimpflug tilt $\tau_x, \tau_y$, absent for an identity sensor |
| `pose/{v}` | 7 | SE3 | per-view `cam_se3_target` (planar intrinsics, laserline device) |
| `extr/{i}` | 7 | SE3 | per-camera extrinsics (rig, hand-eye) |
| `rig_pose/{v}` | 7 | SE3 | per-view `rig_se3_target` (rig extrinsics) |
| `handeye` | 7 | SE3 | the hand-eye transform |
| `target`, `target/{v}` | 7 | SE3 | the target pose of a hand-eye problem, shared or per view |
| `target_ref` | 7 | SE3 | the target pose of the joint rig, hand-eye and laser problem |
| `robot_delta/{v}` | 6 | Euclidean | per-view se(3) correction of the measured robot pose |
| `plane_n`, `plane_n/{i}` | 3 | S2 | laser-plane unit normal, one or per camera |
| `plane_d`, `plane_d/{i}` | 1 | Euclidean | laser-plane distance, one or per camera |

A problem with several cameras indexes the camera blocks per camera: `cam/{i}`, `proj/{i}`, `dist/{i}`, `sensor/{i}`.

## FixedMask

Controls which parameters are held constant during optimization:

```rust,ignore
// Construct via factory methods:
FixedMask::all_free()              // nothing fixed
FixedMask::all_fixed(dim)          // everything fixed
FixedMask::fix_indices(&[2])       // fix specific indices
```

- **Euclidean parameters**: Individual indices can be fixed. For example, `FixedMask::fix_indices(&[2])` on intrinsics fixes $c_x$ while allowing $f_x, f_y, c_y$ to vary.
- **Manifold parameters**: All or nothing. Both backends reject a partially fixed SE3, SO3 or S2 block with an `InvalidInput` error, since no tangent coordinate corresponds to one stored component.

## ResidualBlock

Each residual block connects parameter blocks through a factor:

```rust,ignore
pub struct ResidualBlock {
    pub params: Vec<ParamId>,
    pub loss: RobustLoss,
    pub factor: FactorKind,
    pub residual_dim: usize,
}
```

The `params` vector references `ParamBlock`s by their `ParamId`. The ordering must match what the factor expects.

## Factors

A `FactorKind` variant names the residual family (what is measured); the camera model (`CameraModelSpec`) and the pose chain (`ReprojChain` / `LaserChain`) are **data** carried by the factor, together with the per-residual constants (the 3D point, the observed pixel, the square-root information, the measured robot pose). Adding a camera model therefore adds no factor kind.

`FactorKind::param_layout()` derives each factor's expected block list, `[intrinsics, projection?, distortion?, sensor?, <chain blocks…>]`, as `Vec<ParamSlotSpec { dim, manifold, role }>`. The [Factor Catalog](factor_catalog.md) lists every factor kind and chain with its blocks and residual.

## Validation

`ProblemIR::validate()` checks that:

- each block's `id` is its index, its manifold fits its dimension, and its fixed indices and bounds are in range (a bound with `lower > upper` or a NaN is rejected);
- each residual's robust loss has a positive scale and its `residual_dim` matches the factor;
- every referenced `ParamId` exists, and the referenced blocks match the factor's `param_layout()` in count, dimension and manifold (a custom residual reads its blocks on any manifold);
- the reprojection residuals' square-root information is finite, and either all of them are whitened by stated covariances or none are;
- a free-point reprojection uses a single-pose or rig chain, and a laser line-distance residual a projection that images lines as lines.

## Initial Values

Parameter blocks carry only their structural definition (dimension, manifold, fixing). The **initial values** are passed separately as a `HashMap<String, DVector<f64>>` (keyed by parameter block **name**) to the backend's `solve()` method.

## Example: Planar Intrinsics IR

For a 10-view planar intrinsics problem with 48 points per view and a pinhole, Brown-Conrady camera:

```text
Parameters:
  - "cam": dim=4, Euclidean (fx, fy, cx, cy)
  - "dist": dim=5, Euclidean (k1, k2, k3, p1, p2)
  - "pose/0"..."pose/9": dim=7, SE3

Residuals: 480 blocks
  - Each: ReprojPoint { model: CameraModelSpec { Pinhole, BrownConrady5, Identity },
                        chain: SinglePose, pw, uv,
                        sqrt_info: Isotropic { scale: 1.0 } }
  - Params: [cam_id, dist_id, pose_k_id]
  - Residual dim: 2

Total residuals: 960 scalar values
Total parameters: 4 + 5 + 10*7 = 79
```
