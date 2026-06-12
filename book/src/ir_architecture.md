# Backend-Agnostic IR Architecture

calibration-rs separates the **definition** of optimization problems from their **execution** by a specific solver. Problems are expressed as an Intermediate Representation (IR) that is compiled to a solver-specific form. This design allows swapping backends without changing problem definitions and makes it straightforward to add new problem types.

## Three-Stage Pipeline

```
Problem Builder  →  ProblemIR  →  Backend.compile()  →  Backend.solve()
 (domain code)      (generic)     (solver-specific)     (optimization)
```

1. **Problem Builder**: Domain code (e.g., `build_planar_intrinsics_ir()`) constructs a `ProblemIR` from calibration data and initial parameter estimates
2. **Backend Compilation**: The backend (e.g., `TinySolverBackend`) translates the IR into solver-specific data structures
3. **Solving**: The backend runs the optimizer and returns a `BackendSolution`

## ProblemIR

The central type that describes a complete optimization problem:

```rust
pub struct ProblemIR {
    pub params: Vec<ParamBlock>,
    pub residuals: Vec<ResidualBlock>,
}
```

It is validated on construction to ensure all parameter references are valid, dimensions match, and manifold constraints are respected.

## ParamBlock

Each parameter block represents a group of optimization variables:

```rust
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
| `id` | Unique identifier within the problem |
| `name` | Human-readable name (e.g., `"cam"`, `"pose/0"`, `"plane"`) |
| `dim` | Ambient dimension (e.g., 4 for intrinsics, 7 for SE3) |
| `manifold` | Geometry of the parameter space |
| `fixed` | Which indices (or the whole block) are held constant |
| `bounds` | Optional box constraints on parameter values |

### Naming Conventions

Problem builders use consistent naming:

- `"intrinsics"` or `"cam"` — intrinsics ($f_x, f_y, c_x, c_y$), dimension 4
- `"distortion"` or `"dist"` — distortion ($k_1, k_2, k_3, p_1, p_2$), dimension 5
- `"sensor"` — Scheimpflug tilt ($\tau_x, \tau_y$), dimension 2
- `"pose/0"`, `"pose/1"`, ... — per-view SE(3) poses (planar, rig), dimension 7
- `"pose_0"`, `"pose_1"`, ... — per-view SE(3) poses (laserline), dimension 7
- `"plane_normal"` — laser plane unit normal, dimension 3, $S^2$ manifold
- `"plane_distance"` — laser plane distance, dimension 1
- `"handeye"` — hand-eye SE(3), dimension 7
- `"extrinsics/0"`, ... — per-camera SE(3) extrinsics, dimension 7

## FixedMask

Controls which parameters are held constant during optimization:

```rust
// Construct via factory methods:
FixedMask::all_free()              // nothing fixed
FixedMask::all_fixed(dim)          // everything fixed
FixedMask::fix_indices(&[2])       // fix specific indices
```

- **Euclidean parameters**: Individual indices can be fixed. For example, `FixedMask::fix_indices(&[2])` on intrinsics fixes $c_x$ while allowing $f_x, f_y, c_y$ to vary.
- **Manifold parameters**: All-or-nothing. If any index is fixed, the entire block is fixed. (The tangent space does not support partial fixing.)

## ResidualBlock

Each residual block connects parameter blocks through a factor:

```rust
pub struct ResidualBlock {
    pub params: Vec<ParamId>,
    pub loss: RobustLoss,
    pub factor: FactorKind,
    pub residual_dim: usize,
}
```

The `params` vector references `ParamBlock`s by their `ParamId`. The ordering must match what the factor expects.

## FactorKind

A factor is one of four residual families. The camera model and pose chain
are **data** carried by the factor, not separate enum variants:

```rust
pub enum FactorKind {
    /// 2D pixel reprojection residual.
    ReprojPoint {
        model: CameraModelDesc,
        chain: ReprojChain,
        pw: [f64; 3], uv: [f64; 2], w: f64,
    },
    /// 1D point-to-plane laser residual (meters).
    LaserPointToPlane {
        model: CameraModelDesc,
        chain: LaserChain,
        laser_pixel: [f64; 2], w: f64,
    },
    /// 1D line-distance laser residual (pixels).
    LaserLineDistance {
        model: CameraModelDesc,
        chain: LaserChain,
        laser_pixel: [f64; 2], w: f64,
    },
    /// Zero-mean prior on a 6D se(3) tangent block.
    Se3TangentPrior { sqrt_info: [f64; 6] },
}
```

Each factor also carries the **per-residual data** (3D point coordinates,
observed pixel, weight, measured robot pose, …) that is not part of the
optimizable parameters.

### Camera model as data

```rust
pub struct CameraModelDesc {
    pub projection: ProjectionKind, // Pinhole               → intrinsics (dim 4)
    pub distortion: DistortionKind, // None | BrownConrady5  → dim 0 | 5
    pub sensor: SensorKind,         // None | Scheimpflug2   → dim 0 | 2
}
```

A slot with dimension 0 contributes no parameter block. The backend maps the
descriptor to zero-sized kernel types and monomorphizes the residual over
them once per factor — autodiff stays generic over the scalar type with no
per-evaluation model dispatch. Adding a camera model later (rational
distortion, thin-prism, division, fisheye projection) adds a descriptor
variant + kernel + one dispatch row; no new factor kinds.

### Chains as data

`ReprojChain` selects the pose chain and the trailing parameter blocks:

| Chain | Chain blocks | Transform |
|-------|-------------|-----------|
| `SinglePose` | [pose] | $P_c = T \cdot P_w$ |
| `TwoSe3` | [extrinsics, pose] | $P_c = T_{extr}^{-1} T_{pose} P_w$ |
| `HandEye { base_se3_gripper, mode }` | [extrinsics, handeye, target] | mode-dependent hand-eye chain |
| `HandEyeRobotDelta { … }` | [extrinsics, handeye, target, robot_delta] | hand-eye + per-view se(3) correction |

`LaserChain` mirrors this for the laser families (`SinglePose`,
`RigHandEye`, `RigHandEyeRobotDelta`); every laser chain ends with the
`plane_normal` (S², dim 3) and `plane_distance` (dim 1) blocks, with
`robot_delta` last when present.

### Layout-derived validation

`FactorKind::param_layout()` derives the full expected block list —
`[intrinsics, distortion?, sensor?, <chain blocks…>]` — as
`Vec<ParamSlotSpec { dim, manifold, role }>`. Validation is a single zip of
this layout against the referenced parameter blocks; there are no
per-variant validation arms to maintain.

See the [Factor Catalog](factor_catalog.md) for the full chain math and
residual definitions.

## Validation

`ProblemIR::new()` validates:

- All `ParamId` references in residual blocks resolve to existing parameter blocks
- Parameter dimensions match what each factor expects
- Manifold kinds are compatible with the factor requirements
- No duplicate parameter IDs

## Initial Values

Parameter blocks carry only their structural definition (dimension, manifold, fixing). The **initial values** are passed separately as a `HashMap<String, DVector<f64>>` (keyed by parameter block **name**) to the backend's `solve()` method.

## Example: Planar Intrinsics IR

For a 10-view planar intrinsics problem with 48 points per view:

```
Parameters:
  - "cam": dim=4, Euclidean (fx, fy, cx, cy)
  - "dist": dim=5, Euclidean (k1, k2, k3, p1, p2)
  - "pose/0"..."pose/9": dim=7, SE3

Residuals: 480 blocks
  - Each: ReprojPoint { model: PINHOLE4_DIST5, chain: SinglePose, pw, uv, w=1.0 }
  - Params: [cam_id, dist_id, pose_k_id]
  - Residual dim: 2

Total residuals: 960 scalar values
Total parameters: 4 + 5 + 10*7 = 79
```
