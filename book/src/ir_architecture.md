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

An enum of all supported residual computations:

```rust
pub enum FactorKind {
    // Reprojection factors — all carry per-observation data
    ReprojPointPinhole4 { pw: [f64; 3], uv: [f64; 2], w: f64 },
    ReprojPointPinhole4Dist5 { pw: [f64; 3], uv: [f64; 2], w: f64 },
    ReprojPointPinhole4Dist5Scheimpflug2 { pw: [f64; 3], uv: [f64; 2], w: f64 },
    ReprojPointPinhole4Dist5TwoSE3 { pw: [f64; 3], uv: [f64; 2], w: f64 },
    ReprojPointPinhole4Dist5HandEye {
        pw: [f64; 3], uv: [f64; 2], w: f64,
        base_to_gripper_se3: [f64; 7], mode: HandEyeMode,
    },
    ReprojPointPinhole4Dist5HandEyeRobotDelta {
        pw: [f64; 3], uv: [f64; 2], w: f64,
        base_to_gripper_se3: [f64; 7], mode: HandEyeMode,
    },

    // Laser factors
    LaserPlanePixel { laser_pixel: [f64; 2], w: f64 },
    LaserLineDist2D { laser_pixel: [f64; 2], w: f64 },

    // Prior factors
    Se3TangentPrior { sqrt_info: [f64; 6] },
}
```

Each variant carries the **per-residual data** (3D point coordinates, observed pixel, weight, etc.) that is not part of the optimizable parameters. The factor also specifies which parameter blocks it expects and validates their dimensions.

### Reprojection Factors

All reprojection factors compute the pixel residual:

$$\mathbf{r} = \pi(\boldsymbol{\theta}, \mathbf{P}) - \mathbf{p}_{\text{obs}}$$

The variants differ in which parameters are involved:

| Factor | Parameters | Residual dim |
|--------|-----------|-------------|
| `ReprojPointPinhole4` | [intrinsics, pose] | 2 |
| `ReprojPointPinhole4Dist5` | [intrinsics, distortion, pose] | 2 |
| `ReprojPointPinhole4Dist5Scheimpflug2` | [intrinsics, distortion, sensor, pose] | 2 |
| `ReprojPointPinhole4Dist5TwoSE3` | [intrinsics, distortion, extrinsics, rig_pose] | 2 |
| `ReprojPointPinhole4Dist5HandEye` | [intrinsics, distortion, extrinsics, handeye, target_pose] | 2 |
| `ReprojPointPinhole4Dist5HandEyeRobotDelta` | [intrinsics, distortion, extrinsics, handeye, target_pose, robot_delta] | 2 |

### Laser Factors

| Factor | Parameters | Residual dim | Description |
|--------|-----------|-------------|-------------|
| `LaserPlanePixel` | [intrinsics, distortion, pose, plane] | 1 | Point-to-plane 3D distance |
| `LaserLineDist2D` | [intrinsics, distortion, pose, plane] | 1 | Line distance in normalized plane |

### Prior Factors

| Factor | Parameters | Residual dim | Description |
|--------|-----------|-------------|-------------|
| `Se3TangentPrior` | [se3_param] | 6 | Zero-mean Gaussian prior on SE(3) tangent |

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
  - Each: ReprojPointPinhole4Dist5 { pw, uv, w=1.0 }
  - Params: [cam_id, dist_id, pose_k_id]
  - Residual dim: 2

Total residuals: 960 scalar values
Total parameters: 4 + 5 + 10*7 = 79
```
