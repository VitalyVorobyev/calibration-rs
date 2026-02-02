# Data Types Quick Reference

## Core Math Types

All based on `nalgebra` with `f64` precision:

| Type | Definition | Description |
|------|-----------|-------------|
| `Real` | `f64` | Scalar type |
| `Pt2` | `Point2<f64>` | 2D point |
| `Pt3` | `Point3<f64>` | 3D point |
| `Vec2` | `Vector2<f64>` | 2D vector |
| `Vec3` | `Vector3<f64>` | 3D vector |
| `Mat3` | `Matrix3<f64>` | 3×3 matrix |
| `Mat4` | `Matrix4<f64>` | 4×4 matrix |
| `Iso3` | `Isometry3<f64>` | SE(3) rigid transform |

## Camera Model Types

| Type | Parameters | Description |
|------|-----------|-------------|
| `FxFyCxCySkew<S>` | `fx, fy, cx, cy, skew` | Intrinsics matrix |
| `BrownConrady5<S>` | `k1, k2, k3, p1, p2, iters` | Distortion model |
| `Pinhole` | (none) | Projection model |
| `IdentitySensor` | (none) | No sensor tilt |
| `ScheimpflugParams` | `tilt_x, tilt_y` | Scheimpflug tilt |
| `Camera<S,P,D,Sm,K>` | `proj, dist, sensor, k` | Composable camera |

## Observation Types

| Type | Fields | Description |
|------|--------|-------------|
| `CorrespondenceView` | `points_3d, points_2d, weights` | 2D-3D correspondences |
| `View<Meta>` | `obs, meta` | Observation + metadata |
| `PlanarDataset` | `views: Vec<View<NoMeta>>` | Planar calibration input |
| `RigView<Meta>` | `obs: RigViewObs, meta` | Multi-camera view |
| `RigDataset<Meta>` | `num_cameras, views` | Multi-camera input |
| `ReprojectionStats` | `mean, rms, max, count` | Error statistics |

## Fix Masks

| Type | Fields | Description |
|------|--------|-------------|
| `IntrinsicsFixMask` | `fx, fy, cx, cy` (bool each) | Fix individual intrinsics |
| `DistortionFixMask` | `k1, k2, k3, p1, p2` (bool each) | Fix individual distortion params |
| `CameraFixMask` | `intrinsics, distortion` | Combined camera fix mask |
| `FixedMask` | opaque (`all_free()`, `all_fixed(dim)`, `fix_indices(&[usize])`) | IR-level parameter fixing |

## Optimization IR Types

| Type | Description |
|------|-------------|
| `ProblemIR` | Complete optimization problem |
| `ParamBlock` | Parameter group (id, name, dim, manifold, fixed) |
| `ResidualBlock` | Residual definition (params, loss, factor) |
| `ParamId` | Unique parameter identifier |
| `FactorKind` | Residual computation type (enum) |
| `ManifoldKind` | Parameter geometry (Euclidean, SE3, SO3, S2) |
| `RobustLoss` | Loss function (None, Huber, Cauchy, Arctan) |

## Session Types

| Type | Description |
|------|-------------|
| `CalibrationSession<P>` | Generic session container |
| `SessionMetadata` | Problem name, version, timestamps |
| `LogEntry` | Audit log entry (timestamp, operation, success) |
| `ExportRecord<E>` | Timestamped export |
| `InvalidationPolicy` | What to clear on input/config change |

## Backend Types

| Type | Description |
|------|-------------|
| `BackendSolveOptions` | Solver settings (max_iters, tolerances, etc.) |
| `BackendSolution` | Optimized params + solve report |
| `SolveReport` | Initial/final cost, iterations, termination |
| `LinearSolverType` | SparseCholesky or SparseQR |

## Hand-Eye Types

| Type | Description |
|------|-------------|
| `HandEyeMode` | EyeInHand or EyeToHand |
| `HandeyeMeta` | `base_se3_gripper: Iso3` |
| `RobotPoseMeta` | `base_se3_gripper: Iso3` |
