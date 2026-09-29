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
| `CameraParams` | `projection, distortion, sensor, intrinsics` | Serializable camera description (see [Serialization](params.md)) |

## Selectors and Modes

| Type | Variants | Used by |
|------|----------|---------|
| `DistortionKind` | `None`, `BrownConrady5`, `Rational8`, `ThinPrism9`, `Division1` | `distortion_model` in the planar and Scheimpflug intrinsics configs, and (Brown-Conrady only) the Scheimpflug rig mode |
| `SensorMode` | `Pinhole`, `Scheimpflug { init_tilt_x, init_tilt_y, fix_scheimpflug, distortion_mask_in_percam_ba, refine_scheimpflug_in_rig_ba, distortion_model }` | `sensor` in `RigExtrinsicsConfig` and `RigHandeyeConfig` (see [Scheimpflug and Laser Rigs](scheimpflug_laser_rigs.md)) |
| `RobustLoss` | `None`, `Huber { scale }`, `Cauchy { scale }`, `Arctan { scale }` | `solver.robust_loss` and the laser `calib_loss` / `laser_loss` |
| `HandEyeMode` | `EyeInHand`, `EyeToHand` | `handeye_init.handeye_mode` |

## Observation Types

| Type | Fields | Description |
|------|--------|-------------|
| `CorrespondenceView` | `points_3d, points_2d, weights` | 2D-3D correspondences (build with `CorrespondenceView::new(..)?`) |
| `View<Meta>` | `obs, meta` | Observation + metadata |
| `PlanarDataset` | `views: Vec<View<NoMeta>>` | Planar calibration input |
| `RigViewObs` | `cameras: Vec<Option<CorrespondenceView>>` | Per-camera observations of one rig frame |
| `RigView<Meta>` | `obs: RigViewObs, meta` | Multi-camera view |
| `RigDataset<Meta>` | `num_cameras, views` | Multi-camera input |
| `ReprojectionStats` | `mean, rms, max, count` | Error statistics |

## Fix Masks

| Type | Fields | Description |
|------|--------|-------------|
| `IntrinsicsFixMask` | `fx, fy, cx, cy` (bool each) | Fix individual intrinsics |
| `DistortionFixMask` | `k1, k2, k3, p1, p2` (bool each) | Fix individual distortion params |
| `CameraFixMask` | `intrinsics, distortion` | Combined camera fix mask |
| `ScheimpflugFixMask` | `tilt_x, tilt_y` (bool each) | Fix individual sensor tilt angles |

## Shared Config Groups

Every problem's `*Config` embeds the same small structs by stage:

| Type | Fields | Purpose |
|------|--------|---------|
| `IntrinsicsInitConfig` | `init_iterations, fix_k3, fix_tangential, zero_skew` | Per-camera linear initialization |
| `SolverConfig` | `max_iters, verbosity, robust_loss` | Non-linear solve |
| `RobotPoseConfig` | `refine, rot_sigma, trans_sigma` | Robot-pose refinement in hand-eye bundle adjustment |
| `HandeyeInitConfig` | `handeye_mode, min_motion_angle_deg` | Hand-eye linear initialization |
| `RigConfig` | `reference_camera_idx, refine_intrinsics_in_rig_ba` | Rig frame options |

## Session Types

| Type | Description |
|------|-------------|
| `CalibrationSession<P>` | Generic session container |
| `SessionMetadata` | Problem name, version, timestamps (read via `session.metadata()`) |
| `LogEntry` | Audit log entry (timestamp, operation, success, notes; read via `session.log()`) |
| `ExportRecord<E>` | Timestamped export |
| `InvalidationPolicy` | What to clear on input/config change |

## Optimization Backend Types

`vision-calibration-optim` keeps the IR (`ProblemIR`, `ParamBlock`, `ResidualBlock`, `FactorKind`, `ManifoldKind`, ...) and backend dispatch internal; see [Backend-Agnostic IR Architecture](ir_architecture.md). Two backend types are public:

| Type | Description |
|------|-------------|
| `BackendSolveOptions` | `max_iters`, `verbosity`, `linear_solver: Option<LinearSolverKind>`, `min_abs_decrease`, `min_rel_decrease`, `min_error` |
| `SolveReport` | `final_cost`, `num_iters` |

Inside the crate, `LinearSolverKind` (`SparseCholesky` or `SparseQR`) selects the linear solver and `BackendSolution` carries the optimized `params` and a `solve_report`.

## Hand-Eye Types

| Type | Description |
|------|-------------|
| `HandEyeMode` | EyeInHand or EyeToHand |
| `HandeyeMeta` | `base_se3_gripper: Iso3` |
| `RobotPoseMeta` | `base_se3_gripper: Iso3` |
