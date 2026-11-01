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
| `FxFyCxCy<S>` | `fx, fy, cx, cy` | Intrinsics matrix |
| `BrownConrady5<S>` | `k1, k2, k3, p1, p2, iters` | Distortion model |
| `Pinhole` | (none) | Projection model |
| `IdentitySensor` | (none) | No sensor tilt |
| `ScheimpflugParams` | `tilt_x, tilt_y` | Scheimpflug tilt |
| `Camera<S,P,D,Sm,K>` | `proj, dist, sensor, k` | Composable camera |
| `CameraParams` | `projection, distortion, sensor, intrinsics` | Serializable camera description (see [Serialization](params.md)) |

## Selectors and Modes

| Type | Variants | Used by |
|------|----------|---------|
| `DistortionKind` | `None`, `BrownConrady5`, `Rational8`, `ThinPrism9`, `Division1` | `camera.model.distortion` in every config that calibrates a camera (per camera through `camera_overrides` in a rig) |
| `SensorKind` | `Identity`, `Scheimpflug` | `camera.model.sensor` in the same configs |
| `CameraModelSpec` | `{ projection, distortion, sensor }` | `camera.model` in the planar, single-camera hand-eye, laserline-device, rig extrinsics and rig hand-eye configs |
| `CameraFix` | `{ intrinsics, projection, distortion, sensor }`, each `Option<Vec<String>>` of parameter names | `camera.fix` in the same configs, and `joint_optimize.fix` (one per camera) of the joint rig laser solve; `None` keeps the model's default |
| `CameraOverride` | `{ camera, model, fix }` | `camera_overrides` in `RigExtrinsicsConfig` and `RigHandeyeConfig`: replaces one camera's model and fix lists (see [Scheimpflug and Laser Rigs](scheimpflug_laser_rigs.md)) |
| `RobustLoss` | `None`, `Huber { scale }`, `Cauchy { scale }`, `Arctan { scale }` | `solver.robust_loss` and the laser `calib_loss` / `laser_loss` |
| `HandeyeMode` | `EyeInHand`, `EyeToHand` | `handeye_init.handeye_mode` |

## Observation Types

| Type | Fields | Description |
|------|--------|-------------|
| `CorrespondenceView` | `points_3d, points_2d`, optional `cov_px` | 2D-3D correspondences (build with `CorrespondenceView::new(..)?`, or `new_with_covariances(..)?` with a `PointCovariance` per point) |
| `View<Meta>` | `obs, meta` | Observation + metadata |
| `PlanarDataset` | `views: Vec<View<NoMeta>>` | Planar calibration input |
| `RigViewObs` | `cameras: Vec<Option<CorrespondenceView>>` | Per-camera observations of one rig frame |
| `RigView<Meta>` | `obs: RigViewObs, meta` | Multi-camera view |
| `RigDataset<Meta>` | `num_cameras, views` | Multi-camera input |
| `ReprojectionStats` | `mean, rms, max, count` | Error statistics |

## Shared Config Groups

Every problem's `*Config` embeds the same small structs by stage:

| Type | Fields | Purpose |
|------|--------|---------|
| `IntrinsicsInitConfig` | `iterations, fix_k3` | Per-camera linear initialization; fits only the distortion terms the camera's solves estimate |
| `SolverConfig` | `max_iters, verbosity, robust_loss` | Non-linear solve |
| `RobotPoseConfig` | `refine, rot_sigma, trans_sigma, estimate_sigma` | Robot-pose refinement in hand-eye bundle adjustment |
| `HandeyeInitConfig` | `handeye_mode, min_motion_angle_deg` | Hand-eye linear initialization |
| `RigConfig` | `reference_camera_idx, refine_intrinsics, refine_sensor` | Rig frame options |

## Session Types

| Type | Description |
|------|-------------|
| `CalibrationSession<P>` | Generic session container |
| `SessionMetadata` | Problem name, version, timestamps (read via `session.metadata()`) |
| `LogEntry` | Audit log entry (timestamp, operation, success, notes; read via `session.log()`) |
| `ExportRecord<E>` | Timestamped export |
| `InvalidationPolicy` | What to clear on input/config change |

## Optimization Backend Types

`vision-calibration-optim` keeps the IR (`ProblemIR`, `ParamBlock`, `ResidualBlock`, `FactorKind`, `ManifoldKind`, ...), the backend dispatch and the `optimize_*` entry points internal; see [Backend-Agnostic IR Architecture](ir_architecture.md). Each estimate carries a `SolveReport` (`final_cost`, `num_iters`); a config's `SolverConfig` (`backend`, `max_iters`, `verbosity`, `robust_loss`) sets how each stage solves.

## Hand-Eye Types

| Type | Description |
|------|-------------|
| `HandeyeMode` | EyeInHand or EyeToHand |
| `RobotPoseMeta` | `base_se3_gripper: Iso3`, a view's robot pose (every hand-eye workflow) |
