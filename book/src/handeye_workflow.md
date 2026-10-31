# Single-Camera Hand-Eye Calibration

Hand-eye calibration estimates the rigid transform between a camera and a robot's gripper (or base). This workflow combines intrinsics calibration with hand-eye estimation in a 4-step pipeline.

## Problem Formulation

### Transformation Chain (Eye-in-Hand)

For a camera mounted on the robot gripper:

$$T_{C,T} = T_{C,G} \cdot T_{G,B} \cdot T_{B,T}$$

where:
- $T_{C,G} = T_{\text{handeye}}^{-1}$: camera-to-gripper (the calibrated hand-eye transform, inverted)
- $T_{G,B} = T_{\text{robot}}^{-1}$: gripper-to-base (known from robot kinematics, inverted)
- $T_{B,T}$: base-to-target (the target's pose in the robot base frame, also calibrated)

Equivalently:

$$T_{C,T} = (T_{\text{robot}} \cdot T_{\text{handeye}})^{-1} \cdot T_{B,T}$$

### Eye-to-Hand Variant

For a fixed camera observing a target on the gripper:

$$T_{C,T} = T_{C,B} \cdot T_{B,G} \cdot T_{G,T}$$

where $T_{C,B}$ is the camera-to-base transform and $T_{G,T}$ is the target-to-gripper transform.

### Optimization Objective

Minimize reprojection error across all views, jointly over all parameters:

$$\min_{K, \mathbf{d}, X, T_{B,T}} \sum_{v=1}^{M} \sum_{j=1}^{N_v} \left\| \pi\!\left(K, \mathbf{d}, T_{C,T}^{(v)}, \mathbf{P}_j\right) - \mathbf{p}_{vj} \right\|^2$$

where $T_{C,T}^{(v)}$ depends on the known robot pose for view $v$, the hand-eye transform $X$, and the target pose $T_{B,T}$.

## 4-Step Pipeline

### Step 1: Intrinsics Initialization (`step_intrinsics_init`)

Runs the [Iterative Intrinsics](iterative_intrinsics.md) solver to estimate $K$ and distortion from the calibration board observations, ignoring robot poses entirely.

### Step 2: Intrinsics Optimization (`step_intrinsics_optimize`)

Runs planar intrinsics bundle adjustment to refine $K$, distortion, and per-view camera-to-target poses. This gives accurate per-view poses $T_{C,T}^{(v)}$ needed for hand-eye initialization.

### Step 3: Hand-Eye Initialization (`step_handeye_init`)

Uses the Tsai-Lenz linear method (see [Hand-Eye Linear](handeye_linear.md)):

1. Compute relative motions from robot poses and camera poses
2. Filter pairs by rotation magnitude (reject small rotations)
3. Estimate rotation via quaternion SVD
4. Estimate translation via linear least squares
5. Estimate $T_{B,T}$ from the hand-eye and camera poses

### Step 4: Hand-Eye Optimization (`step_handeye_optimize`)

Joint non-linear optimization of all parameters:

**Parameter blocks**:
- `"cam"` (4D, Euclidean): intrinsics
- `"dist"` (Euclidean): the distortion coefficients of the configured model
  (absent without distortion)
- `"sensor"` (2D, Euclidean): the sensor tilt, for a Scheimpflug camera
- `"handeye"` (7D, SE3): gripper-to-camera transform
- `"target"` (7D, SE3): base-to-target transform

**Residual blocks**: one `ReprojPoint { model, chain: Handeye { .. }, .. }` per observation, `model` being the camera's `CameraModelSpec`.

Without robot-pose refinement the robot poses are **not optimized**: they are
known constants from the robot kinematics (passed as per-residual data).

### Robot Pose Refinement

Robot poses carry errors of their own, usually larger than the corner noise
once projected through the lever arm to the board: 0.05° at 1 m moves the
camera's view by almost a millimetre. Taken as exact, these errors end up in
the camera parameters. `robot_poses.refine` (on by default) gives every view
an SE(3) correction of its measured pose:

**Additional parameter blocks**: `"robot_delta/v"` (6D, Euclidean se(3)
tangent `[ω, v]`) per view, applied as `exp(δ) · base_se3_gripper`

**Additional residuals**: a zero-mean `Se3TangentPrior` per view, with the
robot's rotation and translation standard deviations `σ_r` (`rot_sigma`) and
`σ_t` (`trans_sigma`), weighed against the corner residuals at their pixel
noise `σ_px`:

$$\mathbf{r}_{\text{prior}} = \sigma_{px} \begin{bmatrix} \boldsymbol{\omega} / \sigma_r \\ \mathbf{v} / \sigma_t \end{bmatrix}$$

The objective is then the maximum-likelihood one, in units of `σ_px²`: a
correction costs as much as the corner evidence it explains away. `σ_px` is
not known in advance, so the solve runs once at 1 px, takes `σ_px` from the
corner residuals and runs again from that solution. Every view's correction
is free; the priors are what tell the corrections apart from a change of the
hand-eye or target pose that would move every view alike.

The priors' standard deviations decide how far the result trusts the robot,
and the covariance takes them as its true accuracy. Too loose, and the
hand-eye and target standard deviations come out larger than the data
warrants; too tight, and the robot's errors leak into the camera parameters
while the standard deviations come out too small. The defaults (0.5°, 1 mm)
are loose for an industrial arm; a result's `robot_deltas` show the size of
the corrections the data asked for.

The target pose in the robot base frame is known only as well as the
robot's absolute rotation: a common rotation of every robot pose about the
base origin looks exactly like a rotation of the target. Its translation's
standard deviation is about `σ_r / √views` times the target's distance from
the base.

## Configuration

Settings are grouped by stage; the shared sub-structs (`IntrinsicsInitConfig`,
`CameraConfig`, `HandeyeInitConfig`, `SolverConfig`, `RobotPoseConfig`) are
reused across every hand-eye problem type:

```rust
pub struct SingleCamHandeyeConfig {
    // Per-camera linear-initialization stage settings.
    pub intrinsics: IntrinsicsInitConfig,
    // { iterations: usize = 2, fix_k3: bool = true }; it fits only the
    //   distortion terms the camera's solves estimate

    // The camera model (any distortion kind, an identity or Scheimpflug
    // sensor) and the parameters its solves hold fixed, named per slot.
    pub camera: CameraConfig, // { model: CameraModelSpec, fix: CameraFix }

    // Hand-eye linear-initialization stage settings (Tsai-Lenz DLT).
    pub handeye_init: HandeyeInitConfig,
    // { handeye_mode: HandeyeMode, min_motion_angle_deg: f64 = 5.0 }

    // Non-linear solve stage settings.
    pub solver: SolverConfig, // { max_iters, verbosity, robust_loss }

    // Robot-pose refinement settings for the hand-eye bundle adjustment.
    pub robot_poses: RobotPoseConfig,
    // { refine: bool = true, rot_sigma: f64, trans_sigma: f64 }
}
```

## Input Format

Each view is a `View<HandeyeMeta>` (aliased as `SingleCamHandeyeView`):

```rust
pub struct HandeyeMeta {
    pub base_se3_gripper: Iso3,  // Robot pose (from kinematics)
}

// Type alias
pub type SingleCamHandeyeView = View<HandeyeMeta>;
// Where View<Meta> has fields: obs: CorrespondenceView, meta: Meta
```

## Mounting Modes

Poses follow the `frame_se3_frame` naming: `base_se3_gripper` reads "gripper pose expressed in the base frame" ($T_{B,G}$). The unknowns are named the same way. The mounting mode is `HandeyeInitConfig::handeye_mode`:

- `HandeyeMode::EyeInHand` (the default): the camera rides on the gripper and observes a **fixed** target. The calibration solves for `gripper_se3_camera` ($T_{G,C}$) and `base_se3_target` ($T_{B,T}$).
- `HandeyeMode::EyeToHand`: the camera is fixed in the scene and observes a target **mounted on the gripper**. The calibration solves for `camera_se3_base` ($T_{C,B}$) and `gripper_se3_target` ($T_{G,T}$).

The export's four pose fields are mutually exclusive by mode: the active mode fills its pair and leaves the other pair `None`.

To switch to eye-to-hand:

```rust
use vision_calibration::optim::HandeyeMode;
use vision_calibration::single_cam_handeye::SingleCamHandeyeConfig;

let mut config = SingleCamHandeyeConfig::default();
config.handeye_init.handeye_mode = HandeyeMode::EyeToHand;
session.set_config(config)?;
```

In eye-to-hand mode the pixel-to-3D chain for a fixed camera observing a gripper-mounted target composes as $T_{C,T} = T_{C,B} \cdot T_{B,G} \cdot T_{G,T}$, the same identity `step_handeye_init_with_seed` uses to derive `camera_se3_base` when only `gripper_se3_target` is seeded (see [Seeding with Prior Knowledge](manual_init.md#hand-eye-coupling)).

## Complete Example

```rust
use vision_calibration::prelude::*;
use vision_calibration::single_cam_handeye::*;

let views: Vec<SingleCamHandeyeView> = /* load data */;
let input = SingleCamHandeyeInput::new(views)?;

let mut session = CalibrationSession::<SingleCamHandeyeProblem>::new();
session.set_input(input)?;

step_intrinsics_init(&mut session, None)?;
step_intrinsics_optimize(&mut session, None)?;
step_handeye_init(&mut session, None)?;
step_handeye_optimize(&mut session, None)?;

let export = session.export()?;
println!("Camera: {:?}", export.camera); // CameraParams, in the configured model
println!("Hand-eye: {:?}", export.gripper_se3_camera);
println!("Target in base: {:?}", export.base_se3_target);
println!("Reprojection error: {:.4} px", export.mean_reproj_error);
```

### Reading the Export

```rust
use vision_calibration::optim::HandeyeMode;

match export.handeye_mode {
    HandeyeMode::EyeInHand => {
        let gripper_se3_camera = export.gripper_se3_camera.unwrap(); // T_G_C
        let base_se3_target = export.base_se3_target.unwrap();       // T_B_T
    }
    HandeyeMode::EyeToHand => {
        let camera_se3_base = export.camera_se3_base.unwrap();       // T_C_B
        let gripper_se3_target = export.gripper_se3_target.unwrap(); // T_G_T
    }
}
```

`export.camera` is the calibrated camera (`CameraParams`) in the configured model, `export.robot_deltas` the per-view robot-pose corrections when refinement is on, and `export.per_feature_residuals` the per-corner reprojection records (see [Per-Feature Residuals](per_feature_residuals.md)).

The `SingleCamHandeyeInput::new` constructor rejects an empty set of views and any view with fewer than 4 correspondences (the minimum for a homography). Each stage has a `*_with_seed` sibling and a typed step result (`IntrinsicsInitResult`, `HandeyeOptimizeResult`, ...); see [Seeding with Prior Knowledge](manual_init.md).

### From a Dataset Manifest

Instead of assembling the views by hand, a manifest with `topology = "single_cam_handeye"` describes where the images, the target and the robot-pose file live. `vision_calibration::dataset_runner::build_single_cam_handeye_input` runs the (cached) corner detection and returns the `SingleCamHandeyeInput`; see [The Dataset Manifest](manifest.md).

## Camera Models

The camera model is the config's `camera.model`: any distortion kind
(Brown-Conrady by default, rational, thin-prism, division, or none) and an
identity or Scheimpflug sensor. The intrinsics stage dispatches on it like
[Planar Intrinsics](planar_intrinsics.md): Zhang's method embedded into the
model's distortion, and for a tilted sensor the linear tilt sweep followed by
the staged tilt multi-start. A known tilt goes in the intrinsics manual-init
`sensor` seed (`step_intrinsics_init_with_seed`); a tilt seed for an untilted
camera must be zero. `camera.fix` names the parameters both solves hold.

## Practical Considerations

- **Rotation diversity** is critical for Tsai-Lenz initialization. Include rotations around at least 2 axes.
- **5-10 views minimum**, with 10-20 recommended for robustness.
- **Robot accuracy**: If the robot's reported poses have errors > 1mm or > 0.1°, enable robot pose refinement.
- **Initialization failure**: If Tsai-Lenz produces a poor estimate (large reprojection error after optimization), the most likely cause is insufficient rotation diversity.

> **OpenCV equivalence**: `cv::calibrateHandEye` provides the linear initialization step. calibration-rs adds joint non-linear refinement and optional robot pose correction.
