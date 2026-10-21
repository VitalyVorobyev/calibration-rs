# Planar Intrinsics Calibration

This is the most common calibration workflow: estimate camera intrinsics and lens distortion from multiple views of a planar calibration board. It combines Zhang's linear initialization with Levenberg-Marquardt bundle adjustment.

## Problem Formulation

**Parameters**:
- Intrinsics: $K = (f_x, f_y, c_x, c_y)$ — 4 scalars
- Distortion: $\mathbf{d} = (k_1, k_2, k_3, p_1, p_2)$ — 5 scalars
- Per-view poses: $\{T_v\}_{v=1}^M$ — $M$ SE(3) transforms (6 DOF each)

**Total**: $9 + 6M$ parameters.

**Observations**: For each view $v$ and board point $j$:
- Known 3D position $\mathbf{P}_j$ (on the board, at $Z = 0$)
- Observed pixel $\mathbf{p}_{vj}$

**Objective**: Minimize total reprojection error:

$$\min_{K, \mathbf{d}, \{T_v\}} \sum_{v=1}^{M} \sum_{j=1}^{N_v} \left\| \pi(K, \mathbf{d}, T_v, \mathbf{P}_j) - \mathbf{p}_{vj} \right\|^2$$

where $\pi$ is the full camera projection pipeline: SE(3) transform → pinhole → distortion → intrinsics.

With robust loss $\rho$:

$$\min_{K, \mathbf{d}, \{T_v\}} \sum_{v=1}^{M} \sum_{j=1}^{N_v} \rho\left( \left\| \pi(K, \mathbf{d}, T_v, \mathbf{P}_j) - \mathbf{p}_{vj} \right\| \right)$$

## Two-Step Pipeline

### Step 1: Linear Initialization (`step_init`)

1. **Homographies**: Compute $H_v$ for each view via DLT (from board points at $Z = 0$ to observed pixels)
2. **Intrinsics**: Estimate $K$ from homographies using Zhang's method, iteratively refined with distortion estimation (see [Iterative Intrinsics](iterative_intrinsics.md))
3. **Distortion**: Estimate $(k_1, k_2, p_1, p_2)$ from homography residuals (see [Distortion Fit](distortion_fit.md))
4. **Poses**: Decompose each homography to recover $T_v$ (see [Pose from Homography](planar_pose.md))

After initialization, intrinsics are typically within 10-40% of the true values.

### Step 2: Non-Linear Refinement (`step_optimize`)

Constructs the optimization problem as IR:

- **Parameter blocks**: `"cam"` (4D, Euclidean), `"dist"` (the distortion model's coefficients, Euclidean; absent for no distortion), `"sensor"` (`tilt_x, tilt_y`, Scheimpflug sensor only), `"pose/0"`...`"pose/M-1"` (7D, SE3)
- **Residual blocks**: One `ReprojPoint { model: PINHOLE4_DIST5, chain: SinglePose, .. }` per observation (2D residual)
- **Solver**: the Levenberg–Marquardt loop on the backend `solver.backend` selects (tiny-solver by default; see [Solver Backends](solver_backends.md))

After optimization, expect <2% intrinsics error and <1 px mean reprojection error.

## Configuration

`init` and `solver` are shared sub-structs reused by every intrinsics-bearing
problem type; `camera` names the camera model and its fixed parameters:

```rust
pub struct PlanarIntrinsicsConfig {
    // Linear-initialization stage settings.
    pub init: IntrinsicsInitConfig,
    // { init_iterations: usize = 2, fix_k3: bool = true,
    //   fix_tangential: Option<bool> = None }  (None: true for a Scheimpflug sensor)

    // Non-linear solve stage settings.
    pub solver: SolverConfig,
    // { max_iters: usize = 120, verbosity: usize = 0, robust_loss: RobustLoss = None }

    // The camera model and the parameters held fixed.
    pub camera: CameraConfig,
    // { model: CameraModelSpec { projection, distortion, sensor },
    //   fix: CameraFix { intrinsics, projection, distortion, sensor } }

    pub fix_poses: Vec<usize>,              // Fix specific view poses
}
```

`solver.max_iters` is sized for a tilted sensor's tilt valley; the solver stops
at convergence, so the cap only bounds runs that do not converge.

### Camera Model

`camera.model` names one kind per slot. The default is a pinhole projection,
Brown-Conrady distortion and an identity (untilted) sensor.

`camera.model.distortion` selects the lens model that is fitted:

| Variant | Parameters | Model |
|---------|-----------|-------|
| `DistortionKind::None` | (none) | No distortion |
| `DistortionKind::BrownConrady5` | `k1, k2, k3, p1, p2` | Brown-Conrady radial + tangential (default) |
| `DistortionKind::Rational8` | `k1..k6, p1, p2` | OpenCV rational polynomial |
| `DistortionKind::ThinPrism9` | `k1, k2, k3, p1, p2, s1..s4` | Brown-Conrady + thin prism |
| `DistortionKind::Division1` | `lambda` | Fitzgibbon division model |

```rust
use vision_calibration::core::DistortionKind;

session.update_config(|c| c.camera.model.distortion = DistortionKind::Rational8)?;
```

For the non-default models, `export.params.distortion()` returns `None`; read the coefficients from `export.params.camera.distortion` (`DistortionParams`, see [Serialization and Runtime-Dynamic Types](params.md)).

`camera.model.sensor = SensorKind::Scheimpflug` calibrates a tilted sensor
as well; see [Tilted (Scheimpflug) Sensor](#tilted-scheimpflug-sensor) below.

### Fixed Parameters

`camera.fix` lists, per slot, the parameter names held fixed during the
non-linear solve. A slot left `None` keeps the model's default: `k3` for
Brown-Conrady, and everything but the low-order radial terms (`k1`, `k2`; the
division model's `lambda`) under a Scheimpflug sensor. An explicit list
replaces the default, and an unknown name is a configuration error raised
before any work.

```rust
// Fix cx, cy but optimize fx, fy
session.update_config(|c| {
    c.camera.fix.intrinsics = Some(vec!["cx".into(), "cy".into()]);
})?;

// Fix k3 and tangential distortion
session.update_config(|c| {
    c.camera.fix.distortion = Some(vec!["k3".into(), "p1".into(), "p2".into()]);
})?;
```

In JSON the same config reads
`"camera": {"model": {...}, "fix": {"intrinsics": ["cx", "cy"], "distortion": ["k3", "p1", "p2"]}}`.

## Tilted (Scheimpflug) Sensor

With `camera.model.sensor = SensorKind::Scheimpflug` the same workflow also
estimates the sensor tilt `tilt_x, tilt_y` (radians). The tilt trades off
against the principal point and radial distortion, so the sensor changes three
defaults and both steps:

- The default fix list leaves only the low-order radial terms free.
- `init.fix_tangential` defaults to `true`: a free tangential term is
  ill-posed next to the tilt in the linear stage.
- A Scheimpflug sensor needs a distortion model (`DistortionKind::None` is a
  configuration error).

`step_init_with_seed` takes an optional `sensor` seed in `PlanarManualInit`
(a typed error for an untilted camera). With a **seeded tilt** — the nominal
mount angle, plus a coarse focal — `step_optimize` trusts that basin: it sweeps
the leading radial coefficient over `{0, -0.20, -0.40}` with intrinsics and
tilt held, then runs a joint solve bounded around the seed (focals
`[0.75, 1.5]×`, principal point `±0.15·f`, tilt `±0.10` rad). Without a tilt
seed (**cold start**, experimental) it runs a staged multi-start: a cheap
tilt sweep with the principal point frozen, a refine of the winning basin,
and a bounded joint refine. The init log warns when the intrinsics are not
seeded, because a from-scratch start can settle in a wrong tilt/focal basin.

```rust
use vision_calibration::core::{FxFyCxCy, ScheimpflugParams, SensorKind};
use vision_calibration::planar_intrinsics::{
    PlanarIntrinsicsConfig, PlanarIntrinsicsProblem, PlanarManualInit, step_init_with_seed,
    step_optimize,
};

let mut config = PlanarIntrinsicsConfig::default();
config.camera.model.sensor = SensorKind::Scheimpflug;
let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
session.set_config(config)?;
session.set_input(dataset)?;

let mut seed = PlanarManualInit::default();
seed.intrinsics = Some(FxFyCxCy { fx: 1050.0, fy: 1050.0, cx: 360.0, cy: 270.0 });
seed.sensor = Some(ScheimpflugParams { tilt_x: -0.087, tilt_y: 0.0 }); // nominal mount tilt
step_init_with_seed(&mut session, seed, None)?;
step_optimize(&mut session, None)?;
```

The export is the same `PlanarIntrinsicsExport`; `params.camera.sensor` holds
the calibrated tilt and the covariance reports `tilt_x ± σ`. A device spec
derives the seed with `device_seed::scheimpflug_seed`.

## Complete Example

```rust
use vision_calibration::prelude::*;
use vision_calibration::planar_intrinsics::{step_init, step_optimize};

let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
session.set_input(dataset)?;

// Optional: customize configuration
session.update_config(|c| {
    c.solver.max_iters = 80;
    c.solver.robust_loss = RobustLoss::Huber { scale: 2.0 };
})?;

// Run pipeline
// Inspect initialization
let init = step_init(&mut session, None)?;
println!("Init fx={:.1}, fy={:.1}", init.intrinsics.fx, init.intrinsics.fy);

step_optimize(&mut session, None)?;

// Export results
let export = session.export()?;
let k = export.params.intrinsics();
println!("Final fx={:.1}, fy={:.1}", k.fx, k.fy);
println!("Reprojection error: {:.4} px", export.mean_reproj_error);
```

## Filtering (Optional Step)

After optimization, views or individual observations with high reprojection error can be filtered out:

```rust
use vision_calibration::planar_intrinsics::{step_filter, FilterOptions};

let mut filter_opts = FilterOptions::default();
filter_opts.max_reproj_error = 2.0;      // Remove observations > 2 px
filter_opts.min_points_per_view = 10;    // Minimum points to keep a view
filter_opts.remove_sparse_views = true;  // Drop views below threshold
step_filter(&mut session, filter_opts)?;

// Filtering edits the input, which clears the computed state: run both steps again
step_init(&mut session, None)?;
step_optimize(&mut session, None)?;

// Or, in one call: run_calibration_with_filtering(&mut session, filter_opts)?
```

> **OpenCV equivalence**: `cv::calibrateCamera` performs both initialization and optimization internally. calibration-rs separates these steps for inspection and customization.

## Accuracy Expectations

| Stage | Intrinsics error | Reprojection error |
|-------|-----------------|-------------------|
| After `step_init` | 10-40% | Not computed |
| After `step_optimize` | <2% | <1 px mean |
| After filtering + re-optimize | <1% | <0.5 px mean |

## Input Requirements

- **Minimum 3 views** (for Zhang's closed form)
- **Minimum 4 points per view** (for homography estimation)
- **View diversity**: Views should include rotation around both axes and vary in distance
- **Board coverage**: Points should span the full image area for good distortion estimation
