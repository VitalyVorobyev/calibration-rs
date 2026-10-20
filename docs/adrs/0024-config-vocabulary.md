# ADR 0024: One Config Vocabulary Across the Eight Problem Types

- Status: Accepted
- Date: 2026-07-08

## Context

The eight problem configs describe the same concepts (init iterations, distortion and
intrinsics fixing, pose-gauge fixing, solver budget, robot-pose priors) and must spell
them identically, or the JSON, the schema-driven forms (ADR 0018), the Python mirror and
the docs each need per-problem special cases. Configs are grouped by stage, with shared
building blocks defined once. Pre-1.0, a hard rename without compatibility aliases is the
accepted way to keep the vocabulary single-valued (the precedent of ADR 0010).

## Decision

### grouped shape everywhere, shared sub-structs

All eight top-level configs are grouped. The shared building blocks are
defined once in `vision_calibration_pipeline::common::config` (re-exported
through the facade's `common` module):

```rust
/// Per-camera linear-initialization stage.
IntrinsicsInitConfig {
    init_iterations: usize = 2,
    fix_k3: bool = true,
    fix_tangential: bool = false,   // scheimpflug default: true
    zero_skew: bool = true,
}

/// Non-linear solve stage.
SolverConfig {
    max_iters: usize,               // per-problem default, documented below
    verbosity: usize = 0,
    robust_loss: RobustLoss = None,
    backend: SolverBackend = TinySolver,   // ADR 0025
}

/// Robot-pose refinement prior.
RobotPoseConfig {
    refine: bool = true,
    rot_sigma: f64 = 0.5_f64.to_radians(),
    trans_sigma: f64 = 1.0e-3,
}
```

Naming rules:

- The init-iteration count is `init_iterations` everywhere (self-documenting
  in JSON out of context). No `_in_init` suffixes — the group carries scope.
- Per-camera fixing uses core's `CameraFixMask { intrinsics, distortion }`
  as the *only* mask shape. Configs never use plain booleans for
  camera parameter fixing: `LaserlineDeviceOptimizeConfig` has
  `fix_camera: CameraFixMask` (default: intrinsics all-free, distortion
  `{k3: fixed}`), and the joint rig BA uses `default_camera_fix: CameraFixMask`.
- The Scheimpflug tilt mask is the single `ScheimpflugFixMask`,
  and its config field is `fix_scheimpflug` in all three exposure points
  (single-cam config, `SensorMode::Scheimpflug`, joint BA; the joint BA default is both-fixed).
- Free per-view pose fixing is `fix_poses: Vec<usize>` (default `vec![0]` where a
  gauge is needed). `fix_target_ref` names one specific pose, not an index list.
- `max_iters`/`verbosity` are plain `usize` with concrete defaults, never `Option`.
- Documented `max_iters` defaults: 50 standard; 120 Scheimpflug single-cam
  (tilt valley needs headroom); 30 joint rig+hand-eye+laser BA (warm-started
  from converged stages); 200 rig-laserline frozen-geometry stage (cheap
  1-DOF-per-view problem, iterations are nearly free).

### Pose gauge

The reference camera (`reference_camera_idx`) extrinsics fix is the only gauge removal for a
rig: it removes the full 6-DOF rig gauge. View 0's rig pose is free; an additional first-pose
fix is redundant and measurably mildly pessimizing (`docs/notes/rig-extrinsics.md`, Gauge).
The joint BA always pins `reference_camera_idx`, not camera 0.

### robust-loss vocabulary

Non-laser problems: the single `robust_loss` inside `SolverConfig`.
Laser-carrying stages keep the two-family split (`calib_loss`, `laser_loss`,
`calib_weight`, `laser_weight`) because target corners and laser stripes
are genuinely different residual families. The per-stage defaults are
tuned values and stay per-stage:

- `LaserlineDeviceConfig.optimize`: `calib_loss = Huber{1.0}`,
  `laser_loss = Huber{1.0}` (both residuals in pixels by default; a loss
  scale is in its residual's unit), weights 1.0/1.0 — a from-scratch solve
  that must robustify detector tails.
- Joint rig BA: `calib_loss = None`, `laser_loss = None`,
  `laser_weight = 1e4` — a warm-started polish on already-cleaned inputs
  where the laser term pins metric scale (see `docs/notes/laserline-bundle.md`, metric scale).

### `distortion_model` placement

A live multi-valued choice in the two single-camera intrinsics
configs; schema-symmetric but validated-to-BC5 inside
`SensorMode::Scheimpflug` for rigs; absent (implicitly BC5) elsewhere.

### target top-level shapes

| Config | Groups |
|---|---|
| `PlanarIntrinsicsConfig` | `init`, `solver`, + `distortion_model`, `fix_camera`, `fix_poses` |
| `ScheimpflugIntrinsicsConfig` | `init` (fix_tangential = true), `solver` (120), + `distortion_model`, `fix_camera` (radial-only default), `fix_scheimpflug`, `fix_poses = [0]` |
| `SingleCamHandeyeConfig` | `intrinsics`, `handeye_init` (`handeye_mode`, `min_motion_angle_deg`), `solver`, `robot_poses` |
| `LaserlineDeviceConfig` | `init` (+ `sensor_init`), `solver`, `optimize` (`calib_loss`, `laser_loss`, weights, `fix_camera`, `fix_sensor`, `fix_poses = [0]`, `fix_plane`, `laser_residual_type`) |
| `RigExtrinsicsConfig` | `intrinsics`, `sensor`, `rig` (`reference_camera_idx`, `refine_intrinsics_in_rig_ba`), `solver` |
| `RigHandeyeConfig` | `intrinsics` (+ `manual_init`), `sensor`, `rig`, `handeye_init`, `solver`, `handeye_ba` (`robot_poses`, `refine_cam_se3_rig`, `refine_scheimpflug`) |
| `RigLaserlineDeviceConfig` | `solver` (200), `laser_residual_type` |
| `RigHandeyeLaserlineConfig` | `handeye`, `laserline_init`, `joint_ba` (`solver` (30), `laser_residual_type`, losses/weights, `default_camera_fix: CameraFixMask`, `fix_scheimpflug`, `fix_handeye`, `fix_target_ref`, `robot_poses`) |

## Consequences

- The wire shape of all eight configs is the grouped shape above; consumers (pipeline,
  bench registry overrides, the app's preset override trees, the Python mirror,
  tutorials and the book) follow it.
- The schema-driven UI (ADR 0018) regenerates from `default_config_cmd` and picks up the
  grouped shapes automatically; only hand-written preset overrides need edits.
- Adding a config field means choosing its group and reusing the shared sub-structs, not
  inventing a new spelling.
