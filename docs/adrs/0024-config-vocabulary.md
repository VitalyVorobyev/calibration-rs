# ADR 0024: One Config Vocabulary Across the Eight Problem Types

- Status: Accepted
- Date: 2026-07-08

## Context

The eight problem configs grew independently and now spell the same concept
up to four different ways. A field-inventory survey (2026-07-08, R2) found:

1. The linear-init iteration count is named `init_iterations`
   (planar, scheimpflug, rig-handeye), `intrinsics_init_iterations`
   (single-cam hand-eye, rig extrinsics), and `iterations`
   (laserline init group) — all defaulting to `2`.
2. `fix_k3` / `fix_tangential` appear with and without an `_in_init`
   suffix depending on the config, and `LaserlineDeviceConfig` reuses the
   bare name `fix_k3` for two *different* pipeline stages in one tree.
   `ScheimpflugIntrinsicsConfig` has no tangential knob at all — it was
   hard-coded `true` in `steps.rs`.
3. `fix_intrinsics` / `fix_distortion` are per-parameter mask structs in
   three configs but plain `bool`s in `LaserlineDeviceOptimizeConfig`,
   under identical field names.
4. The Scheimpflug tilt mask has three config spellings
   (`fix_scheimpflug`, `fix_scheimpflug_in_intrinsics`,
   `fix_scheimpflug_tilt`) over two distinct Rust types both named
   `ScheimpflugFixMask` (merged by R1).
5. Pose-gauge fixing is spelled four ways: `fix_poses: Vec<usize>`,
   `fix_first_pose: bool`, `fix_first_rig_pose: bool`,
   `fix_target_ref: bool` — plus `fix_first_camera_extrinsic: bool`,
   which hard-codes camera index 0 instead of honoring
   `reference_camera_idx` (a bug).
6. `fix_first_rig_pose` is redundant with the reference-camera gauge fix
   and empirically mildly pessimizing (Q8 rig-extrinsics proof pack:
   0.134 px vs 0.124 px on the wide matrix cell with it on vs off; see
   `docs/notes/rig-extrinsics.md` §Gauge).
7. `max_iters`/`verbosity` are `usize` everywhere except
   `RigLaserlineDeviceConfig` (`Option<usize>` whose `None` default no
   real caller ever uses — bench, app, and examples all override it).
8. The robot-pose prior (`robot_rot_sigma`, `robot_trans_sigma`) is
   duplicated verbatim in three configs.
9. Grouping is inconsistent: five configs are flat while `RigHandeyeConfig`
   models the same rig concepts in five named sub-structs.

Pre-1.0, breaking changes are acceptable; ADR 0010 set the precedent of
hard renames with no compatibility aliases.

## Decision

### D1 — grouped shape everywhere, shared sub-structs

All eight top-level configs become grouped. The shared building blocks are
defined once in `vision_calibration_pipeline::common::config` (re-exported
through the facade's `common` module):

```rust
/// Per-camera linear-initialization stage.
IntrinsicsInitConfig {
    init_iterations: usize = 2,
    fix_k3: bool = true,
    fix_tangential: bool = false,   // scheimpflug default: true (was hard-coded)
    zero_skew: bool = true,
}

/// Non-linear solve stage.
SolverConfig {
    max_iters: usize,               // per-problem default, documented below
    verbosity: usize = 0,
    robust_loss: RobustLoss = None,
}

/// Robot-pose refinement (replaces three verbatim-duplicated blocks).
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
  as the *only* mask shape. `LaserlineDeviceOptimizeConfig`'s boolean trio
  (`fix_intrinsics`/`fix_distortion`/`fix_k3`) collapses into one
  `fix_camera: CameraFixMask` (default preserves old behavior: intrinsics
  all-free, distortion `{k3: fixed}`). The structurally-identical
  `JointCameraFixMask` (rig-handeye-laserline) is deleted in favor of
  `CameraFixMask`.
- The Scheimpflug tilt mask is the single (post-R1) `ScheimpflugFixMask`,
  and its config field is `fix_scheimpflug` in all three exposure points
  (single-cam config, `SensorMode::Scheimpflug`, joint BA — the joint BA's
  `fix_scheimpflug_tilt: bool` widens to the mask with default both-fixed).
- Free per-view pose fixing is `fix_poses: Vec<usize>` everywhere;
  `fix_first_pose: bool = true` becomes `fix_poses = vec![0]`.
  `fix_target_ref` survives (it names one specific pose, not an index list).
- `max_iters`/`verbosity` are plain `usize` with concrete defaults;
  `RigLaserlineDeviceConfig`'s `Option<usize>` dies
  (`max_iters = 200`, `verbosity = 0` — the values every caller used).
- Documented `max_iters` defaults: 50 standard; 120 Scheimpflug single-cam
  (tilt valley needs headroom); 30 joint rig+hand-eye+laser BA (warm-started
  from converged stages); 200 rig-laserline frozen-geometry stage (cheap
  1-DOF-per-view problem, iterations are nearly free).

### D2 — remove the redundant gauge knob (the only behavior change)

- `fix_first_rig_pose` is deleted from `RigExtrinsicsConfig` and
  `RigHandeyeRigConfig`. The reference-camera fix alone removes the full
  6-DOF rig gauge; the extra constraint is evidence-backed redundant and
  mildly pessimizing (Context §6).
- `fix_first_camera_extrinsic: bool` is deleted from the joint BA config;
  the implementation now always pins `reference_camera_idx` (fixing the
  index-0 hard-coding bug).
- `RigLaserlineDeviceConfig`'s effective `max_iters`-when-unset changes from
  100 to 200. Pre-R3, the step function's own `Option<usize>` override
  parameter defaulted to `cfg.max_iters.unwrap_or(100)`; post-R3 the config
  itself is grouped into `solver: SolverConfig` with `max_iters: usize = 200`
  and there is no lower `unwrap_or` layer left to disagree with it. Every
  real caller (bench, app, examples) already passed `200` explicitly, so
  this only changes the behavior of a config that never sets `max_iters` and
  never overrides it at the step-function call site — a case none of our
  callers exercise. The frozen-geometry rig-laserline stage is a cheap
  1-DOF-per-view problem, so a higher iteration ceiling costs effectively
  nothing when it does trigger.

These are the only intentional numeric-behavior changes in the R3 rollout.
Expected effect is unchanged-or-slightly-improved fits; the acceptance
drift gates arbitrate, and any refreeze rides the R3 PR with before/after
numbers.

### D3 — robust-loss vocabulary

Non-laser problems: the single `robust_loss` inside `SolverConfig`.
Laser-carrying stages keep the two-family split (`calib_loss`, `laser_loss`,
`calib_weight`, `laser_weight`) because target corners and laser stripes
are genuinely different residual families. The per-stage defaults are
**preserved as-is** — they are tuned values, not naming drift:

- `LaserlineDeviceConfig.optimize`: `calib_loss = Huber{1.0}`,
  `laser_loss = Huber{0.01}`, weights 1.0/1.0 — a from-scratch solve that
  must robustify detector tails.
- Joint rig BA: `calib_loss = None`, `laser_loss = None`,
  `laser_weight = 1e4` — a warm-started polish on already-cleaned inputs
  where the laser term pins metric scale (see `docs/notes/rtv3d-scale.md`).

### D4 — `distortion_model` placement

Unchanged: a live multi-valued choice in the two single-camera intrinsics
configs; schema-symmetric but validated-to-BC5 inside
`SensorMode::Scheimpflug` for rigs; absent (implicitly BC5) elsewhere.

### D5 — target top-level shapes

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

### Old → new field mapping (rename gate: grep must find zero old names)

| Old | New |
|---|---|
| `intrinsics_init_iterations`, `iterations` (laserline init) | `init_iterations` |
| `fix_k3_in_init`, `fix_tangential_in_init` | `init.fix_k3`, `init.fix_tangential` |
| `LaserlineDeviceOptimizeConfig.{fix_intrinsics: bool, fix_distortion: bool, fix_k3: bool}` | `optimize.fix_camera: CameraFixMask` |
| `fix_scheimpflug_in_intrinsics`, `fix_scheimpflug_tilt` | `fix_scheimpflug` |
| `fix_first_pose: bool` | `fix_poses: Vec<usize>` (`[0]`) |
| `fix_first_rig_pose` | *(deleted — D2)* |
| `fix_first_camera_extrinsic` | *(deleted — D2; reference camera always pinned)* |
| `robot_rot_sigma` + `robot_trans_sigma` + `refine_robot_poses` | `robot_poses: RobotPoseConfig { refine, rot_sigma, trans_sigma }` |
| `max_iters: Option<usize>`, `verbosity: Option<usize>` (rig-laserline) | `solver.max_iters: usize = 200`, `solver.verbosity: usize = 0` |
| flat `max_iters`/`verbosity`/`robust_loss` | `solver: SolverConfig` |
| `JointCameraFixMask` | `CameraFixMask` (core) |

## Consequences

- **Wire shape changes for all eight configs.** Consumers migrating in R3:
  pipeline problem/steps modules; bench (`registry.rs` override structs,
  `run.rs` config literals, `record.rs`); the app (`run.rs` test payloads,
  `RunWorkspace/presets.ts` override trees); the Python mirror
  (`models.py`); private examples (six files); public
  `laserline_device_session.rs`; tutorials; `book/src/rig_extrinsics.md`
  (embedded struct copy — rewritten).
- The schema-driven UI (ADR 0018) regenerates from `default_config_cmd`
  and picks up the grouped shapes automatically; only hand-written preset
  overrides need edits.
- The D2 deletions may shift rig-family fit numbers marginally (improvement
  direction expected); the Q2 drift gates and a reviewed refreeze arbitrate.
- Python parity gaps discovered during the survey (missing
  `RigHandeyeConfig.sensor` mirror, missing `RigHandeyeLaserlineConfig`
  mirror) are R5's scope, tracked separately.
