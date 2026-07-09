# Single-camera hand-eye calibration

> Onboarding tutorial. Runnable companion:
> [`handeye_session.rs`](../../crates/vision-calibration/examples/handeye_session.rs).
> Run it with `cargo run -p vision-calibration --example handeye_session --release`.

## Why

A single camera mounted on (or observing) a robot arm needs both its
intrinsics and the fixed transform between the camera and the robot's
frames — the "hand-eye" transform — before pixel measurements can be
turned into robot-frame 3D points. `SingleCamHandeyeProblem` calibrates
both jointly from one dataset: a planar target seen from several robot
poses.

## Mental model

- **Input**: `SingleCamHandeyeInput { views: Vec<View<HandeyeMeta>> }`.
  Each view pairs planar-target 2D-3D correspondences with the gripper
  pose at capture time — `HandeyeMeta { base_se3_gripper: Iso3 }`.
- **Pose naming**: the project's `frame_se3_frame` convention (ADR 0009).
  `base_se3_gripper` reads "gripper pose expressed in the base frame"
  (`T_B_G`). The unknowns the calibration solves for are named the same
  way: `gripper_se3_camera` (EyeInHand) or `camera_se3_base` (EyeToHand).
- **Two mounting modes**, chosen by `HandeyeInitConfig::handeye_mode`:
  - `HandEyeMode::EyeInHand` (the config default) — the camera rides on
    the gripper, observing a **fixed** target. Solves for
    `gripper_se3_camera` (`T_G_C`) and `base_se3_target` (`T_B_T`).
  - `HandEyeMode::EyeToHand` — the camera is fixed in the scene,
    observing a target **mounted on the gripper**. Solves for
    `camera_se3_base` (`T_C_B`) and `gripper_se3_target` (`T_G_T`).

  The export's four pose fields are mutually exclusive by mode: the
  active mode populates its pair, the other pair is `None`.
- **Four steps**, the same intrinsics→handeye staging `RigHandeye` uses
  but for one camera:

  ```
  step_intrinsics_init      ← Zhang's method + iterative distortion fit
  step_intrinsics_optimize  ← per-camera bundle adjustment
  step_handeye_init         ← Tsai-Lenz DLT
  step_handeye_optimize     ← joint hand-eye bundle adjustment
  ```

  Each has a `*_with_seed` sibling (ADR 0011) and its own typed
  step-result struct (`SingleCamIntrinsicsInitResult`,
  `SingleCamHandeyeOptimizeResult`, …) mirroring the fields written into
  `session.state`.

## Walkthrough

We'll calibrate the KUKA chessboard dataset shipped in `data/kuka_1/`
(30 images + robot poses).

### 1. Build the input

```rust
use vision_calibration::prelude::*;
use vision_calibration::single_cam_handeye::{
    HandeyeMeta, SingleCamHandeyeInput, SingleCamHandeyeProblem, SingleCamHandeyeView,
    step_handeye_init, step_handeye_optimize, step_intrinsics_init, step_intrinsics_optimize,
};

let views: Vec<SingleCamHandeyeView> = samples
    .into_iter()
    .map(|s| SingleCamHandeyeView {
        obs: s.view,                                            // 2D-3D chessboard correspondences
        meta: HandeyeMeta { base_se3_gripper: s.robot_pose },    // T_B_G at capture time
    })
    .collect();

let input = SingleCamHandeyeInput::new(views)?;
```

`SingleCamHandeyeInput::new` rejects fewer than 3 views, or any view with
fewer than 4 correspondences (the minimum for a homography).

### 2. Run the four steps

```rust
let mut session = CalibrationSession::<SingleCamHandeyeProblem>::new();
session.set_input(input)?;

step_intrinsics_init(&mut session, None)?;
step_intrinsics_optimize(&mut session, None)?;
step_handeye_init(&mut session, None)?;
step_handeye_optimize(&mut session, None)?;

let export = session.export()?;
```

Or, equivalently, the convenience wrapper:

```rust
vision_calibration::single_cam_handeye::run_calibration(&mut session)?;
```

Real output on `data/kuka_1/` (30 chessboard views, EyeInHand — the
config default):

```
--- Step 2: Intrinsics Optimization ---
  Intrinsics: fx=2056.9, fy=2057.5, cx=962.9, cy=608.0
  Reprojection error: 0.1453 px

--- Step 3: Hand-Eye Initialization (Tsai-Lenz) ---
  Hand-eye |t|: 0.2809m

--- Step 4: Hand-Eye Optimization ---
  Final reprojection error: 1.1934 px
```

The intrinsics-only reprojection error (0.15 px) is much tighter than the
final joint hand-eye error (1.19 px) — expected: the hand-eye stage
additionally has to explain every view through one shared rigid
`gripper_se3_camera` and one shared `base_se3_target`, so any robot-pose
noise or mechanical play shows up here instead of in the per-view target
pose (which was free during intrinsics optimization).

### 3. Read the export

```rust
use vision_calibration::optim::HandEyeMode;

match export.handeye_mode {
    HandEyeMode::EyeInHand => {
        let gripper_se3_camera = export.gripper_se3_camera.unwrap(); // T_G_C
        let base_se3_target = export.base_se3_target.unwrap();       // T_B_T
    }
    HandEyeMode::EyeToHand => {
        let camera_se3_base = export.camera_se3_base.unwrap();       // T_C_B
        let gripper_se3_target = export.gripper_se3_target.unwrap(); // T_G_T
    }
}
```

`export.camera` is the calibrated `PinholeCamera` (intrinsics +
Brown-Conrady distortion); `export.per_feature_residuals` carries
per-corner reprojection records (see
[Per-feature residuals](./per-feature-residuals.md)).

## Config shape (post-ADR-0024)

`SingleCamHandeyeConfig` groups four shared sub-structs from
`vision_calibration::common::config`:

```rust
pub struct SingleCamHandeyeConfig {
    pub intrinsics: IntrinsicsInitConfig,  // init_iterations, fix_k3, fix_tangential, zero_skew
    pub handeye_init: HandeyeInitConfig,   // handeye_mode, min_motion_angle_deg
    pub solver: SolverConfig,              // max_iters, verbosity, robust_loss
    pub robot_poses: RobotPoseConfig,      // refine, rot_sigma, trans_sigma
}
```

All four groups keep their shared workspace defaults —
`SingleCamHandeyeConfig` is the one problem type that needs no
per-problem override (unlike, say, `ScheimpflugIntrinsicsConfig`, which
bumps `solver.max_iters` to 120 and flips two `init` defaults). See
[distortion model selection](./distortion-model-selection.md) for the
`IntrinsicsInitConfig` fields in more depth.

`robot_poses.refine` (default `true`) adds a per-view se(3) correction
during `step_handeye_optimize`, softly regularized by `rot_sigma` /
`trans_sigma`; the corrections land in `export.robot_deltas`.

To switch to EyeToHand:

```rust
use vision_calibration::single_cam_handeye::SingleCamHandeyeConfig;
use vision_calibration::optim::HandEyeMode;

let mut config = SingleCamHandeyeConfig::default();
config.handeye_init.handeye_mode = HandEyeMode::EyeToHand;
session.set_config(config)?;
```

## Common variations

### Manual seeding

See [Manual initialization](./manual-init.md). The load-bearing entry
points are `step_intrinsics_init_with_seed(&mut session,
SingleCamIntrinsicsManualInit { intrinsics: Some(k), .. }, None)` and
`step_handeye_init_with_seed(&mut session, SingleCamHandeyeManualInit {
.. }, None)`; both accept partial or full seeds and the session log
records which fields came from where.

### Loading real data: `dataset.toml` vs. `spec.json`

Two different manifests cover two different needs — it's worth being
precise about which one `SingleCamHandeyeProblem` actually consumes
today:

- **`dataset.toml`** ([ADR 0016](../adrs/0016-dataset-manifest.md))
  describes *where the data lives*: image glob patterns, the target
  definition, the robot-pose file and its `pose_convention`, and how
  images pair with pose rows (`pose_pairing`).
  `vision_calibration::dataset_runner::build_single_cam_handeye_input`
  turns a `DatasetSpec` with `topology = "single_cam_handeye"` directly
  into a `SingleCamHandeyeInput`, running (cached) corner detection along
  the way:

  ```rust
  let result = vision_calibration::dataset_runner::build_single_cam_handeye_input(
      &spec, base_dir, &cache, false,
  )?;
  let input = result.input;
  ```

- **`spec.json`** ([ADR 0023](../adrs/0023-device-spec-seed-derivation.md),
  design record: [`docs/DESIGN-device-spec.md`](../DESIGN-device-spec.md))
  describes *the physical device*: lens focal length, pixel pitch,
  Scheimpflug mount tilt, nominal rig layout. It feeds
  `vision_calibration::device_seed`'s derivation functions
  (`scheimpflug_seed`, `rig_intrinsics_seed`, `nominal_cam_se3_rig`,
  `handeye_seed`, `rig_layout_seed`), which build ADR 0011 manual-init
  seeds for the **rig** family of problem types.

  As of this writing there is **no `single_cam_handeye`-specific
  derivation function** — `SingleCamHandeyeProblem` has no rig layout to
  seed, and its intrinsics are plain pinhole (no Scheimpflug tilt), so
  the existing `device_seed` helpers don't apply here. If you have a
  `spec.json` for a single-camera hand-eye rig, read its
  `focal_mm` / `pixel_pitch_um` / `resolution_px` fields yourself and
  build an `FxFyCxCySkew` for `SingleCamIntrinsicsManualInit` — the same
  formula `device_seed::scheimpflug_seed` uses internally
  (`f_px = focal_mm * 1000 / pixel_pitch_um`, principal point defaulting
  to the resolution center).

### Eye-to-hand geometry sanity check

In EyeToHand mode, the pixel→3D chain for a fixed camera observing a
gripper-mounted target composes as `T_C_T = T_C_B · T_B_G · T_G_T` — the
same identity `step_handeye_init_with_seed` uses to auto-derive
`camera_se3_base` when only `gripper_se3_target` is seeded (see
[Manual initialization § Hand-eye coupling](./manual-init.md#hand-eye-coupling)).

## What to read next

- [ADR 0009](../adrs/0009-coordinate-and-pose-conventions.md) —
  `frame_se3_frame` naming and the EyeInHand/EyeToHand pose conventions
  in full.
- [ADR 0011](../adrs/0011-manual-initialization-workflow.md) — manual
  seeding, including the hand-eye coupling rule.
- [ADR 0024](../adrs/0024-config-vocabulary.md) — why the config is
  grouped this way.
- [Manual initialization](./manual-init.md) and
  [Per-feature residuals](./per-feature-residuals.md) — the two
  tutorials this one builds on.
- [`handeye_session.rs`](../../crates/vision-calibration/examples/handeye_session.rs)
  — the example this tutorial is built on.
