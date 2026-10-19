# ADR 0011: Manual Parameter Initialization Workflow

- Status: Accepted
- Date: 2026-03-15 (revised 2026-04-29)

## Context

In industrial calibration environments, users often have prior knowledge of camera parameters:
focal lengths from datasheets, sensor tilt angles from factory measurements, laser plane geometry
from mechanical drawings. The automatic linear initialization (Zhang's method, DLT) can produce a
poor starting point for non-linear optimization when conditioning is limited (narrow baseline, few
views, large tilt angles, noisy homographies); in the worst case it fails outright (for example
"invalid sign for lambda; check homographies"). Users need a way to seed individual parameter
groups without rewriting the pipeline.

## Decision

Every problem type exposes a seeded init step, `step_init_with_seed()`, alongside `step_init()`.
Multi-stage problem types name the stage in the function (for example
`step_intrinsics_init_all_with_seed`, `step_rig_init_with_seed`, `step_handeye_init_with_seed`).

```rust
// Automatic path:
step_init(&mut session, None)?;
step_optimize(&mut session, None)?;

// Seeded path:
step_init_with_seed(&mut session, PlanarManualInit {
    intrinsics: Some(nominal_camera_k),
    distortion: None,  // auto-initialized (or zeros when intrinsics seeded)
    poses: None,       // auto-recovered from homographies using manual intrinsics
}, None)?;
step_optimize(&mut session, None)?;
```

**Behavioral contract**:

1. `step_init_with_seed(session, manual, opts)` seeds any provided fields into the session state,
   then auto-initializes all remaining fields using the same routines as `step_init`.
2. After it returns, the session state is fully initialized, the same postcondition as
   `step_init`. `step_optimize` is unchanged.
3. `step_init(session, opts)` is a thin wrapper that calls the seeded step with
   `ManualInit::default()`, so the auto path and the all-`None` seeded path are identical.
4. For problem types with multiple init stages, each stage has its own seeded step with a
   corresponding typed `ManualInit` struct.

**ManualInit struct rules**:

- All fields are `Option<T>`. `None` means "auto-initialize this group"; `Some(value)` means "use
  this value verbatim, do not auto-initialize".
- Derive `Debug`, `Clone`, `Default`, `serde::Serialize`, `serde::Deserialize`. Serde is required
  for Python binding deserialization via `pythonize` and for JSON config files.
- Marked `#[non_exhaustive]`, so adding a seedable field is not a breaking change; construct them
  with `Default::default()` and assign fields.
- Placed in `steps.rs` alongside the step functions; re-exported from `mod.rs`.

**Partial initialization semantics**:

- When `intrinsics` is provided manually but `poses` is `None`, poses are auto-recovered from
  per-view homographies using the **manually provided intrinsics** (not auto-estimated ones). This
  keeps the geometric chain consistent with the seed.
- When `intrinsics` is seeded and `distortion` is `None`, distortion defaults to
  `BrownConrady5::default()` (zeros): the auto distortion fit is coupled with iterative
  intrinsics estimation and does not run when intrinsics are seeded.
- Workflow-specific zeroing (e.g., Scheimpflug pipelines fix tangential distortion `p1=p2=0`) only
  applies on the auto path. When the user supplies a manual `distortion`, they get exactly what
  they pass.
- The init log entry records which fields were manual vs auto, e.g.:
  `"init: fx=800.0, fy=780.0 ... (manual: intrinsics; auto: distortion, poses)"`.
- `per_cam_distortion` only takes effect when `per_cam_intrinsics` is also seeded.

**RigExtrinsics coupling rule**:

`cam_se3_rig` and `rig_se3_target` are geometrically coupled; providing one without the other is
ambiguous. The `ManualInit` for the rig stage requires both or neither, enforced at runtime with a
clear `Error::InvalidInput` message and a `log_failure` entry. This applies to
`RigExtrinsicsManualInit` and `RigHandeyeRigManualInit`.

**Sensor model exception (LaserlineDevice)**:

For `LaserlineDevice`, the sensor model is a hardware property (not a calibrated parameter) and is
always taken from `session.config.init.sensor_init` regardless of manual init. The corresponding
`LaserlineDeviceManualInit` struct intentionally omits a `sensor` field.

**Seedable fields per problem type**:

| Problem type                  | ManualInit type(s)                                                 |
|-------------------------------|---------------------------------------------------------------------|
| `PlanarIntrinsics`            | `PlanarManualInit`                                                  |
| `ScheimpflugIntrinsics`       | `ScheimpflugManualInit`                                             |
| `LaserlineDevice`             | `LaserlineDeviceManualInit` (sensor excluded)                       |
| `SingleCamHandeye`            | `SingleCamIntrinsicsManualInit` + `SingleCamHandeyeManualInit`      |
| `RigExtrinsics`               | `RigIntrinsicsManualInit` + `RigExtrinsicsManualInit`               |
| `RigHandeye`                  | `RigHandeyeIntrinsicsManualInit` + `RigHandeyeRigManualInit` + `RigHandeyeHandeyeManualInit` |
| `RigLaserlineDevice`          | `RigLaserlineDeviceManualInit` (planes only)                        |

## Consequences

Positive:

- Expert users can bypass unreliable auto-init for specific parameter groups.
- `step_init` and `step_optimize` keep their signatures and behavior.
- Typing is per-problem and discoverable via rustdoc and IDE completion.
- Init source is recorded in the session log (auditable, human-readable).

Negative:

- Each problem module carries one or more extra structs and seeded step functions.
- Users who provide incorrect manual seeds may get worse results: there is no automatic validation
  of physical plausibility beyond cardinality checks (length matches num_cameras / num_views) and
  the RigExtrinsics coupling rule.
