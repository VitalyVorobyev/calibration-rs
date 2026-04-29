# ADR 0011: Manual Parameter Initialization Workflow

- Status: Accepted
- Date: 2026-03-15 (revised 2026-04-29)

## Context

In industrial calibration environments, users often have prior knowledge of camera parameters —
focal lengths from datasheets, sensor tilt angles from factory measurements, laser plane geometry
from mechanical drawings. The default pipeline always runs automatic linear initialization (Zhang's
method, Tsai-Lenz DLT), which may produce poor starting points for non-linear optimization when
the auto-init quality is limited (e.g., narrow baseline, few views, large tilt angles, real-data
homography conditioning issues).

A concrete failure mode this ADR unblocks: the puzzle 130x130 rig sees Zhang's method fail on a
specific camera with "invalid sign for lambda; check homographies" — the user has no way to seed
that camera's intrinsics from datasheet values without rewriting the pipeline.

## Decision

Add a `step_set_init()` function (and per-stage variants for multi-stage problem types) alongside
the existing `step_init()`:

```rust
// Automatic path (unchanged):
step_init(&mut session, None)?;
step_optimize(&mut session, None)?;

// Manual path:
step_set_init(&mut session, PlanarManualInit {
    intrinsics: Some(nominal_camera_k),
    distortion: None,  // auto-initialized (or zeros when intrinsics seeded)
    poses: None,       // auto-recovered from homographies using manual intrinsics
}, None)?;
step_optimize(&mut session, None)?;
```

**Behavioral contract**:

1. `step_set_init(session, manual, opts)` seeds any provided fields into the session state, then
   auto-initializes all remaining fields using the same routines as `step_init`.
2. After `step_set_init` returns, the session state is fully initialized — same postcondition as
   `step_init`. `step_optimize` requires no changes.
3. `step_init(session, opts)` is refactored to call
   `step_set_init(session, ManualInit::default(), opts)`, preserving its existing signature and
   behavior.
4. For problem types with multiple init stages (SingleCamHandeye, RigExtrinsics, RigHandeye, the
   Scheimpflug rig variants), each stage gets its own `step_set_*` function with a corresponding
   typed `ManualInit` struct.

**ManualInit struct rules**:

- All fields are `Option<T>`. `None` means "auto-initialize this group"; `Some(value)` means "use
  this value verbatim, do not auto-initialize".
- Derive `Debug`, `Clone`, `Default`, `serde::Serialize`, `serde::Deserialize`. Serde is required
  for Python binding deserialization via `pythonize` and for JSON config files.
- Do NOT apply `#[non_exhaustive]` — users construct these with `..Default::default()` and
  `#[non_exhaustive]` breaks that pattern across crate boundaries.
- Placed in `steps.rs` alongside the step functions; re-exported from `mod.rs`.

**Partial initialization semantics**:

- When `intrinsics` is provided manually but `poses` is `None`, poses are auto-recovered from
  per-view homographies using the **manually provided intrinsics** (not auto-estimated ones). This
  keeps the geometric chain consistent with the seed.
- When `intrinsics` is seeded and `distortion` is `None`, distortion defaults to
  `BrownConrady5::default()` (zeros) — the auto distortion fit is coupled with iterative
  intrinsics estimation and does not run when intrinsics are seeded.
- Workflow-specific zeroing (e.g., Scheimpflug pipelines fix tangential distortion `p1=p2=0`) only
  applies on the auto path. When the user supplies a manual `distortion`, they get exactly what
  they pass.
- The init log entry records which fields were manual vs auto, e.g.:
  `"init: fx=800.0, fy=780.0 ... (manual: intrinsics; auto: distortion, poses)"`.

**RigExtrinsics coupling rule**:

`cam_se3_rig` and `rig_se3_target` are geometrically coupled; providing one without the other is
ambiguous. The corresponding `ManualInit` for the rig stage requires both or neither, enforced at
runtime with a clear `Error::InvalidInput` message and a `log_failure` entry. This rule applies to
`RigExtrinsicsManualInit`, `RigHandeyeRigManualInit`, `RigScheimpflugExtrinsicsRigManualInit`, and
`RigScheimpflugHandeyeRigManualInit`.

**Sensor model exception (LaserlineDevice)**:

For `LaserlineDevice`, the sensor model is a hardware property (not a calibrated parameter) and is
always taken from `session.config.init.sensor_init` regardless of manual init. The corresponding
`LaserlineDeviceManualInit` struct intentionally omits a `sensor` field.

## Implementation status (2026-04-29)

All nine problem types support manual init. Per-problem types and their step_set_* functions:

| Problem type                  | ManualInit type(s)                                                 |
|-------------------------------|---------------------------------------------------------------------|
| `PlanarIntrinsics`            | `PlanarManualInit`                                                  |
| `ScheimpflugIntrinsics`       | `ScheimpflugManualInit`                                             |
| `LaserlineDevice`             | `LaserlineDeviceManualInit` (sensor excluded)                       |
| `SingleCamHandeye`            | `SingleCamIntrinsicsManualInit` + `SingleCamHandeyeManualInit`      |
| `RigExtrinsics`               | `RigIntrinsicsManualInit` + `RigExtrinsicsManualInit`               |
| `RigHandeye`                  | `RigHandeyeIntrinsicsManualInit` + `RigHandeyeRigManualInit` + `RigHandeyeHandeyeManualInit` |
| `RigScheimpflugExtrinsics`    | `RigScheimpflugIntrinsicsManualInit` + `RigScheimpflugExtrinsicsRigManualInit` |
| `RigScheimpflugHandeye`       | `RigScheimpflugHandeyeIntrinsicsManualInit` + `RigScheimpflugHandeyeRigManualInit` + `RigScheimpflugHandeyeHandeyeManualInit` |
| `RigLaserlineDevice`          | `RigLaserlineDeviceManualInit` (planes only)                        |

The `RigScheimpflugHandeye` intrinsics stage internally delegates to the existing
`step_intrinsics_init_all` with `manual` seeds temporarily injected into
`config.intrinsics.initial_cameras` / `initial_sensors` (preserving Zhang+fallback logic).
`per_cam_distortion` only takes effect when `per_cam_intrinsics` is also seeded for that problem
type.

## Consequences

Positive:

- Expert users can bypass unreliable auto-init for specific parameter groups.
- `step_init` backward compatibility is maintained (same signature, same behavior).
- `step_optimize` requires no changes.
- Typing is per-problem and discoverable via rustdoc and IDE completion.
- Init source recorded in session log (auditable, human-readable).
- Unblocks the puzzle 130x130 rig's stuck Zhang's-init failure on real data.

Negative:

- Each problem module grows by one or more structs and step_set_* functions.
- Users who provide incorrect manual seeds may get worse results — no automatic validation of
  physical plausibility beyond cardinality checks (length matches num_cameras / num_views) and the
  RigExtrinsics coupling rule.

## Required follow-up

- Python bindings for the `ManualInit` types across all problem types (matching the typed
  warm-start pattern landed in commit `8351680` for `LaserlinePlane` in `RigLaserlineDeviceInput`).
- Add a regression test using the puzzle rig real-data dataset that confirms manual intrinsics
  seeding for the failing camera produces a successful end-to-end calibration.
- Promote `RigScheimpflugHandeye`'s intrinsics-stage manual init from the temp-config-mutation
  delegate pattern to a direct seed-driven path once the Zhang fallback logic is factored out into
  a shared helper.
