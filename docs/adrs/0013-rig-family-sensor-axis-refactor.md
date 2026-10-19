# ADR 0013: Rig Problems: Sensor Mode as a Config Axis

- Status: Accepted
- Date: 2026-05-01

## Context

Multi-camera rig calibration comes in two sensor flavours (pinhole and Scheimpflug) and several
workflows (extrinsics-only, hand-eye, laser plane). Pinhole and Scheimpflug variants of a workflow
share state shape, config fields and the same step skeleton (per-camera intrinsics bootstrap, rig
fit, optional hand-eye fit); they differ only in one extra `per_cam_sensors` field, a few tilt
config fields and the optim entry point. Workflows, by contrast, differ fundamentally in state,
step sequence, residual blocks and exports. Modelling every (sensor x workflow) pair as its own
module duplicates code, and every cross-cutting change then has to be applied once per sibling.

## Decision

Factor rig problems along **one axis, the sensor model**, and only that axis. The workflow stays a
separate module per problem type: `rig_extrinsics`, `rig_handeye`, `rig_laserline_device`
(plus `rig_handeye_laserline`).

The factoring uses **composition over traits**:

1. An internal helper module `crate::rig_family` owns the shared sensor-aware primitives:
   - `RigSensorBundle` (`pub(crate)`): `cameras: Vec<PinholeCamera>` plus an optional
     `Option<Vec<ScheimpflugParams>>`.
   - `RigIntrinsicsSeeds`: manual-init seeds carrying pinhole intrinsics/distortion and
     Scheimpflug sensors as `Option<Vec<...>>`.
   - `SensorFlavour` (`pub(crate)`): tags which bootstrap to run.
   - `bootstrap_rig_intrinsics(...)`: the per-camera bootstrap loop; for Scheimpflug it also
     threads per-camera tilt parameters (seeded or default).
   - Small pure helpers (`views_to_planar_dataset`, `estimate_target_pose`, ...) used across
     workflow modules.

2. A user-facing `SensorMode` enum, defined once in `rig_family` and re-exported by
   `rig_extrinsics` and `rig_handeye`:

   ```rust
   #[non_exhaustive]
   #[serde(tag = "kind")]
   pub enum SensorMode {
       Pinhole,
       Scheimpflug {
           init_tilt_x: f64,
           init_tilt_y: f64,
           fix_scheimpflug: ScheimpflugFixMask,
           distortion_mask_in_percam_ba: DistortionFixMask,
           refine_scheimpflug_in_rig_ba: bool,
           distortion_model: DistortionKind,
       },
   }
   ```

   `RigExtrinsicsConfig` and `RigHandeyeConfig` carry `sensor: SensorMode`. Step functions match
   on it and dispatch to the pinhole or Scheimpflug optim entry point. The per-camera Scheimpflug
   distortion mask defaults to `DistortionFixMask::radial_only()`, because tangential distortion
   can absorb tilt-like signal.

3. Workflow `Output` types are enums that preserve each flavour's optim estimate:

   ```rust
   pub enum RigExtrinsicsOutput { Pinhole(..), Scheimpflug(..) }
   pub enum RigHandeyeOutput    { Pinhole(..), Scheimpflug(..) }
   ```

   Accessor methods (`cam_to_rig()`, `cameras()`, `sensors()`, `mean_reproj_error()`, ...) give a
   uniform read surface so downstream code rarely matches on the variant.

4. Workflow `Export` types carry Scheimpflug-only data as
   `sensors: Option<Vec<ScheimpflugParams>>` (`None` for pinhole rigs, `Some(_)` for Scheimpflug).

`rig_laserline_device` stays its own module: it is the only laser-plane rig workflow. Its upstream
calibration is built from a rig hand-eye export and accepts both flavours (a pinhole upstream is
exactly zero tilt).

## Rejected alternatives

### Two-axis trait matrix (sensor x workflow)

A `RigWorkflow` trait parameterised by sensor model would maximise generic reuse. Rejected:

- Hand-eye and extrinsics-only workflows have genuinely different state shapes; making state
  generic adds layers of associated types that readers must re-derive every time.
- Step sequences differ; abstracting the sequence into a trait obscures the per-workflow control
  flow a calibration engineer needs to reason about.
- The common code is small free functions, not behaviour worth an interface.

### Separate module per (sensor x workflow), sharing only a helper module

Extracting helpers removes pure duplication but leaves sibling modules whose configs, states and
exports must each be edited for every cross-cutting change. Collapsing the sibling modules, not the
helper extraction, is the load-bearing simplification.

### Scheimpflug-only advanced knobs in the unified config

Extra per-camera Scheimpflug workarounds (seed overrides, shared-init fallbacks, separate
fix-masks for the per-camera BA) were not carried into `SensorMode::Scheimpflug`: they served
narrow-FOV / Zhang-failure cases that manual init (ADR 0011) now covers. If a real workflow needs
more, fields can be added to the `#[non_exhaustive]` variant without changing its shape.

## Consequences

Positive:

- Adding a sensor variant is a `SensorMode` variant plus an inner branch in the step functions,
  not a new module.
- Cross-cutting changes touch one file per workflow.
- `SensorMode` has one definition, in `rig_family`.

Negative:

- Output enums need a match at internal use sites (mitigated by the accessors).
- `Option<Vec<ScheimpflugParams>>` on exports carries a runtime invariant ("`Some(_)` iff the
  sensor mode was Scheimpflug") that the type system does not enforce. Helpers that need sensors
  return an explicit error when handed a pinhole export.

## Out of scope

- Workflow-axis abstraction (kept per module).
- Sensor variants beyond Pinhole and Scheimpflug.
