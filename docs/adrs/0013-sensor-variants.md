# ADR 0013: Lens and Sensor Variants Are Camera Slots, Not Workflows

## Context

A planar intrinsics calibration, a rig, a hand-eye or a laser calibration
can each run with any lens and any sensor:
- a perspective, fisheye or telecentric lens;
- a straight or tilted (Scheimpflug) sensor.

Two ways of offering these variants had grown side by side:
- a separate workflow per sensor (`planar_intrinsics` next to
  `scheimpflug_intrinsics`);
- a sensor enum on rigs (`SensorMode`).

The optimizer had also grown a builder per sensor (`rig_extrinsics` next
to `rig_extrinsics_scheimpflug`, `handeye` next to `handeye_scheimpflug`).
Every new lens model would have multiplied that pattern.

## Decision

The workflow is a module, and the camera model is a config value
(ADR 0005).

- **One workflow per calibration task.** Planar intrinsics, hand-eye, rig
  extrinsics, rig hand-eye and the laserline device each take
  `camera: CameraConfig`; rigs add per-camera overrides, and the rig laser
  workflows take each camera's model from the upstream rig. There is no
  per-sensor workflow and no `SensorMode`.
- **One optimizer builder per task.** Builders are written against
  `CameraModelSpec`. They build the camera blocks through one shared helper
  that handles packing, fix lists and covariance labels.
- **Sensor-specific behaviour hangs off the spec's kinds, not the
  workflow:**
  - the default fix list (a Scheimpflug sensor holds everything but the
    low-order radial terms: `k3, p1, p2` for Brown-Conrady);
  - zeroing tangential terms in the linear stage of a single-camera
    workflow (the rig workflows keep them free for every camera);
  - the cold-start tilt sweep, which runs when the sensor is Scheimpflug
    and no tilt seed is given: the linear tilt sweep, then the staged tilt
    multi-start on the camera's own views;
  - the seeded route (ADR 0011);
  - a tilted sensor needs a distortion model in the single-camera
    workflows, whose radial terms tell the tilt apart from a
    principal-point shift.

  Each is defined once (`CameraModelSpec::default_fix`,
  `planar_family::{bootstrap_camera, optimize_camera}`,
  `CameraConfig::validate_single_camera`) and runs in every workflow that
  meets the condition.
- **Each step's output is unambiguous**, because the result holds
  `CameraParams`, whose sensor slot states what was calibrated.

The single-camera Scheimpflug workflow therefore merges into planar
intrinsics. Its distinct behaviour (the radial-only default, the
tangential-free linear stage, the seeded tilt route) becomes
kind-dependent defaults of the merged workflow.

## Consequences

- **Fewer types and modules:** the planar-intrinsics module serves both
  sensors; one workflow, its config, export and Python binding go away;
  the per-sensor optimizer builders collapse into one per task.
- **Defaults depend on the sensor:** a config field whose default depends
  on the sensor, such as `init.fix_tangential`, is an `Option`; `None`
  means the spec's default, documented on the field. The solve budget
  has one default for both sensors, sized for the Scheimpflug tilt valley:
  the solver stops at convergence, so the cap only bounds runs that do not
  converge.
- **Wider testing:** every workflow is tested with both sensors and with
  each lens model, through the synthetic matrix.

## Alternatives rejected

- **Workflow per sensor** — every cross-cutting change edits sibling
  configs, states and exports, and each new lens multiplies modules.
- **A trait matrix (sensor × workflow)** — workflows differ in state shape
  and step sequence; a generic workflow hides the control flow a
  calibration engineer needs to read. The variation is in the camera
  model, and that is data.
