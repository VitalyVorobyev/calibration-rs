# ADR 0011: Seeded Initialization

## Context

Linear initialization (Zhang, DLT, Tsai–Lenz) gives a poor start, or fails
outright, when conditioning is limited: few or similar views, a narrow
baseline, a tilted sensor. Scheimpflug intrinsics are ill-conditioned from
scratch: tilt, focal length and radial distortion trade off along a long,
flat valley, and Zhang's pinhole init on a tilted sensor lands in the wrong
basin. Meanwhile the values that fix this are on hand: lens focal length
and pixel pitch from a datasheet, the Scheimpflug mount tilt and the camera
mounts from a mechanical drawing.

## Decision

### Every problem type can be seeded

Each problem type exposes `step_init_with_seed` next to `step_init`;
multi-stage problems name the stage (`step_rig_init_with_seed`,
`step_handeye_init_with_seed`, …).

- The seeded step uses each provided field verbatim and auto-initializes
  the rest with the same routines as `step_init`. Its postcondition equals
  `step_init`'s, and `step_init` is the seeded step with an empty seed.
- Seeds are typed per stage (`*ManualInit`). Every field is an `Option`
  (`None` = auto); the structs are serde-enabled, `Default` and
  `#[non_exhaustive]`, and live in the module's `steps.rs`.
- Auto-initialized fields are derived *from* the seed: poses recovered with
  seeded intrinsics use those intrinsics. Seeded intrinsics without a
  distortion seed start from zero distortion, because the auto distortion
  fit is coupled to the auto intrinsics estimate. Workflow-specific zeroing
  (Scheimpflug `p1 = p2 = 0`) applies only on the auto path.
- `cam_se3_rig` and the per-view `rig_se3_target` are geometrically coupled:
  a rig seed provides both or neither, enforced with a typed error.
- A hardware property that is not calibrated (the laserline device's
  sensor model) is not seedable; it comes from the config.
- The init log records which fields were seeded and which automatic.

### Seeding is the official route for Scheimpflug and rig calibration

Seeded initialization is the supported default for Scheimpflug intrinsics
and for rigs, and the route every acceptance dataset runs. Unseeded
Scheimpflug init remains available, is documented as experimental and logs
a warning. A trusted tilt seed switches the optimizer from its cold-start
tilt sweep to a bounded refine around the seed; the solve's design and the
convergence-basin evidence are in `docs/notes/scheimpflug-intrinsics.md`.

Each acceptance entry carries a hard per-camera reprojection gate in the
bench registry. A gate is never relaxed to make a run pass.

### Seeds come from a device specification

A `DeviceSpec` sidecar (`spec.json` next to the dataset manifest, in
`vision-calibration-dataset`) describes the hardware, not the data:
per-camera focal length, pixel pitch, resolution, optional principal point
and Scheimpflug mount tilt, and optionally the rig layout (camera mounts,
hand-eye mount, laser planes). `vision_calibration_pipeline::device_seed`
derives the `*ManualInit` seeds from it.

- **Datasheet units, named in the field** (`focal_mm`, `pixel_pitch_um`,
  `tilt_x_deg`, `translation_mm`): a spec is a transcription of a datasheet
  and a drawing, and transcription must not require unit conversion.
  Conversion happens once, in the derivation.
- **Plain serde types with no core dependency**, so internal refactors do
  not change the wire format.
- **Cameras are keyed by id**; derivation takes the caller's id list, so a
  mismatch is a typed error, never positional misalignment.
- **Frames follow ADR 0009.** The spec stores the drawing's `rig_se3_cam`;
  the hand-eye mount is mode-tagged (`EyeInHand { gripper_se3_rig }` |
  `EyeToHand { rig_se3_base }`) so its frame cannot be misread. Nominal
  camera mounts are returned as poses, not as a rig seed, because of the
  coupling rule above.
- Validation is typed and fail-fast; unknown fields are load errors;
  `version` bumps on breaking revisions. Specs of private datasets stay out
  of the repository.

## Consequences

- An expert can bypass unreliable auto-init for any parameter group.
- A coarse datasheet seed calibrates strongly tilted Scheimpflug cameras
  deterministically; the basin study measures how much spec error it
  tolerates.
- Wrong seeds can give worse results: beyond cardinality and the coupling
  rule there is no plausibility check.
