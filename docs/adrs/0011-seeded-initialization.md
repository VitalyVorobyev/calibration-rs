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
- The camera model (which kinds, e.g. a Scheimpflug sensor) is
  configuration, not a seed (ADR 0013). The values it calibrates, the
  sensor tilt included, are seedable like any other: `sensor` in the
  single-camera seeds, `per_cam_sensors` in the rig seeds. A non-zero tilt
  for an untilted camera is a typed error. A known mount tilt that should
  not move is a seed plus a `camera.fix.sensor` list. The laserline device
  holds its tilt by default: its mount tilt is a hardware property measured
  separately, so an unset `camera.fix.sensor` holds it at the seed (zero
  when unseeded), and an explicit list opts into estimating it.
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

### Seeds come from a device: its datasheet or a calibration of it

One seed value, `device_seed::DeviceSeed`, holds the device quantities by
camera id: each camera's lens, its mount in the rig (`cam_se3_rig`), its
laser plane (camera frame) and the rig's hand-eye. It has two sources, a
`DeviceSpec` (`DeviceSeed::from_spec`) and any calibration export through
`CalibratedCameras` (`DeviceSeed::from_calibration`, the camera ids and
image sizes given, since an export holds neither), and one sink: its
derivations into the `*ManualInit` seeds above, which stay the only
mechanism the steps consume.

- **A lens is `Calibrated { camera }` or `Nominal { intrinsics, sensor }`**,
  not parallel optional fields. A calibrated lens is the camera whole in its
  model; the model is configuration, so the lens must be of the configured
  model, and `DeviceSeed::nominal()` is the explicit way to start another
  model from a calibration (exact embeddings between nested models are not
  converted). A sensor homography cannot be expressed by any `*ManualInit`
  and is a typed `UnseedableSensor`.
- **A seed is an initial value, never a prior.** It carries no σ: a unit's
  own previous posterior used as a prior would bias a recalibration toward
  the old value exactly when the device changed. A prior's width, if one is
  ever added, is the spread across units, measured for that use.
- **Nothing of a capture is carried**: target poses, robot-pose
  corrections, residuals, the covariance and the configuration stay behind.
- **The rig frame is re-gauged at use.** A seed's mounts and hand-eye are in
  its own rig frame (a drawing's, or another run's reference camera). The
  derivations that take the config's `reference_camera_idx` re-express them
  so the reference camera is the identity (`C_i · C_ref⁻¹`, `gripper_se3_rig
  · C_ref⁻¹`, `C_ref · rig_se3_base`); laser planes are in camera frames.
  A single camera is a rig of one whose frame is the camera's.
- **Two workflows treat a seeded value as more than a start**, and both are
  kept and documented: a seeded Scheimpflug tilt switches planar intrinsics
  to the trusted warm start, whose `k1` sweep replaces a seeded `k1` with
  its best fixed start and whose refine is bounded around the seed; and a
  laserline device holds its tilt at the seed by default (above).
- Per-field seeds are all-or-none across a rig's cameras: mixed calibrated
  and nominal lenses are a typed `PartialField` for the rig's distortion,
  never zeros for the nominal ones.
- **A device type's seed is a median** (`DeviceSeed::aggregate`, one seed
  per unit, calibrated lenses only): every unit re-gauged to the camera
  with the smallest id, the componentwise median of each camera parameter
  and translation, the chordal (Markley) mean of a rotation corrected by
  the componentwise median of the tangent vectors about it, laser planes
  signed `d > 0` then medianed. Quantities only some units have are omitted
  and listed, never filled. A median lens that folds where every unit's is
  valid is `AggregateFolds`. The result is bit-identical under any order
  of the units. Its spread (median, `1.4826·MAD`, min, max per value) is
  descriptive, not a prior. The componentwise tangent median is not
  rotation-equivariant: with an outlier pulling the chordal mean and inlier
  turns off the tangent axes, it lands within the inliers' spread, not at
  their mean (pinned by the aggregate tests).

The `DeviceSpec` free functions (`scheimpflug_seed`, `rig_intrinsics_seed`,
`nominal_cam_se3_rig`, `rig_layout_seed`, `handeye_seed`) stay and share
the seed's conversions, so their values equal the `DeviceSeed` path's bit
for bit (in the drawing's frame).

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
- A previous calibration of the unit, or of another unit of its type, can
  start the next one through the same seeded steps.
