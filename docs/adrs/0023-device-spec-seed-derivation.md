# ADR 0023: `DeviceSpec` — Device Specification Schema and Seed Derivation

- Status: Accepted
- Date: 2026-07-02

## Context

ADR 0022 made spec-seeded initialization the official acceptance route for
Scheimpflug intrinsics: the values needed to seed the solve — lens focal
length, sensor pixel pitch, and the mechanically-set Scheimpflug mount tilt —
are datasheet/drawing values the device owner already has. But as shipped,
those seeds are hand-coded constants and env knobs scattered across the
private examples (`RTV3D_REF_FOCAL`, `RTV3D_RINGGRID_FOCAL`,
`RTV3D_RINGGRID_TILT_X`, `NOMINAL_TILT_X`, `RTV3D_SEED=generic|oracle`).
Every new dataset re-invents its own seeding, the ringgrid rig needed an
env-var *sweep* to find a focal the spec should have provided, and the
acceptance harness (S4) has no structured way to ask "what does the device
spec say?".

Track S introduces a structured **device-spec → seed** layer: a sidecar
`spec.json` next to each dataset manifest describing the *hardware* (not the
data), plus derivation functions producing the existing ADR 0011 manual-init
structs. Note the manual-init surface has moved since ADR 0011 was written:
the `RigScheimpflug*` problem types were collapsed by ADR 0013 — the rig
targets today are `RigExtrinsics`/`RigHandeye` with
`SensorMode::Scheimpflug`, seeded through `RigHandeyeIntrinsicsManualInit`
(`per_cam_intrinsics`/`per_cam_sensors`), `RigHandeyeRigManualInit`, and
`RigHandeyeHandeyeManualInit`.

## Decision

### Schema (`vision_calibration_dataset::device_spec`)

`DeviceSpec` lives in `vision-calibration-dataset` (the crate that owns
dataset-adjacent metadata, ADR 0016), as **plain-serde types with no core
dependencies**:

- `DeviceSpec { version, cameras: Vec<CameraDeviceSpec>, rig: Option<RigLayoutSpec>, description }`
- `CameraDeviceSpec { id, focal_mm, pixel_pitch_um, resolution_px: [u32;2], principal_point_px: Option<[f64;2]>, scheimpflug: Option<ScheimpflugMountSpec> }`
- `ScheimpflugMountSpec { tilt_x_deg, tilt_y_deg }` — `None` means frontal sensor
- `RigLayoutSpec { cameras: Vec<CameraMountSpec>, handeye: Option<HandeyeMountSpec>, laser_planes: Option<Vec<LaserPlaneSpec>> }`
- `CameraMountSpec { id, rig_se3_cam: NominalPoseSpec }`
- `HandeyeMountSpec` — a `mode`-tagged enum: `EyeInHand { gripper_se3_rig }` |
  `EyeToHand { rig_se3_base }`, so the frame meaning of the nominal pose is
  fixed by construction (illegal states unrepresentable)
- `NominalPoseSpec { rpy_deg: [f64;3], translation_mm: [f64;3] }`
- `LaserPlaneSpec { camera_id, normal: [f64;3], distance_mm }` — the plane
  `{p : n·p = d}` in the **rig** frame, keyed to its paired camera

**Units are datasheet-natural and encoded in field names** (`_mm`, `_um`,
`_px`, `_deg`): a `DeviceSpec` is a transcription of a datasheet and a
mechanical drawing, and transcription must not require unit conversion —
conversion happens exactly once, in the derivation layer (deg→rad, mm+µm→px,
and translations mm→m: the pipeline's world unit is **metres** — target 3D
points are built as `mm / 1000`, laser errors and plane distances are metres
throughout).

**Frames follow ADR 0009 naming.** The spec stores `rig_se3_cam` — the camera
mount pose *in the rig frame*, which is what a mechanical drawing states;
derivation inverts it into the `cam_se3_rig` (`T_C_R`) the manual-init surface
expects. Rotation convention for `rpy_deg`: fixed-axes XYZ,
`R = Rz(yaw)·Ry(pitch)·Rx(roll)` (nalgebra `from_euler_angles`).

**Placement and privacy.** The sidecar is `spec.json` next to the dataset
manifest (`DEVICE_SPEC_FILENAME`). For the private datasets it lives at
`privatedata/<dataset>/spec.json` and stays **gitignored** — the repo is
public and the hardware specs are as private as the datasets themselves
(consistent with `registry/private.json`). Committed fixtures use synthetic
example specs.

**Validation is fail-fast and typed** (ADR 0019): `DeviceSpec::from_path`
loads, then `validate()` rejects empty/duplicate camera ids, non-positive
focal/pitch/resolution, non-unit laser normals, and unsupported versions;
`deny_unknown_fields` turns typos in hand-authored JSON into load errors.
`version` (default 1) is bumped on breaking schema revisions, like
`DatasetSpec`.

### Derivation (`vision_calibration_pipeline::device_seed`)

Pipeline already depends on dataset, so this adds no dependency edge. The
functions map spec → existing manual-init types and return typed
`DeviceSeedError`s (unknown camera id, missing layout/hand-eye/mount):

- `scheimpflug_seed(spec, camera_id) -> ScheimpflugManualInit` — intrinsics
  (`fx = fy = focal_mm·1000/pixel_pitch_um`, principal point defaulting to the
  resolution center, `skew = 0`) + sensor tilt (deg→rad); distortion and poses
  stay `None` (auto), per ADR 0022.
- `rig_intrinsics_seed(spec, camera_ids) -> RigHandeyeIntrinsicsManualInit` —
  per-camera vectors ordered by the **caller's** id list, so alignment with
  the dataset's camera order is explicit, never positional guessing.
- `nominal_cam_se3_rig(spec, camera_ids) -> Vec<Iso3>` — deliberately *not* a
  `RigHandeyeRigManualInit`: ADR 0011 couples `cam_se3_rig` with per-view
  `rig_se3_target` (both-or-neither), and target poses are data-dependent.
  S3 combines these nominals with per-view estimates.
- `handeye_seed(spec) -> RigHandeyeHandeyeManualInit` — the mode-tagged mount
  maps onto the mode-dependent `handeye` field; `mode_target_pose` stays
  `None`.

Everything is re-exported through the facade so examples, bench, and the app
consume it facade-only.

### Rejected alternatives

- **Core types (`FxFyCxCySkew`, `Iso3`) in the schema** — couples the wire
  format to internal refactors and adds a core dep to the dataset crate.
- **Metres everywhere** (like `TargetSpec::marker_outer_radius_m`) — device
  datasheets quote mm/µm; forcing authors to convert invites transcription
  errors, which defeats the point of a spec layer.
- **Positional camera lists** — silent misalignment; id-keying makes mismatch
  a typed error.
- **`sensor_size_mm` field** — derivable (`pitch × resolution`), would create
  a second source of truth.
- **Metric-anchor field now** — reserved for Q5 (known target dimension /
  baseline prior pins the rtv3d absolute scale); added when it has a consumer.
- **Multiple pose representations (quaternion variant)** — one representation
  suffices for nominal mounts; additive schema change if ever needed.

## Consequences

- S2 replaces the hand-coded constants in `rtv3d_ref_intrinsics` /
  `rtv3d_ringgrid_intrinsics` with `spec.json` loading; the
  `RTV3D_*_FOCAL`/`TILT` env knobs (including the ringgrid focal sweep) are
  deleted. Gate unchanged: all cameras ≤ 0.5 px, now with zero env vars.
- S3 wires `nominal_cam_se3_rig` + `handeye_seed` into the rig examples; S4's
  acceptance registry references spec files per dataset.
- Q6's convergence-basin study quantifies how much spec error the seeded
  route tolerates, closing the loop on "datasheet-grade nominals are enough".
- Schema evolution is versioned and additive-first; breaking revisions bump
  `version` and are release events for the dataset crate.

## References

- ADR 0022 — seeded init is the supported default (the consumer of these seeds)
- ADR 0011 — manual initialization workflow (the target surface)
- ADR 0016 — dataset manifest (the sibling schema and serde conventions)
- ADR 0009 — pose naming (`frame_se3_frame`)
- ADR 0019 — fail-fast on ambiguity (validation posture)
