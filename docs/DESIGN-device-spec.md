# Design: `DeviceSpec` — device specification → initialization seeds (ADR 0023)

Status: implemented in the S1+S2 PR. This is the working design artifact; the
durable decision record is [ADR 0023](adrs/0023-device-spec-seed-derivation.md).

## Problem

Spec-seeded initialization is the official acceptance route (ADR 0022), but
today the seeds are hand-coded constants and env knobs scattered across the
private examples (`RTV3D_REF_FOCAL`, `RTV3D_RINGGRID_FOCAL`, `NOMINAL_TILT_X`,
…). Input: a human-transcribed device datasheet + mechanical drawing (lens
focal, pixel pitch, sensor resolution, Scheimpflug mount tilt, nominal rig
layout) as a sidecar `spec.json` next to a dataset. Output: the existing ADR
0011 manual-init structs (`ScheimpflugManualInit`,
`RigHandeyeIntrinsicsManualInit`, nominal `cam_se3_rig` poses,
`RigHandeyeHandeyeManualInit`), derived in exactly one place.

## Assumptions

- Spec values are datasheet-grade nominals, not calibrated values; the seeded
  optimize step (ADR 0022) tolerates focal error within its `[0.75, 1.5]×`
  bound and tilt within ±0.10 rad. Q6 quantifies the basin.
- Camera ids in the spec match the dataset's camera identifiers
  (`CameraSource::id` convention, e.g. `"cam0"`).
- `resolution_px` is the resolution of the images actually calibrated (for
  rtv3d: the 720×540 tile, not the full sensor).
- World/translation units are millimetres (board cell sizes are given in mm
  throughout the pipeline); angles in the spec are degrees (datasheet units),
  radians internally.
- JSON input (serde_json rejects NaN/Inf literals, so finiteness comes free).

## Failure Modes

All prevented by `DeviceSpec::validate()` (called by `from_path`) or returned
as typed errors from derivation fns — never panics:

- empty `cameras`, duplicate camera ids → `DeviceSpecError` at load/validate
- `focal_mm ≤ 0`, `pixel_pitch_um ≤ 0`, zero `resolution_px` → load/validate
- non-unit laser-plane normal (‖n‖ − 1 > 1e-6) → load/validate
- unsupported `version` (> current) → load/validate
- unknown JSON field → serde `deny_unknown_fields` (catches typos in
  hand-authored files)
- derivation asked for an id not in the spec → `DeviceSeedError::UnknownCameraId`
- derivation asked for rig layout / hand-eye the spec doesn't carry →
  `DeviceSeedError::{MissingRigLayout, MissingHandeye, MissingCameraMount}`

## Domain Types (`vision-calibration-dataset/src/device_spec.rs`)

Plain-serde types only — **no core types in the schema** (keeps the dataset
crate dependency-free and the wire format stable). Datasheet-natural units
with mandatory unit suffixes in field names; conversion happens exactly once,
in the derivation layer. Follows the crate's serde conventions
(`deny_unknown_fields`, `version` with default fn, snake_case tags,
`skip_serializing_if` on options, schemars behind the feature).

```rust
pub struct DeviceSpec {
    pub version: u32,                       // default 1; reject > CURRENT
    pub cameras: Vec<CameraDeviceSpec>,     // non-empty, unique ids
    pub rig: Option<RigLayoutSpec>,
    pub description: Option<String>,
}
pub struct CameraDeviceSpec {
    pub id: String,                         // matches CameraSource::id
    pub focal_mm: f64,                      // lens datasheet focal length
    pub pixel_pitch_um: f64,                // sensor datasheet pixel pitch
    pub resolution_px: [u32; 2],            // [width, height] as calibrated
    pub principal_point_px: Option<[f64; 2]>, // default: resolution center
    pub scheimpflug: Option<ScheimpflugMountSpec>, // None = frontal sensor
}
pub struct ScheimpflugMountSpec { pub tilt_x_deg: f64, pub tilt_y_deg: f64 } // defaults 0
pub struct RigLayoutSpec {
    pub cameras: Vec<CameraMountSpec>,      // nominal mounts, id-keyed
    pub handeye: Option<HandeyeMountSpec>,
    pub laser_planes: Option<Vec<LaserPlaneSpec>>,
}
pub struct CameraMountSpec { pub id: String, pub rig_se3_cam: NominalPoseSpec }
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum HandeyeMountSpec {                 // illegal states unrepresentable:
    EyeInHand { gripper_se3_rig: NominalPoseSpec },  // frame meaning fixed by mode
    EyeToHand { rig_se3_base: NominalPoseSpec },
}
pub struct NominalPoseSpec { pub rpy_deg: [f64; 3], pub translation_mm: [f64; 3] }
pub struct LaserPlaneSpec {                 // plane {p : n·p = d} in RIG frame
    pub camera_id: String,                  // the paired camera (rtv3d pairs)
    pub normal: [f64; 3],                   // unit, rig frame
    pub distance_mm: f64,
}
```

Design choices:

- **Id-keyed cameras** (not positional): silent misalignment with the dataset
  manifest becomes a typed error instead of a wrong seed.
- **`rig_se3_cam` in the spec** (camera mount pose in rig frame — what a
  mechanical drawing states); derivation inverts to the `cam_se3_rig` the
  manual-init surface wants. Both directions explicitly named (ADR 0009).
- **One pose representation**: roll-pitch-yaw degrees, fixed-axes XYZ
  (`R = Rz(yaw)·Ry(pitch)·Rx(roll)`, nalgebra `from_euler_angles`) +
  translation mm. Quaternions can be added as an additive schema change if a
  real need appears (KISS).
- **No `sensor_size_mm` field**: it is `pixel_pitch × resolution` (DRY).
- **No metric-anchor field yet**: reserved for Q5 (YAGNI now); additive later.

## API

```rust
// vision-calibration-dataset
pub const DEVICE_SPEC_FILENAME: &str = "spec.json";  // sidecar next to manifest
impl DeviceSpec {
    pub fn from_path(path: &Path) -> Result<Self, DeviceSpecError>; // load + validate
    pub fn validate(&self) -> Result<(), DeviceSpecError>;
    pub fn camera(&self, id: &str) -> Option<&CameraDeviceSpec>;
}

// vision-calibration-pipeline::device_seed  (new module; pipeline already deps dataset)
pub fn scheimpflug_seed(spec: &DeviceSpec, camera_id: &str)
    -> Result<ScheimpflugManualInit, DeviceSeedError>;      // intrinsics+sensor Some, rest None
pub fn rig_intrinsics_seed(spec: &DeviceSpec, camera_ids: &[&str])
    -> Result<RigHandeyeIntrinsicsManualInit, DeviceSeedError>;
pub fn nominal_cam_se3_rig(spec: &DeviceSpec, camera_ids: &[&str])
    -> Result<Vec<Iso3>, DeviceSeedError>;                  // building block for S3
pub fn handeye_seed(spec: &DeviceSpec)
    -> Result<RigHandeyeHandeyeManualInit, DeviceSeedError>; // handeye Some, target None
```

- `f_px = focal_mm * 1000 / pixel_pitch_um`; principal point defaults to
  `resolution/2`; `fx = fy`, `skew = 0`.
- `nominal_cam_se3_rig` deliberately does **not** return a
  `RigHandeyeRigManualInit`: that struct couples `cam_se3_rig` with per-view
  `rig_se3_target` (both-or-neither, ADR 0011), and target poses are
  data-dependent — S3 combines the nominal poses with per-view estimates.
- `RigExtrinsics`' twin manual-init types get derivation fns when a consumer
  exists (currently none) — not speculatively.
- All fns are cheap setup-path code: allocation-per-call, no genericity
  (`f64` only), no buffers.

## Error Model

```rust
// dataset crate
pub enum DeviceSpecError {
    Io(std::io::Error), Json(serde_json::Error),
    UnsupportedVersion { got: u32, max: u32 },
    NoCameras, DuplicateCameraId { id: String },
    InvalidCamera { id: String, field: &'static str },   // focal/pitch/resolution
    NonUnitLaserNormal { camera_id: String, norm: f64 },
}
// pipeline
pub enum DeviceSeedError {
    UnknownCameraId { id: String },
    MissingRigLayout, MissingHandeye,
    MissingCameraMount { id: String },
}
```

## Placement

- Schema + load/validate: `vision-calibration-dataset/src/device_spec.rs`
  (crate owns dataset-adjacent metadata, ADR 0016). No new deps.
- Derivation: `vision-calibration-pipeline/src/device_seed.rs` (needs core's
  `FxFyCxCySkew`/`ScheimpflugParams`/`Iso3` and pipeline's manual-init types;
  pipeline already depends on dataset — no new graph edge). Re-exported
  through the facade so examples/app stay facade-only.
- Private `spec.json` files: `privatedata/<dataset>/spec.json`, gitignored
  (public repo; consistent with `registry/private.json`).

## Test Plan

- **dataset**: JSON roundtrip of a full spec (rig + hand-eye enum + laser
  plane) and a minimal spec (defaults); `deny_unknown_fields` rejection;
  version default + `UnsupportedVersion`; each `validate()` failure mode.
- **pipeline `device_seed`**: `f_px` formula (16 mm / 4.8 µm → 3333.33…px;
  8 mm / 7 µm → 1142.857 px); principal-point center fallback; tilt deg→rad
  (−5° → −0.08727 rad); `rig_se3_cam` inversion round-trip on a hand-computed
  pose; rpy convention check (rpy = [90,0,0] maps ŷ→ẑ); every error variant.
- **Integration (S2 gate)**: `rtv3d_ref_intrinsics` + `rtv3d_ringgrid_intrinsics`
  run end-to-end from `spec.json` with no env vars, all cameras ≤ 0.5 px.

## Performance Notes

Setup-path code that runs once per calibration; no benchmarks warranted.

## Next Steps

1. Implement the design above (S1), wire the two intrinsics examples (S2)
2. `/algo-review` — verify correctness, robustness, test adequacy
3. `/calibration-review` — pose-convention and units checks on the derivation
