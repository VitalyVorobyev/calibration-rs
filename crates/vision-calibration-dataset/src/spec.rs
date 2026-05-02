//! On-disk manifest types.
//!
//! All types here derive `Serialize + Deserialize` (always) and
//! `JsonSchema` (under the `schemars` feature). The TOML/JSON wire
//! format is the public contract; the corresponding Rust types may
//! grow new variants/fields pre-1.0 but additions must remain
//! backwards-compatible at the serde layer.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[cfg(feature = "schemars")]
use schemars::JsonSchema;

// ─────────────────────────────────────────────────────────────────────────────
// Top-level manifest
// ─────────────────────────────────────────────────────────────────────────────

/// Top-level dataset manifest. Authoritative description of where
/// the user's calibration data lives and how to interpret it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct DatasetSpec {
    /// Format-version sentinel. Always `1` for now; bumped on
    /// breaking schema revisions.
    #[serde(default = "default_version")]
    pub version: u32,

    /// Per-camera image sources. Length must match the topology's
    /// expected camera count (e.g. `Topology::Mono` requires exactly
    /// one entry).
    pub cameras: Vec<CameraSource>,

    /// Calibration-target specification. Tagged enum on `kind`.
    pub target: TargetSpec,

    /// Robot-pose source for hand-eye topologies. `None` for
    /// pure-intrinsics calibrations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub robot_poses: Option<RobotPoseSource>,

    /// Calibration topology — selects which problem type the runner
    /// will dispatch to.
    pub topology: Topology,

    /// How per-camera images are paired across views (and optionally
    /// with robot poses). `None` is permitted only for `Topology::Mono`
    /// where pose-pairing is irrelevant.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pose_pairing: Option<PosePairing>,

    /// Frame conventions for robot poses. **No defaults** — every
    /// component is required when `robot_poses` is set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pose_convention: Option<PoseConvention>,

    /// Field paths the AI manifest generator could not determine.
    /// Validated to be empty before the runner accepts the manifest.
    /// Populated as e.g. `["pose_convention.transform"]`.
    #[serde(default, skip_serializing_if = "Vec::is_empty", rename = "_unresolved")]
    pub unresolved: Vec<String>,

    /// Free-form human-readable description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

fn default_version() -> u32 {
    1
}

// ─────────────────────────────────────────────────────────────────────────────
// Camera source
// ─────────────────────────────────────────────────────────────────────────────

/// Per-camera image source.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct CameraSource {
    /// Stable camera identifier (e.g. `"cam0"`). Used as a key in
    /// pose pairings and downstream exports.
    pub id: String,

    /// Glob pattern (e.g. `"cam0/*.png"`) or explicit list of image
    /// paths, relative to the manifest's directory unless absolute.
    pub images: ImagePattern,

    /// Optional rectangular ROI in source-image pixels — `[x, y, w, h]`.
    /// When set, only the cropped region is fed to the detector and
    /// stored in the export's image manifest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub roi_xywh: Option<[u32; 4]>,
}

/// How image paths for a camera are listed.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum ImagePattern {
    /// Single shell-glob pattern, evaluated relative to the manifest
    /// directory. Example: `"cam0/*.png"`.
    Glob {
        /// The glob pattern.
        pattern: String,
    },
    /// Explicit ordered list of paths.
    List {
        /// Paths relative to the manifest directory unless absolute.
        paths: Vec<PathBuf>,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Target specification
// ─────────────────────────────────────────────────────────────────────────────

/// Calibration-target metadata. Tagged on `kind`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum TargetSpec {
    /// Standard checkerboard. Counts are interior corners
    /// (i.e. `9x6` ⇒ a 10×7 black-and-white square grid).
    Chessboard {
        /// Number of interior corners along the rows axis.
        rows: u32,
        /// Number of interior corners along the cols axis.
        cols: u32,
        /// Edge length of one square in metres.
        square_size_m: f64,
    },
    /// ChArUco board. Marker IDs and dictionary live here so detection
    /// is fully deterministic from the manifest.
    Charuco {
        /// Number of squares along the rows axis (full grid, not
        /// interior corners).
        rows: u32,
        /// Number of squares along the cols axis.
        cols: u32,
        /// Edge length of one square in metres.
        square_size_m: f64,
        /// Edge length of an embedded marker in metres.
        marker_size_m: f64,
        /// ArUco dictionary identifier (e.g. `"DICT_4X4_50"`). The
        /// detector validates this against the supported set.
        dictionary: String,
    },
    /// Puzzleboard target (calib-targets crate). Layout is fully
    /// described by the named variant in calib-targets; the manifest
    /// just carries the discriminator.
    Puzzleboard {
        /// Named layout (e.g. `"puzzle_130x130"`).
        layout: String,
        /// Edge length of one cell in metres.
        cell_size_m: f64,
    },
    /// Ringgrid target (ringgrid crate).
    Ringgrid {
        /// Number of rings along the rows axis.
        rows: u32,
        /// Number of rings along the cols axis.
        cols: u32,
        /// Centre-to-centre spacing of adjacent rings in metres.
        spacing_m: f64,
        /// Inner ring radius in metres.
        inner_radius_m: f64,
        /// Outer ring radius in metres.
        outer_radius_m: f64,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Robot poses
// ─────────────────────────────────────────────────────────────────────────────

/// Where and how robot poses are stored on disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct RobotPoseSource {
    /// Path to the pose file, relative to the manifest directory
    /// unless absolute.
    pub path: PathBuf,
    /// File-format selector.
    pub format: RobotPoseFormat,
    /// Column / field mapping from the user's file to the canonical
    /// fields the converter consumes.
    pub columns: PoseColumnMap,
}

/// Robot-pose file format.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum RobotPoseFormat {
    /// Comma-separated values with a header row.
    Csv,
    /// JSON array of objects.
    Json,
    /// JSONL (one JSON object per line).
    Jsonl,
}

/// Mapping from user-defined column / field names to the canonical
/// pose fields. All seven of `tx, ty, tz, qx, qy, qz, qw` are required
/// when `rotation_format` is a quaternion; the runner rejects partial
/// mappings as a fail-fast event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct PoseColumnMap {
    /// Optional view-id column (string or integer). When absent, the
    /// pose row index is used as the implicit pose id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pose_id: Option<String>,
    /// Translation x column.
    pub tx: String,
    /// Translation y column.
    pub ty: String,
    /// Translation z column.
    pub tz: String,
    /// Rotation columns. Length and meaning depend on the
    /// `pose_convention.rotation_format`.
    pub rotation: Vec<String>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Topology
// ─────────────────────────────────────────────────────────────────────────────

/// Calibration topology — selects the problem-type dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum Topology {
    /// Single camera, planar target — `PlanarIntrinsicsProblem`.
    PlanarIntrinsics,
    /// Single camera with Scheimpflug tilt — `ScheimpflugIntrinsicsProblem`.
    ScheimpflugIntrinsics,
    /// Single camera mounted on a robot — `SingleCamHandeyeProblem`.
    SingleCamHandeye,
    /// Camera + laser plane — `LaserlineDeviceProblem`.
    LaserlineDevice,
    /// Multi-camera rig, target only — `RigExtrinsicsProblem`.
    RigExtrinsics,
    /// Multi-camera rig + robot — `RigHandeyeProblem`.
    RigHandeye,
    /// Multi-camera rig + laser planes — `RigLaserlineDeviceProblem`.
    RigLaserlineDevice,
}

// ─────────────────────────────────────────────────────────────────────────────
// Pose pairing
// ─────────────────────────────────────────────────────────────────────────────

/// How per-camera image lists align across views and (optionally)
/// align with robot poses.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum PosePairing {
    /// All cameras have the same number of images, paired by index
    /// (image[i] from cam0 ↔ image[i] from cam1 ↔ pose[i]).
    ByIndex,
    /// A regex applied to filenames extracts a shared "view token"
    /// (e.g. timestamp or pose id). All cameras and the pose file
    /// must produce the same set of tokens.
    SharedFilenameToken {
        /// Regex with at least one named capture group.
        regex: String,
        /// Name of the capture group whose value identifies the view.
        group: String,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Pose conventions
// ─────────────────────────────────────────────────────────────────────────────

/// Frame conventions for the robot pose stream.
///
/// All three components are required (no defaults). Each is a closed
/// enum so the runtime supports a finite, validated set of conventions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct PoseConvention {
    /// Which transform the pose row encodes.
    pub transform: TransformConvention,
    /// How rotation is parameterised in the file.
    pub rotation_format: RotationFormat,
    /// Translation units in the file.
    pub translation_units: TranslationUnits,
}

/// Which transform a pose row encodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum TransformConvention {
    /// `T_base_tcp` — gripper pose in the robot base frame (KUKA, UR
    /// default).
    TBaseTcp,
    /// `T_tcp_base` — base pose in the gripper frame (some ABB exports).
    TTcpBase,
    /// `T_world_tcp` — gripper pose in a fixed world frame (less
    /// common; some lab-frame setups).
    TWorldTcp,
    /// `T_tcp_world` — world pose in the gripper frame.
    TTcpWorld,
}

/// Rotation parameterisation in the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum RotationFormat {
    /// Hamilton quaternion as `[qx, qy, qz, qw]`.
    QuatXyzw,
    /// Hamilton quaternion as `[qw, qx, qy, qz]`.
    QuatWxyz,
    /// Intrinsic XYZ Euler angles in degrees.
    EulerXyzDeg,
    /// Intrinsic XYZ Euler angles in radians.
    EulerXyzRad,
    /// Intrinsic ZYX Euler angles in degrees.
    EulerZyxDeg,
    /// Intrinsic ZYX Euler angles in radians.
    EulerZyxRad,
    /// Axis-angle vector (length encodes magnitude, radians).
    AxisAngleRad,
    /// Row-major 4×4 homogeneous matrix flattened across 16 columns.
    Matrix4x4RowMajor,
}

/// Translation units in the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum TranslationUnits {
    /// Metres (SI, preferred).
    M,
    /// Millimetres (some ABB / Fanuc exports).
    Mm,
}
