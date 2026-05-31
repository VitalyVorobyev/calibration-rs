//! Declarative dataset registry.
//!
//! The registry is the data-driven catalog of benchmark datasets. Each
//! [`BenchEntry`] *wraps* a [`DatasetSpec`] (via [`SpecRef`]) rather than
//! extending it: `DatasetSpec` is `#[serde(deny_unknown_fields)]`, so bench
//! metadata (tiers, camera layout, detector overrides, …) lives on the
//! surrounding entry, never inside the spec.
//!
//! Enums are `#[serde(rename_all = "snake_case")]` and most optional fields
//! carry `#[serde(default)]` so a registry written in TOML stays terse.
//!
//! # No `PartialEq`
//!
//! [`DatasetSpec`] does not implement `PartialEq`, so neither [`SpecRef`] nor
//! [`BenchEntry`]/[`BenchRegistry`] can. The roundtrip tests assert equality by
//! comparing re-serialized JSON instead.

use std::collections::BTreeMap;
use std::path::Path;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use vision_calibration_core::DistortionFixMask;
use vision_calibration_dataset::DatasetSpec;
use vision_calibration_optim::{HandEyeMode, RobustLoss, ScheimpflugFixMask};
use vision_calibration_pipeline::rig_handeye::{RigHandeyeConfig, SensorMode};
use vision_calibration_pipeline::single_cam_handeye::SingleCamHandeyeConfig;

/// Top-level benchmark registry: the full set of datasets the harness knows.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BenchRegistry {
    /// All registered datasets.
    #[serde(default)]
    pub datasets: Vec<BenchEntry>,
}

impl BenchRegistry {
    /// Find the entry with the given `id`, if any.
    pub fn find<'a>(&'a self, id: &str) -> Option<&'a BenchEntry> {
        self.datasets.iter().find(|e| e.id == id)
    }
}

/// Load a [`BenchRegistry`] from a JSON file.
///
/// The registry wire format is **JSON** (not TOML): the workspace `toml` crate
/// is unavailable in this build, and `serde_json` round-trips the registry types
/// (which is what the existing `registry_roundtrips` test exercises).
///
/// # Errors
///
/// Returns an error if the file cannot be read or does not parse as a
/// [`BenchRegistry`].
pub fn load_registry(path: &Path) -> anyhow::Result<BenchRegistry> {
    use anyhow::Context;
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read registry {}", path.display()))?;
    serde_json::from_str(&text)
        .with_context(|| format!("failed to parse registry JSON {}", path.display()))
}

/// A single dataset registered with the benchmark harness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchEntry {
    /// Stable identifier for this dataset.
    pub id: String,
    /// Whether the dataset is publicly committed or a private real-world set.
    pub visibility: Visibility,
    /// Tiers this dataset participates in.
    pub tiers: Vec<Tier>,
    /// Problem kind this dataset targets.
    pub problem: ProblemKind,
    /// Filesystem root for the dataset's images / artifacts.
    pub data_root: PathBuf,
    /// The dataset specification (inline or referenced by path).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spec: Option<SpecRef>,
    /// Detector configuration override, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detector: Option<DetectorOverride>,
    /// Laser configuration, for laserline problems.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub laser: Option<LaserConfig>,
    /// Calibration board geometry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub board: Option<BoardGeometry>,
    /// Per-camera layout (folders, globs, tiles).
    #[serde(default)]
    pub cameras: Vec<CameraLayout>,
    /// Source of robot poses, for hand-eye problems.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub robot_poses: Option<PoseSource>,
    /// Path to a frozen prior calibration export, for delta-to-prior metrics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prior_export: Option<PathBuf>,
    /// Path to a Tier-A frozen fixture for this dataset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fixture: Option<PathBuf>,
    /// Manual initialization seed, if the problem needs one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<ManualInitSeed>,
    /// Single-camera hand-eye configuration overrides.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub single_cam_handeye: Option<SingleCamHandeyeOverride>,
    /// Rig hand-eye configuration overrides.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rig_handeye: Option<RigHandeyeOverride>,
    /// Stability-sampling configuration.
    #[serde(default)]
    pub stability: StabilityCfg,
    /// Cross-validation configuration.
    #[serde(default)]
    pub crossval: CrossvalCfg,
    /// Free-form notes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

/// Whether a dataset is publicly committed or private.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Visibility {
    /// Committed fixtures, runnable in CI.
    Public,
    /// Private real-world datasets, run locally / nightly.
    Private,
}

/// Benchmark tier a dataset participates in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    /// Tier-A: math/serde only, on frozen fixtures.
    A,
    /// Tier-B: full pipeline with detector + image I/O.
    B,
}

/// Problem kind a dataset targets.
///
/// Mirrors [`vision_calibration_dataset::Topology`] one-for-one; the
/// `problem_kind_mirrors_topology` test is a compile-time guard against drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProblemKind {
    /// Single-camera planar intrinsics (Zhang).
    PlanarIntrinsics,
    /// Scheimpflug (tilted-sensor) intrinsics.
    ScheimpflugIntrinsics,
    /// Single-camera hand-eye.
    SingleCamHandeye,
    /// Multi-camera rig extrinsics.
    RigExtrinsics,
    /// Multi-camera rig hand-eye.
    RigHandeye,
    /// Single laserline device.
    LaserlineDevice,
    /// Multi-camera rig laserline device.
    RigLaserlineDevice,
}

/// Reference to a dataset specification: inline or by path.
///
/// Inline carries a full [`DatasetSpec`] embedded in the registry; `Path` points
/// at a separate manifest the runner loads.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpecRef {
    /// The spec embedded directly in the registry.
    Inline(Box<DatasetSpec>),
    /// The spec stored at a separate path.
    Path {
        /// Path to the spec file.
        path: PathBuf,
    },
}

/// Layout of a single camera within a dataset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraLayout {
    /// Camera identifier.
    pub id: String,
    /// Folder (relative to `data_root`) holding this camera's images.
    pub folder: String,
    /// Glob matching this camera's image filenames.
    pub filename_glob: String,
    /// Optional ROI tile `[x, y, w, h]` within a tiled multi-camera image.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tile: Option<[u32; 4]>,
    /// Expected image size `[width, height]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_size: Option<[u32; 2]>,
    /// Whether the camera produces color images.
    #[serde(default)]
    pub color: bool,
}

/// Detector configuration override.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct DetectorOverride {
    /// ChESS corner extractor options shared by chessboard-like detectors.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chess_corners: Option<ChessCornersDetectorOverride>,
    /// Preserve older free-form detector keys that the benchmark does not
    /// interpret yet.
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// ChESS corner extractor overrides.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ChessCornersDetectorOverride {
    /// Acceptance threshold mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold_mode: Option<BenchChessThresholdMode>,
    /// Acceptance threshold value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold_value: Option<f32>,
}

/// Registry ChESS threshold mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchChessThresholdMode {
    /// Threshold in native ChESS response units.
    Absolute,
    /// Threshold as a fraction of the image maximum response.
    Relative,
}

/// Laser configuration for laserline problems.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LaserConfig {
    /// Laser extraction method.
    pub method: String,
    /// Method parameters (free-form for now).
    #[serde(default)]
    pub params: serde_json::Value,
    /// Residual type used for laser metrics.
    pub residual_type: String,
}

/// Calibration board geometry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoardGeometry {
    /// Number of rows of features.
    pub rows: usize,
    /// Number of columns of features.
    pub cols: usize,
    /// Cell size in metres.
    pub cell_size_m: f64,
    /// Marker dictionary, for ChArUco-style boards.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dictionary: Option<String>,
    /// Board layout descriptor (e.g. `"checkerboard"`, `"charuco"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layout: Option<String>,
    /// Marker-to-cell size ratio for ChArUco boards (e.g. `0.75`).
    /// Defaults to `0.75` when absent (the OpenCV ChArUco default).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub marker_size_rel: Option<f32>,
    /// Require checkerboard detections to match the declared rows/cols exactly.
    ///
    /// This is important for hand-eye datasets where partial, locally indexed
    /// checkerboard detections move the target coordinate frame between views.
    #[serde(default, skip_serializing_if = "is_false")]
    pub strict_grid: bool,
}

fn is_false(value: &bool) -> bool {
    !*value
}

/// Source of robot poses for hand-eye problems.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoseSource {
    /// Path to the pose file.
    pub path: PathBuf,
    /// File format. Supported by the Tier-B runners:
    /// - `"rowmajor4x4"` — one row-major 4×4 matrix per line (16 values).
    /// - `"counted4x4"` — a leading integer count, then that many 4×4 matrices
    ///   as whitespace-separated values (4 lines × 4 values each).
    pub format: String,
    /// Pose convention (e.g. `"base_se3_gripper"`).
    pub convention: String,
    /// Translation units in the file. `"mm"` scales translations by `1e-3` to
    /// metres; `"m"` or omitted leaves them unscaled. Rotation is unaffected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub units: Option<String>,
}

/// Manual initialization seed.
///
/// Placeholder (`serde_json::Value`); the real, typed shape arrives in Phase 2
/// alongside manual-init support.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ManualInitSeed(pub serde_json::Value);

/// Shared hand-eye bundle-adjustment overrides.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct HandeyeBaOverride {
    /// Enable/disable per-view robot pose refinement.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refine_robot_poses: Option<bool>,
    /// Enable/disable rig-extrinsic refinement in final rig hand-eye BA.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refine_cam_se3_rig_in_handeye_ba: Option<bool>,
    /// Enable/disable Scheimpflug tilt refinement in final rig hand-eye BA.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refine_scheimpflug_in_handeye_ba: Option<bool>,
    /// Robot rotation prior sigma in radians.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub robot_rot_sigma: Option<f64>,
    /// Robot translation prior sigma in metres.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub robot_trans_sigma: Option<f64>,
}

/// Single-camera hand-eye config overrides.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SingleCamHandeyeOverride {
    /// Hand-eye mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handeye_mode: Option<BenchHandEyeMode>,
    /// Solver max iterations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_iters: Option<usize>,
    /// Robust loss.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub robust_loss: Option<BenchRobustLoss>,
    /// Bundle-adjustment robot-pose settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handeye_ba: Option<HandeyeBaOverride>,
}

/// Rig hand-eye config overrides.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct RigHandeyeOverride {
    /// Sensor flavour.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sensor: Option<BenchSensorMode>,
    /// Hand-eye mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handeye_mode: Option<BenchHandEyeMode>,
    /// Solver max iterations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_iters: Option<usize>,
    /// Robust loss.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub robust_loss: Option<BenchRobustLoss>,
    /// Re-refine intrinsics in rig BA.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refine_intrinsics_in_rig_ba: Option<bool>,
    /// Bundle-adjustment robot-pose settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handeye_ba: Option<HandeyeBaOverride>,
}

/// Registry hand-eye mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchHandEyeMode {
    /// Camera mounted on the gripper.
    EyeInHand,
    /// Camera fixed in workspace / rig sees robot-held target.
    EyeToHand,
}

impl From<BenchHandEyeMode> for HandEyeMode {
    fn from(value: BenchHandEyeMode) -> Self {
        match value {
            BenchHandEyeMode::EyeInHand => HandEyeMode::EyeInHand,
            BenchHandEyeMode::EyeToHand => HandEyeMode::EyeToHand,
        }
    }
}

/// Registry robust-loss override.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BenchRobustLoss {
    /// Plain squared residuals.
    None,
    /// Huber loss.
    Huber {
        /// Scale parameter.
        scale: f64,
    },
    /// Cauchy loss.
    Cauchy {
        /// Scale parameter.
        scale: f64,
    },
    /// Arctangent loss.
    Arctan {
        /// Scale parameter.
        scale: f64,
    },
}

impl From<BenchRobustLoss> for RobustLoss {
    fn from(value: BenchRobustLoss) -> Self {
        match value {
            BenchRobustLoss::None => RobustLoss::None,
            BenchRobustLoss::Huber { scale } => RobustLoss::Huber { scale },
            BenchRobustLoss::Cauchy { scale } => RobustLoss::Cauchy { scale },
            BenchRobustLoss::Arctan { scale } => RobustLoss::Arctan { scale },
        }
    }
}

/// Registry sensor mode.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BenchSensorMode {
    /// Pinhole + Brown-Conrady.
    Pinhole,
    /// Scheimpflug tilted sensor.
    Scheimpflug {
        /// Initial tilt around X in radians.
        #[serde(default)]
        init_tilt_x: f64,
        /// Initial tilt around Y in radians.
        #[serde(default)]
        init_tilt_y: f64,
        /// Scheimpflug fix mask during per-camera intrinsics.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        fix_scheimpflug_in_intrinsics: Option<BenchScheimpflugFixMask>,
        /// Distortion mask during per-camera BA.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        distortion_mask_in_percam_ba: Option<BenchDistortionFixMask>,
        /// Re-refine Scheimpflug parameters in rig BA.
        #[serde(default)]
        refine_scheimpflug_in_rig_ba: bool,
    },
}

impl From<BenchSensorMode> for SensorMode {
    fn from(value: BenchSensorMode) -> Self {
        match value {
            BenchSensorMode::Pinhole => SensorMode::Pinhole,
            BenchSensorMode::Scheimpflug {
                init_tilt_x,
                init_tilt_y,
                fix_scheimpflug_in_intrinsics,
                distortion_mask_in_percam_ba,
                refine_scheimpflug_in_rig_ba,
            } => SensorMode::Scheimpflug {
                init_tilt_x,
                init_tilt_y,
                fix_scheimpflug_in_intrinsics: fix_scheimpflug_in_intrinsics
                    .map(Into::into)
                    .unwrap_or_default(),
                distortion_mask_in_percam_ba: distortion_mask_in_percam_ba
                    .map(Into::into)
                    .unwrap_or_default(),
                refine_scheimpflug_in_rig_ba,
            },
        }
    }
}

/// Registry Scheimpflug fix mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct BenchScheimpflugFixMask {
    /// Keep `tilt_x` fixed.
    pub tilt_x: bool,
    /// Keep `tilt_y` fixed.
    pub tilt_y: bool,
}

impl From<BenchScheimpflugFixMask> for ScheimpflugFixMask {
    fn from(value: BenchScheimpflugFixMask) -> Self {
        Self {
            tilt_x: value.tilt_x,
            tilt_y: value.tilt_y,
        }
    }
}

/// Registry distortion fix mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct BenchDistortionFixMask {
    /// Fix k1.
    pub k1: bool,
    /// Fix k2.
    pub k2: bool,
    /// Fix k3.
    pub k3: bool,
    /// Fix p1.
    pub p1: bool,
    /// Fix p2.
    pub p2: bool,
}

impl From<BenchDistortionFixMask> for DistortionFixMask {
    fn from(value: BenchDistortionFixMask) -> Self {
        Self {
            k1: value.k1,
            k2: value.k2,
            k3: value.k3,
            p1: value.p1,
            p2: value.p2,
        }
    }
}

impl SingleCamHandeyeOverride {
    /// Apply overrides to a default single-camera hand-eye config.
    pub fn apply_to(&self, config: &mut SingleCamHandeyeConfig) {
        if let Some(mode) = self.handeye_mode {
            config.handeye_mode = mode.into();
        }
        if let Some(max_iters) = self.max_iters {
            config.max_iters = max_iters;
        }
        if let Some(loss) = self.robust_loss {
            config.robust_loss = loss.into();
        }
        if let Some(ba) = &self.handeye_ba {
            apply_single_ba(ba, config);
        }
    }
}

impl RigHandeyeOverride {
    /// Apply overrides to a default rig hand-eye config.
    pub fn apply_to(&self, config: &mut RigHandeyeConfig) {
        if let Some(sensor) = self.sensor {
            config.sensor = sensor.into();
        }
        if let Some(mode) = self.handeye_mode {
            config.handeye_init.handeye_mode = mode.into();
        }
        if let Some(max_iters) = self.max_iters {
            config.solver.max_iters = max_iters;
        }
        if let Some(loss) = self.robust_loss {
            config.solver.robust_loss = loss.into();
        }
        if let Some(refine) = self.refine_intrinsics_in_rig_ba {
            config.rig.refine_intrinsics_in_rig_ba = refine;
        }
        if let Some(ba) = &self.handeye_ba {
            apply_rig_ba(ba, config);
        }
    }
}

fn apply_single_ba(ba: &HandeyeBaOverride, config: &mut SingleCamHandeyeConfig) {
    if let Some(refine) = ba.refine_robot_poses {
        config.refine_robot_poses = refine;
    }
    if let Some(sigma) = ba.robot_rot_sigma {
        config.robot_rot_sigma = sigma;
    }
    if let Some(sigma) = ba.robot_trans_sigma {
        config.robot_trans_sigma = sigma;
    }
}

fn apply_rig_ba(ba: &HandeyeBaOverride, config: &mut RigHandeyeConfig) {
    if let Some(refine) = ba.refine_robot_poses {
        config.handeye_ba.refine_robot_poses = refine;
    }
    if let Some(refine) = ba.refine_cam_se3_rig_in_handeye_ba {
        config.handeye_ba.refine_cam_se3_rig_in_handeye_ba = refine;
    }
    if let Some(refine) = ba.refine_scheimpflug_in_handeye_ba {
        config.handeye_ba.refine_scheimpflug_in_handeye_ba = refine;
    }
    if let Some(sigma) = ba.robot_rot_sigma {
        config.handeye_ba.robot_rot_sigma = sigma;
    }
    if let Some(sigma) = ba.robot_trans_sigma {
        config.handeye_ba.robot_trans_sigma = sigma;
    }
}

/// Stability-sampling configuration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StabilityCfg {
    /// Number of resampled runs.
    pub n_runs: usize,
    /// Fraction of data used per subset.
    pub subset_frac: f64,
    /// Base seed for resampling.
    pub base_seed: u64,
}

impl Default for StabilityCfg {
    fn default() -> Self {
        Self {
            n_runs: 16,
            subset_frac: 0.7,
            base_seed: 0,
        }
    }
}

/// Cross-validation configuration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CrossvalCfg {
    /// Number of folds.
    pub folds: usize,
    /// Holdout fraction per fold.
    pub holdout_frac: f64,
}

impl Default for CrossvalCfg {
    fn default() -> Self {
        Self {
            folds: 5,
            holdout_frac: 0.2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vision_calibration_dataset::{CameraSource, ImagePattern, TargetSpec, Topology};

    /// Compile-time guard: every [`ProblemKind`] variant has a `Topology`
    /// counterpart with the same name, and vice versa. If this `match` stops
    /// compiling (a variant added/removed on either side), the two enums have
    /// drifted — re-sync them.
    #[allow(dead_code)]
    fn problem_kind_mirrors_topology(t: Topology) -> ProblemKind {
        match t {
            Topology::PlanarIntrinsics => ProblemKind::PlanarIntrinsics,
            Topology::ScheimpflugIntrinsics => ProblemKind::ScheimpflugIntrinsics,
            Topology::SingleCamHandeye => ProblemKind::SingleCamHandeye,
            Topology::RigExtrinsics => ProblemKind::RigExtrinsics,
            Topology::RigHandeye => ProblemKind::RigHandeye,
            Topology::LaserlineDevice => ProblemKind::LaserlineDevice,
            Topology::RigLaserlineDevice => ProblemKind::RigLaserlineDevice,
        }
    }

    fn sample_spec() -> DatasetSpec {
        DatasetSpec {
            version: 1,
            cameras: vec![CameraSource {
                id: "cam0".into(),
                images: ImagePattern::Glob {
                    pattern: "cam0/*.png".into(),
                },
                roi_xywh: None,
            }],
            target: TargetSpec::Chessboard {
                rows: 7,
                cols: 11,
                square_size_m: 0.03,
            },
            robot_poses: None,
            topology: Topology::RigHandeye,
            pose_pairing: None,
            pose_convention: None,
            unresolved: Vec::new(),
            description: Some("130x130 Scheimpflug rig".into()),
        }
    }

    fn sample_entry() -> BenchEntry {
        BenchEntry {
            id: "puzzle_130x130".into(),
            visibility: Visibility::Private,
            tiers: vec![Tier::A, Tier::B],
            problem: ProblemKind::RigHandeye,
            data_root: PathBuf::from("/data/puzzle"),
            spec: Some(SpecRef::Inline(Box::new(sample_spec()))),
            detector: Some(DetectorOverride {
                chess_corners: None,
                extra: BTreeMap::from([("refine".into(), serde_json::json!(true))]),
            }),
            laser: None,
            board: Some(BoardGeometry {
                rows: 13,
                cols: 13,
                cell_size_m: 0.01,
                dictionary: Some("DICT_4X4_50".into()),
                layout: Some("charuco".into()),
                marker_size_rel: None,
                strict_grid: false,
            }),
            cameras: vec![CameraLayout {
                id: "cam0".into(),
                folder: "cam0".into(),
                filename_glob: "*.png".into(),
                tile: Some([0, 0, 1280, 1024]),
                expected_size: Some([1280, 1024]),
                color: false,
            }],
            robot_poses: Some(PoseSource {
                path: PathBuf::from("poses.csv"),
                format: "csv".into(),
                convention: "base_se3_gripper".into(),
                units: None,
            }),
            prior_export: Some(PathBuf::from("prior.json")),
            fixture: Some(PathBuf::from("fixtures/puzzle.json")),
            seed: Some(ManualInitSeed(serde_json::json!({"fx": 1000.0}))),
            single_cam_handeye: None,
            rig_handeye: Some(RigHandeyeOverride {
                sensor: Some(BenchSensorMode::Scheimpflug {
                    init_tilt_x: 0.0,
                    init_tilt_y: 0.0,
                    fix_scheimpflug_in_intrinsics: Some(BenchScheimpflugFixMask {
                        tilt_x: false,
                        tilt_y: true,
                    }),
                    distortion_mask_in_percam_ba: Some(BenchDistortionFixMask {
                        k1: false,
                        k2: true,
                        k3: true,
                        p1: false,
                        p2: false,
                    }),
                    refine_scheimpflug_in_rig_ba: false,
                }),
                handeye_mode: Some(BenchHandEyeMode::EyeToHand),
                max_iters: Some(200),
                robust_loss: Some(BenchRobustLoss::Huber { scale: 1.0 }),
                refine_intrinsics_in_rig_ba: Some(false),
                handeye_ba: Some(HandeyeBaOverride {
                    refine_robot_poses: Some(true),
                    refine_cam_se3_rig_in_handeye_ba: Some(false),
                    refine_scheimpflug_in_handeye_ba: Some(false),
                    robot_rot_sigma: None,
                    robot_trans_sigma: None,
                }),
            }),
            stability: StabilityCfg::default(),
            crossval: CrossvalCfg::default(),
            notes: Some("real-data acceptance".into()),
        }
    }

    #[test]
    fn defaults_match_spec() {
        assert_eq!(
            StabilityCfg::default(),
            StabilityCfg {
                n_runs: 16,
                subset_frac: 0.7,
                base_seed: 0
            }
        );
        assert_eq!(
            CrossvalCfg::default(),
            CrossvalCfg {
                folds: 5,
                holdout_frac: 0.2
            }
        );
    }

    /// `BenchRegistry` cannot derive `PartialEq` (`DatasetSpec` lacks it), so we
    /// assert the roundtrip by comparing re-serialized JSON.
    #[test]
    fn registry_roundtrips() {
        let registry = BenchRegistry {
            datasets: vec![sample_entry()],
        };
        let json = serde_json::to_string(&registry).expect("serialize");
        let back: BenchRegistry = serde_json::from_str(&json).expect("deserialize");
        let json2 = serde_json::to_string(&back).expect("re-serialize");
        assert_eq!(json, json2);
    }

    #[test]
    fn spec_ref_path_variant_roundtrips() {
        let mut entry = sample_entry();
        entry.spec = Some(SpecRef::Path {
            path: PathBuf::from("specs/puzzle.json"),
        });
        let json = serde_json::to_string(&entry).expect("serialize");
        let back: BenchEntry = serde_json::from_str(&json).expect("deserialize");
        let json2 = serde_json::to_string(&back).expect("re-serialize");
        assert_eq!(json, json2);
    }

    #[test]
    fn rig_handeye_override_applies_to_default_config() {
        let entry = sample_entry();
        let override_cfg = entry.rig_handeye.as_ref().expect("sample override");
        let mut cfg = RigHandeyeConfig::default();
        override_cfg.apply_to(&mut cfg);

        assert!(matches!(cfg.sensor, SensorMode::Scheimpflug { .. }));
        assert_eq!(cfg.handeye_init.handeye_mode, HandEyeMode::EyeToHand);
        assert_eq!(cfg.solver.max_iters, 200);
        assert_eq!(cfg.solver.robust_loss, RobustLoss::Huber { scale: 1.0 });
        assert!(cfg.handeye_ba.refine_robot_poses);
        assert!(!cfg.handeye_ba.refine_cam_se3_rig_in_handeye_ba);
        assert!(!cfg.handeye_ba.refine_scheimpflug_in_handeye_ba);
    }

    #[test]
    fn charuco_3536_override_maps_physical_setup() {
        let json = r#"
        {
          "sensor": {
            "kind": "scheimpflug",
            "init_tilt_x": 0.0,
            "init_tilt_y": 0.0,
            "fix_scheimpflug_in_intrinsics": { "tilt_x": false, "tilt_y": false },
            "distortion_mask_in_percam_ba": {
              "k1": false, "k2": false, "k3": false, "p1": true, "p2": true
            },
            "refine_scheimpflug_in_rig_ba": true
          },
          "handeye_mode": "eye_to_hand",
          "refine_intrinsics_in_rig_ba": true,
          "handeye_ba": {
            "refine_robot_poses": true,
            "refine_cam_se3_rig_in_handeye_ba": false,
            "refine_scheimpflug_in_handeye_ba": false
          }
        }
        "#;
        let override_cfg: RigHandeyeOverride = serde_json::from_str(json).expect("override");
        let mut cfg = RigHandeyeConfig::default();
        override_cfg.apply_to(&mut cfg);

        assert_eq!(cfg.handeye_init.handeye_mode, HandEyeMode::EyeToHand);
        assert!(cfg.rig.refine_intrinsics_in_rig_ba);
        assert!(!cfg.handeye_ba.refine_cam_se3_rig_in_handeye_ba);
        assert!(!cfg.handeye_ba.refine_scheimpflug_in_handeye_ba);
        match cfg.sensor {
            SensorMode::Scheimpflug {
                distortion_mask_in_percam_ba,
                refine_scheimpflug_in_rig_ba,
                ..
            } => {
                assert!(distortion_mask_in_percam_ba.p1);
                assert!(distortion_mask_in_percam_ba.p2);
                assert!(refine_scheimpflug_in_rig_ba);
            }
            SensorMode::Pinhole => panic!("expected Scheimpflug sensor"),
            _ => panic!("expected Scheimpflug sensor"),
        }
    }

    #[test]
    fn ds8_registry_uses_known_grid_eye_in_hand_and_gripper_from_base() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("registry/public.json");
        let registry = load_registry(&path).expect("public registry");
        let ds8 = registry
            .datasets
            .iter()
            .find(|entry| entry.id == "ds8")
            .expect("ds8 entry");
        let board = ds8.board.as_ref().expect("ds8 board");
        assert_eq!(board.rows, 10);
        assert_eq!(board.cols, 14);
        assert!((board.cell_size_m - 0.052).abs() < 1.0e-12);
        assert!(board.strict_grid);
        assert_eq!(
            ds8.rig_handeye.as_ref().and_then(|cfg| cfg.handeye_mode),
            None
        );
        assert_eq!(
            ds8.robot_poses.as_ref().map(|src| src.convention.as_str()),
            Some("gripper_se3_base")
        );
    }
}
