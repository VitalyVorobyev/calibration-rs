//! Declarative dataset registry.
//!
//! The registry is the data-driven catalog of benchmark datasets. Each
//! [`BenchEntry`] *wraps* a [`DatasetSpec`] (via [`SpecRef`]) rather than
//! extending it: `DatasetSpec` is `#[serde(deny_unknown_fields)]`, so bench
//! metadata (tiers, camera layout, detector overrides, …) lives on the
//! surrounding entry, never inside the spec.
//!
//! Enums are `#[serde(rename_all = "snake_case")]` and most optional fields
//! carry `#[serde(default)]` so a registry written in TOML stays terse. Every
//! struct rejects unknown fields, so a renamed config key fails to load
//! instead of being silently ignored; only [`DetectorOverride`] keeps
//! free-form detector keys.
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
use vision_calibration_core::SensorKind;
use vision_calibration_dataset::DatasetSpec;

/// Re-exported so callers naming [`DetectorOverride::chess_corners`] do not
/// need a direct `vision-calibration-dataset` dependency.
pub use vision_calibration_dataset::{ChessCornersDetectorSpec, CornerStrategySpec};
use vision_calibration_optim::{HandeyeMode, RobustLoss, SolverBackend};
use vision_calibration_pipeline::common::config::{CameraConfig, CameraOverride, RobotPoseConfig};
use vision_calibration_pipeline::rig_handeye::RigHandeyeConfig;
use vision_calibration_pipeline::single_cam_handeye::SingleCamHandeyeConfig;

/// Top-level benchmark registry: the full set of datasets the harness knows.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
    /// Sidecar device-spec (`spec.json`, ADR 0011) used to derive the
    /// seeded official initialization for acceptance runs. Relative paths
    /// resolve against `data_root`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_spec: Option<PathBuf>,
    /// Hard acceptance gate evaluated by `calib-bench accept`.
    /// `None` means the entry is not in the acceptance set — the runner
    /// reports it loudly instead of silently passing it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accept: Option<AcceptGate>,
    /// Planar intrinsics configuration overrides.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub planar_intrinsics: Option<PlanarIntrinsicsOverride>,
    /// Single-camera hand-eye configuration overrides.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub single_cam_handeye: Option<SingleCamHandeyeOverride>,
    /// Rig hand-eye configuration overrides.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rig_handeye: Option<RigHandeyeOverride>,
    /// Solver backend for every non-linear stage of `run` / `accept`
    /// (`calib-bench run --backend` sets it); `None` = the default backend.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub solver_backend: Option<SolverBackend>,
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

impl BenchEntry {
    /// The solver backend this entry runs on.
    pub fn solver_backend(&self) -> SolverBackend {
        self.solver_backend.unwrap_or_default()
    }

    /// Whether this is a planar-intrinsics entry for a tilted (Scheimpflug)
    /// camera, run through the seeded Scheimpflug route.
    pub fn is_scheimpflug_planar(&self) -> bool {
        self.problem == ProblemKind::PlanarIntrinsics
            && self
                .planar_intrinsics
                .as_ref()
                .and_then(|o| o.sensor)
                .is_some_and(|s| s == SensorKind::Scheimpflug)
    }
}

/// Planar intrinsics config overrides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct PlanarIntrinsicsOverride {
    /// The camera's sensor. `scheimpflug` runs the seeded Scheimpflug route,
    /// which needs the entry's `device_spec`; absent means an untilted camera.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sensor: Option<SensorKind>,
}

/// Hard per-dataset acceptance gate (`calib-bench accept`).
///
/// The seeded official route (ADR 0011) must bring **every** camera's
/// mean reprojection error at or under the threshold; a non-finite mean is
/// always a failure. Thresholds are per-entry because they encode
/// dataset-specific acceptance criteria (e.g. `0.5 px` for the Scheimpflug
/// intrinsics gates), not a universal quality bar.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcceptGate {
    /// Maximum allowed per-camera mean reprojection error, pixels.
    pub max_per_cam_mean_px: f64,
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
    /// Single-camera planar intrinsics (Zhang); a tilted sensor is selected
    /// by [`PlanarIntrinsicsOverride::sensor`].
    PlanarIntrinsics,
    /// Single-camera hand-eye.
    SingleCamHandeye,
    /// Multi-camera rig extrinsics.
    RigExtrinsics,
    /// Multi-camera rig hand-eye.
    RigHandeye,
    /// Joint multi-camera rig hand-eye + laserline.
    RigHandeyeLaserline,
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
#[serde(deny_unknown_fields)]
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
    ///
    /// Reuses the dataset-manifest type rather than redeclaring it: the
    /// benchmark registry and a `DatasetSpec` describe the same detector
    /// override, so one definition keeps the two from drifting.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chess_corners: Option<ChessCornersDetectorSpec>,
    /// Preserve older free-form detector keys that the benchmark does not
    /// interpret yet.
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// Laser configuration for laserline problems.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
/// Placeholder (`serde_json::Value`); the real, typed shape is not defined yet
/// alongside manual-init support.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ManualInitSeed(pub serde_json::Value);

/// Shared hand-eye bundle-adjustment overrides.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct HandeyeOptimizeOverride {
    /// Enable/disable per-view robot pose refinement.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refine_robot_poses: Option<bool>,
    /// Enable/disable rig-extrinsic refinement in final rig hand-eye BA.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refine_cam_se3_rig: Option<bool>,
    /// Enable/disable sensor-tilt refinement of Scheimpflug cameras in the
    /// final rig hand-eye BA.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refine_sensor: Option<bool>,
    /// Robot rotation prior sigma in radians.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub robot_rot_sigma: Option<f64>,
    /// Robot translation prior sigma in metres.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub robot_trans_sigma: Option<f64>,
}

/// Single-camera hand-eye config overrides.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct SingleCamHandeyeOverride {
    /// Camera model and fix lists.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera: Option<CameraConfig>,
    /// Hand-eye mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handeye_mode: Option<BenchHandeyeMode>,
    /// Solver max iterations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_iters: Option<usize>,
    /// Robust loss.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub robust_loss: Option<BenchRobustLoss>,
    /// Bundle-adjustment robot-pose settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handeye_optimize: Option<HandeyeOptimizeOverride>,
}

/// Rig hand-eye config overrides.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct RigHandeyeOverride {
    /// Camera model and fix lists of every camera.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera: Option<CameraConfig>,
    /// Cameras whose model differs from `camera`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera_overrides: Option<Vec<CameraOverride>>,
    /// Re-refine the tilt of Scheimpflug cameras in rig BA.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refine_sensor: Option<bool>,
    /// Hand-eye mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handeye_mode: Option<BenchHandeyeMode>,
    /// Solver max iterations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_iters: Option<usize>,
    /// Robust loss.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub robust_loss: Option<BenchRobustLoss>,
    /// Re-refine intrinsics in rig BA.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refine_intrinsics: Option<bool>,
    /// Fix tangential distortion (`p1`, `p2`) during per-camera intrinsics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fix_tangential: Option<bool>,
    /// Bundle-adjustment robot-pose settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handeye_optimize: Option<HandeyeOptimizeOverride>,
}

/// Registry hand-eye mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchHandeyeMode {
    /// Camera mounted on the gripper.
    EyeInHand,
    /// Camera fixed in workspace / rig sees robot-held target.
    EyeToHand,
}

impl From<BenchHandeyeMode> for HandeyeMode {
    fn from(value: BenchHandeyeMode) -> Self {
        match value {
            BenchHandeyeMode::EyeInHand => HandeyeMode::EyeInHand,
            BenchHandeyeMode::EyeToHand => HandeyeMode::EyeToHand,
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

impl SingleCamHandeyeOverride {
    /// Apply overrides to a default single-camera hand-eye config.
    pub fn apply_to(&self, config: &mut SingleCamHandeyeConfig) {
        if let Some(camera) = &self.camera {
            config.camera = camera.clone();
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
        if let Some(ba) = &self.handeye_optimize {
            apply_single_handeye_optimize(ba, config);
        }
    }
}

impl RigHandeyeOverride {
    /// Apply overrides to a default rig hand-eye config.
    pub fn apply_to(&self, config: &mut RigHandeyeConfig) {
        if let Some(camera) = &self.camera {
            config.camera = camera.clone();
        }
        if let Some(overrides) = &self.camera_overrides {
            config.camera_overrides = overrides.clone();
        }
        if let Some(refine) = self.refine_sensor {
            config.rig.refine_sensor = refine;
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
        if let Some(refine) = self.refine_intrinsics {
            config.rig.refine_intrinsics = refine;
        }
        if let Some(fix_tangential) = self.fix_tangential {
            config.intrinsics.fix_tangential = Some(fix_tangential);
        }
        if let Some(ba) = &self.handeye_optimize {
            apply_rig_handeye_optimize(ba, config);
        }
    }
}

/// Apply the robot-pose-refinement fields of a [`HandeyeOptimizeOverride`] to a
/// [`RobotPoseConfig`]. Shared by [`apply_single_handeye_optimize`] and [`apply_rig_handeye_optimize`],
/// which differ only in *where* their `RobotPoseConfig` lives
/// (`SingleCamHandeyeConfig::robot_poses` vs
/// `RigHandeyeConfig::handeye_optimize::robot_poses`) and in the rig-only fields
/// `apply_rig_handeye_optimize` applies afterward.
fn apply_robot_pose_override(ba: &HandeyeOptimizeOverride, robot_poses: &mut RobotPoseConfig) {
    if let Some(refine) = ba.refine_robot_poses {
        robot_poses.refine = refine;
    }
    if let Some(sigma) = ba.robot_rot_sigma {
        robot_poses.rot_sigma = sigma;
    }
    if let Some(sigma) = ba.robot_trans_sigma {
        robot_poses.trans_sigma = sigma;
    }
}

fn apply_single_handeye_optimize(
    ba: &HandeyeOptimizeOverride,
    config: &mut SingleCamHandeyeConfig,
) {
    apply_robot_pose_override(ba, &mut config.robot_poses);
}

fn apply_rig_handeye_optimize(ba: &HandeyeOptimizeOverride, config: &mut RigHandeyeConfig) {
    apply_robot_pose_override(ba, &mut config.handeye_optimize.robot_poses);
    if let Some(refine) = ba.refine_cam_se3_rig {
        config.handeye_optimize.refine_cam_se3_rig = refine;
    }
    if let Some(refine) = ba.refine_sensor {
        config.handeye_optimize.refine_sensor = refine;
    }
}

/// Stability-sampling configuration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
    use vision_calibration_core::{CameraFix, CameraModelSpec, DistortionKind, SensorKind};

    /// A Brown-Conrady camera with a Scheimpflug sensor and the given fix
    /// lists.
    fn tilted_camera(distortion: &[&str], sensor: &[&str]) -> CameraConfig {
        let names = |n: &[&str]| Some(n.iter().map(|s| s.to_string()).collect());
        let mut camera = CameraConfig::new(CameraModelSpec::pinhole(
            DistortionKind::BrownConrady5,
            SensorKind::Scheimpflug,
        ));
        camera.fix = CameraFix {
            distortion: names(distortion),
            sensor: names(sensor),
            ..CameraFix::default()
        };
        camera
    }
    use vision_calibration_dataset::{CameraSource, ImagePattern, TargetSpec, Topology};

    /// Compile-time guard: every [`ProblemKind`] variant has a `Topology`
    /// counterpart with the same name, and vice versa. If this `match` stops
    /// compiling (a variant added/removed on either side), the two enums have
    /// drifted — re-sync them.
    #[allow(dead_code)]
    fn problem_kind_mirrors_topology(t: Topology) -> ProblemKind {
        match t {
            Topology::PlanarIntrinsics => ProblemKind::PlanarIntrinsics,
            Topology::SingleCamHandeye => ProblemKind::SingleCamHandeye,
            Topology::RigExtrinsics => ProblemKind::RigExtrinsics,
            Topology::RigHandeye => ProblemKind::RigHandeye,
            Topology::RigHandeyeLaserline => ProblemKind::RigHandeyeLaserline,
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
                laser_images: None,
            }],
            target: TargetSpec::Chessboard {
                rows: 7,
                cols: 11,
                square_size_m: 0.03,
            },
            detector: None,
            robot_poses: None,
            laser: None,
            upstream_calibration: None,
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
            device_spec: Some(PathBuf::from("spec.json")),
            accept: Some(AcceptGate {
                max_per_cam_mean_px: 0.5,
            }),
            planar_intrinsics: None,
            single_cam_handeye: None,
            rig_handeye: Some(RigHandeyeOverride {
                camera: Some(tilted_camera(&["k2", "k3"], &["tilt_y"])),
                camera_overrides: None,
                refine_sensor: Some(false),
                handeye_mode: Some(BenchHandeyeMode::EyeToHand),
                max_iters: Some(200),
                robust_loss: Some(BenchRobustLoss::Huber { scale: 1.0 }),
                refine_intrinsics: Some(false),
                fix_tangential: Some(true),
                handeye_optimize: Some(HandeyeOptimizeOverride {
                    refine_robot_poses: Some(true),
                    refine_cam_se3_rig: Some(false),
                    refine_sensor: Some(false),
                    robot_rot_sigma: None,
                    robot_trans_sigma: None,
                }),
            }),
            solver_backend: Some(SolverBackend::Factrs),
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

        assert_eq!(cfg.camera.model.sensor, SensorKind::Scheimpflug);
        assert_eq!(cfg.handeye_init.handeye_mode, HandeyeMode::EyeToHand);
        assert_eq!(cfg.solver.max_iters, 200);
        assert_eq!(cfg.solver.robust_loss, RobustLoss::Huber { scale: 1.0 });
        assert!(cfg.handeye_optimize.robot_poses.refine);
        assert!(!cfg.handeye_optimize.refine_cam_se3_rig);
        assert!(!cfg.handeye_optimize.refine_sensor);
    }

    #[test]
    fn charuco_3536_override_maps_physical_setup() {
        let json = r#"
        {
          "camera": {
            "model": {
              "projection": "pinhole", "distortion": "brown_conrady5", "sensor": "scheimpflug"
            },
            "fix": { "distortion": ["p1", "p2"], "sensor": [] }
          },
          "refine_sensor": true,
          "handeye_mode": "eye_to_hand",
          "refine_intrinsics": true,
          "handeye_optimize": {
            "refine_robot_poses": true,
            "refine_cam_se3_rig": false,
            "refine_sensor": false
          }
        }
        "#;
        let override_cfg: RigHandeyeOverride = serde_json::from_str(json).expect("override");
        let mut cfg = RigHandeyeConfig::default();
        override_cfg.apply_to(&mut cfg);

        assert_eq!(cfg.handeye_init.handeye_mode, HandeyeMode::EyeToHand);
        assert!(cfg.rig.refine_intrinsics);
        assert!(!cfg.handeye_optimize.refine_cam_se3_rig);
        assert!(!cfg.handeye_optimize.refine_sensor);
        assert_eq!(cfg.camera.model.sensor, SensorKind::Scheimpflug);
        assert_eq!(
            cfg.camera.fix.distortion,
            Some(vec!["p1".to_string(), "p2".to_string()])
        );
        assert!(cfg.rig.refine_sensor);
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
