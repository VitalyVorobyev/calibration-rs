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

use std::path::Path;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use vision_calibration_dataset::DatasetSpec;

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
///
/// Free-form for now (`serde_json::Value`); a typed shape arrives in a later
/// phase once the detector config schema settles.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DetectorOverride(pub serde_json::Value);

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
}

/// Source of robot poses for hand-eye problems.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoseSource {
    /// Path to the pose file.
    pub path: PathBuf,
    /// File format (e.g. `"csv"`, `"json"`).
    pub format: String,
    /// Pose convention (e.g. `"base_se3_gripper"`).
    pub convention: String,
}

/// Manual initialization seed.
///
/// Placeholder (`serde_json::Value`); the real, typed shape arrives in Phase 2
/// alongside manual-init support.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ManualInitSeed(pub serde_json::Value);

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
            detector: Some(DetectorOverride(serde_json::json!({"refine": true}))),
            laser: None,
            board: Some(BoardGeometry {
                rows: 13,
                cols: 13,
                cell_size_m: 0.01,
                dictionary: Some("DICT_4X4_50".into()),
                layout: Some("charuco".into()),
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
            }),
            prior_export: Some(PathBuf::from("prior.json")),
            fixture: Some(PathBuf::from("fixtures/puzzle.json")),
            seed: Some(ManualInitSeed(serde_json::json!({"fx": 1000.0}))),
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
}
