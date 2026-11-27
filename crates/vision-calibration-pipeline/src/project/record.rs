//! What a project keeps of each run: the immutable record
//! ([`RunRecord`], `run.json`), the notes a person adds ([`RunNotes`],
//! `notes.json`) and the listing entry ([`RunInfo`]).

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use vision_calibration_dataset::{DatasetSpec, ProblemKind};

use super::{EXPORT_FILE, Exclusions, FeatureExclusion, QUALITY_FILE, REPORT_FILE};
use crate::analysis::{QualityThresholds, Verdict};
use crate::dataset_runner::{CalibrationConfig, ConfigPatch, RejectReason};

/// The record of one run, `run.json`: what went in, a digest of it, and a
/// summary of what came out. Written once with the run and never changed.
///
/// `timestamp` and `CalibratedRun::duration_ms` are the only clock
/// readings in a run's files; neither enters a digest, the export or the
/// quality report.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct RunRecord {
    /// The record's format version, [`RUN_RECORD_VERSION`](super::RUN_RECORD_VERSION).
    pub version: u32,
    /// When the run was written, seconds since the Unix epoch; for display
    /// and ordering only.
    pub timestamp: u64,
    /// The program that made the run, as it named itself (`vcal 1.1.0`).
    pub tool: String,
    /// The version of this library that wrote the run.
    pub library_version: String,
    /// Where the run's export came from.
    pub source: RunSource,
    /// What came out, for listing without reading the export.
    pub summary: RunSummary,
}

/// Where a run's export came from.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum RunSource {
    /// The project calibrated it.
    Calibrated(Box<CalibratedRun>),
    /// It was imported from an export calibrated elsewhere.
    Imported {
        /// The imported file's name (not its path).
        file_name: String,
    },
}

/// A run the project calibrated: its inputs, their digest and what the
/// detection kept.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct CalibratedRun {
    /// What the run was given.
    pub inputs: RunInputs,
    /// The digest of the inputs.
    pub digest: InputDigest,
    /// Views that entered the calibration.
    pub usable_views: usize,
    /// Views the dataset's images formed before screening.
    pub total_views: usize,
    /// The images left out, rejected by the screening or excluded.
    pub left_out: Vec<LeftOutImage>,
    /// Feature exclusions whose points matched no feature of the detection
    /// they were applied to, with those points only. An image left out
    /// before its features were screened is listed in `left_out` instead.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unmatched_exclusions: Vec<FeatureExclusion>,
    /// Whether every image was detected again rather than read from the
    /// detection cache.
    pub redetected: bool,
    /// How long the run took, milliseconds; for display only.
    pub duration_ms: u64,
}

/// What a calibrated run was given, resolved.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct RunInputs {
    /// The manifest as the run used it.
    pub dataset: DatasetSpec,
    /// The data root as the project stated it: the inline manifest's
    /// `root`, or the manifest file's path. For the reader only.
    pub data_root: PathBuf,
    /// The project's exclusions.
    pub exclusions: Exclusions,
    /// The configuration changes: the project's, then the run's own.
    pub config_patch: ConfigPatch,
    /// The configuration they made.
    pub config: CalibrationConfig,
    /// The thresholds the quality report was judged by.
    pub quality: QualityThresholds,
}

/// The digest of a calibrated run's inputs: each `sha256:<hex>` of the
/// canonical JSON (keys sorted) of what it covers.
///
/// Two runs with equal `inputs` digests, by the same library version, have
/// byte-identical `export.json` and `quality.json`. The data root's
/// location is not part of any digest: a dataset moved whole keeps them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct InputDigest {
    /// Over the other five, in their order here.
    pub inputs: String,
    /// The manifest, and the content of the files it names other than
    /// images (the robot pose file, the upstream calibration).
    pub dataset: String,
    /// The exclusions, notes left out, each list sorted and without
    /// repeats.
    pub exclusions: String,
    /// The resolved configuration.
    pub config: String,
    /// The quality thresholds.
    pub quality: String,
    /// The sorted set of detection cache keys the run read or wrote: each
    /// image's content, its detector and detector configuration and the
    /// detector's version.
    pub detections: String,
}

/// The digests of a project's inputs as they stand, for what is known
/// before any image is read: each part of an [`InputDigest`] but the
/// detections, computed as a run computes it. A calibrated run whose
/// digest differs from its project's in one of these parts was made from
/// other inputs than the project now states (a run's own configuration
/// changes count as such a difference).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct ProjectDigest {
    /// As [`InputDigest::dataset`].
    pub dataset: String,
    /// As [`InputDigest::exclusions`].
    pub exclusions: String,
    /// As [`InputDigest::config`], of the project's configuration.
    pub config: String,
    /// As [`InputDigest::quality`].
    pub quality: String,
}

/// An image a run left out, with the reason.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct LeftOutImage {
    /// The camera's id in the manifest.
    pub camera: String,
    /// The image, relative to the data root (absolute when outside it),
    /// `/`-separated.
    pub image: PathBuf,
    /// Why it was left out.
    pub reason: RejectReason,
}

/// What a run produced, in brief: enough to list runs without reading
/// their exports, and recomputable from the export.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct RunSummary {
    /// The workflow that produced the export.
    pub kind: ProblemKind,
    /// The quality report's verdict; `None` when the quality could not be
    /// assessed.
    pub verdict: Option<Verdict>,
    /// Each camera, in camera order.
    pub cameras: Vec<CameraSummary>,
}

/// One camera of a [`RunSummary`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct CameraSummary {
    /// The camera's id in the manifest.
    pub id: String,
    /// The root-mean-square reprojection error of its features, pixels:
    /// √(mean of the squared error lengths); `None` without residuals.
    pub rms_px: Option<f64>,
    /// Its features in the calibration.
    pub features: usize,
    /// The views it saw.
    pub views: usize,
}

/// What a person notes on a run, `notes.json`: the only file of a run that
/// changes after it is written.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(default)]
#[non_exhaustive]
pub struct RunNotes {
    /// A short name for the run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Free text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
    /// Whether the run is marked as one to keep.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub starred: bool,
}

impl RunNotes {
    /// No label, no notes, not starred.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether nothing is noted.
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

/// One run of a project, as a listing shows it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct RunInfo {
    /// The run's id: its folder's name under `runs/`.
    pub id: String,
    /// The run's folder, absolute.
    pub dir: PathBuf,
    /// What its files say.
    pub state: RunState,
    /// Its notes.
    pub notes: RunNotes,
}

impl RunInfo {
    /// The run's export, `export.json`.
    pub fn export_path(&self) -> PathBuf {
        self.dir.join(EXPORT_FILE)
    }

    /// The run's quality report, `quality.json` (absent for an imported
    /// export whose quality could not be assessed).
    pub fn quality_path(&self) -> PathBuf {
        self.dir.join(QUALITY_FILE)
    }

    /// The run's page, `report.html`, when it was asked for.
    pub fn report_path(&self) -> PathBuf {
        self.dir.join(REPORT_FILE)
    }

    /// The record, for a run written by this library.
    pub fn record(&self) -> Option<&RunRecord> {
        match &self.state {
            RunState::Recorded(record) => Some(record),
            _ => None,
        }
    }

    /// The summary of the run, when its files give one.
    pub fn summary(&self) -> Option<&RunSummary> {
        match &self.state {
            RunState::Recorded(record) => Some(&record.summary),
            RunState::Unrecorded(run) => run.summary.as_ref(),
            RunState::Unreadable { .. } => None,
        }
    }
}

/// What a run's files say.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum RunState {
    /// A run with its record.
    Recorded(Box<RunRecord>),
    /// A run of the desktop app's earlier format: an export with no
    /// record. It is listed, never rewritten.
    Unrecorded(UnrecordedRun),
    /// A run whose record this version cannot read.
    Unreadable {
        /// Why.
        message: String,
    },
}

/// A run of the desktop app's earlier format.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct UnrecordedRun {
    /// The summary of its export, when the export reads.
    pub summary: Option<RunSummary>,
    /// Views that entered the calibration, from its `diagnostics.json`.
    pub usable_views: Option<usize>,
    /// Views attempted, from its `diagnostics.json`.
    pub total_views: Option<usize>,
}
