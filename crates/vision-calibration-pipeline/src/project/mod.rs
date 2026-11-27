//! Calibration projects: one dataset, the changes to its workflow's
//! configuration, what to leave out of it, and the runs calibrated from it.
//!
//! A project is a folder:
//!
//! ```text
//! <project>/
//!   calibration-project.toml   the definition (ProjectSpec)
//!   .detections/               the detection cache (content-addressed, disposable)
//!   runs/<run-id>/
//!     run.json                 the record: inputs, their digest, a summary (RunRecord)
//!     export.json              the workflow's export
//!     quality.json             the quality report
//!     notes.json               label, notes, star: the only file that changes (RunNotes)
//!     report.html              the run as one page, when asked for
//! ```
//!
//! [`Project::open`] reads a project folder, [`Project::create`] makes one
//! and [`Project::save`] writes an edited [`ProjectSpec`] back. The
//! dataset's manifest is written in the project file or kept in a file of
//! its own ([`DatasetSource`]); [`Exclusions`] name what the runs leave out
//! of it by what does not change when the dataset grows.
//!
//! [`Project::calibrate`] runs the dataset and writes the next
//! `runs/run-NNN/`; [`Project::import_export`] adds an export calibrated
//! elsewhere; [`Project::runs`] lists them. A run's [`RunRecord`] carries
//! the digest of its inputs ([`InputDigest`]): two runs with equal digests,
//! by the same library version, have byte-identical exports and quality
//! reports.
//!
//! A project's `[seed]` ([`SeedSource`]) names the seed its runs start
//! from: a previous run, a device specification or a seed file; each run's
//! record keeps the seed it used ([`RunSeed`]).
//!
//! [`Project::compare_runs`] and [`compare_recorded_runs`] put runs side
//! by side, each run's data relation read from the content of the images
//! its record lists ([`UsedImage`]).
//!
//! [`scan_library`] finds the projects under a set of folders and
//! [`LibraryScan::devices`] groups their runs by the device they calibrate;
//! [`DeviceHistory::representative_run`] picks the run that stands for a
//! device.

mod compare;
mod digest;
mod error;
mod exclude;
mod file;
mod fs;
mod library;
mod record;
mod run;
mod seed;
mod spec;

pub use compare::compare_recorded_runs;
pub use error::ProjectError;
pub use exclude::{
    ExclusionFault, Exclusions, FeatureExclusion, ImageExclusion, ViewExclusion, ViewRef,
};
pub use file::Project;
pub use library::{
    DeviceHistory, DeviceRun, LibraryScan, LibraryScanOptions, ProjectEntry, ScanFailure,
    scan_library,
};
pub use record::{
    CalibratedRun, CameraSummary, InputDigest, LeftOutImage, ProjectDigest, RunInfo, RunInputs,
    RunNotes, RunRecord, RunSource, RunState, RunSummary, UnrecordedRun, UsedImage,
};
pub use run::ProjectRunOptions;
pub use seed::{RunSeed, SeedSource};
pub use spec::{DatasetSource, DeviceIdentity, ProjectSpec};

/// The project file format version this library writes. Version 1 is the
/// desktop app's earlier format, read and migrated on save.
pub const PROJECT_VERSION: u32 = 2;
/// The project file of a new project.
pub const PROJECT_FILE: &str = "calibration-project.toml";
/// The project file of a project written as JSON, and of every version 1
/// project.
pub const PROJECT_FILE_JSON: &str = "calibration-project.json";
/// The folder of a project's runs.
pub const RUNS_DIR: &str = "runs";
/// The folder of a project's detection cache.
pub const DETECTIONS_DIR: &str = ".detections";
/// A run's record, [`RunRecord`].
pub const RUN_RECORD_FILE: &str = "run.json";
/// The format version of [`RunRecord`].
pub const RUN_RECORD_VERSION: u32 = 1;
/// A run's notes, [`RunNotes`].
pub const RUN_NOTES_FILE: &str = "notes.json";
/// A run's export.
pub const EXPORT_FILE: &str = "export.json";
/// A run's quality report.
pub const QUALITY_FILE: &str = "quality.json";
/// A run's page, when asked for.
pub const REPORT_FILE: &str = "report.html";
