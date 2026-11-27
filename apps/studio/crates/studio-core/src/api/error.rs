//! One error envelope for every command, job and resource, on both
//! transports: the Tauri command rejects with it, the HTTP route answers it
//! as the body with a status from its kind, and a failed job carries it in
//! its terminal event.

use schemars::JsonSchema;
use serde::Serialize;
use vision_calibration::dataset::ManifestError;
use vision_calibration::dataset_runner::{CalibrateDatasetError, RunError};
use vision_calibration::project::ProjectError;

/// Why a command, job or resource failed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, JsonSchema)]
pub struct StudioError {
    /// What the UI branches on.
    pub kind: ErrorKind,
    /// What it shows.
    pub message: String,
    /// The path the failure is about, when there is one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    /// The field it is about, as a dotted path into the project file or
    /// the manifest (`device.id`, `exclude.images[0]`, `pose_pairing`),
    /// when there is one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub field: Option<String>,
    /// The values to offer for `field` (`needs_input`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub suggestions: Vec<String>,
}

/// The kinds of [`StudioError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    /// A path outside every root folder, or not an absolute path.
    ForbiddenPath,
    /// No command or resource of that name (or not offered by this server).
    UnknownCommand,
    /// The arguments do not match the command's argument object.
    InvalidArgs,
    /// The job queue is full.
    Busy,
    /// The request is larger than the back end serves (an image past its
    /// pixel limit, a body past its byte limit).
    TooLarge,
    /// No job with that id (it never existed or is no longer retained).
    JobNotFound,
    /// The blob was evicted from the store: run its command again.
    BlobExpired,
    /// No session: the request carries neither the session cookie nor the
    /// token.
    Unauthorized,
    /// The roots are fixed while this back end runs (the server's).
    RootsFixed,
    /// A failure of the back end itself.
    Internal,
    /// The folder has no project file.
    NotAProject,
    /// The folder holds both project file names; which one counts is not
    /// guessed.
    AmbiguousProject,
    /// The folder already holds a project; creating one would overwrite it.
    ProjectExists,
    /// The project file has a version this app does not read.
    UnsupportedVersion,
    /// The project file changed on disk since the revision the edit was
    /// made on: open the project again and apply the edit to it.
    ChangedOnDisk,
    /// A file or folder could not be read or written.
    Io,
    /// A file did not parse.
    Parse,
    /// A dataset manifest that is not one, or not a valid one.
    Manifest,
    /// A calibration config that is not valid for its workflow: a field it
    /// does not have, a value of the wrong type or out of range.
    Config,
    /// An exclusion names something the dataset does not have (`field` is
    /// the entry).
    Exclusion,
    /// The project has no run of that id.
    RunNotFound,
    /// An export of another workflow or camera count than the project's.
    ImportMismatch,
    /// The seed cannot be made (a run without a record, a file that does
    /// not read) or does not fit the project's dataset and configuration
    /// (`field` is `seed`).
    Seed,
    /// The file exists and the command was not asked to overwrite it.
    FileExists,
    /// The manifest leaves a choice open: `field` names it and
    /// `suggestions` lists the values to offer.
    NeedsInput,
    /// Target or laser detection failed.
    Detection,
    /// The images could not be listed or paired, or the calibration
    /// failed.
    Calibration,
    /// The request cannot be served as asked: a camera index out of range,
    /// an export that lacks what the command needs, a field the project
    /// cannot have (`field` names it).
    Invalid,
    /// The job was cancelled.
    Cancelled,
}

impl StudioError {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            path: None,
            field: None,
            suggestions: Vec::new(),
        }
    }

    /// The error with the path it is about.
    pub fn at(mut self, path: impl Into<String>) -> Self {
        self.path = Some(path.into());
        self
    }

    /// The error with the field it is about.
    pub fn on_field(mut self, field: impl Into<String>) -> Self {
        self.field = Some(field.into());
        self
    }

    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Invalid, message)
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Internal, message)
    }

    pub fn cancelled() -> Self {
        Self::new(ErrorKind::Cancelled, "cancelled")
    }

    /// The HTTP status a transport answers this error with.
    pub fn http_status(&self) -> u16 {
        match self.kind {
            ErrorKind::Unauthorized => 401,
            ErrorKind::ForbiddenPath => 403,
            ErrorKind::UnknownCommand
            | ErrorKind::JobNotFound
            | ErrorKind::BlobExpired
            | ErrorKind::RunNotFound => 404,
            ErrorKind::TooLarge => 413,
            ErrorKind::InvalidArgs => 400,
            ErrorKind::Busy
            | ErrorKind::ProjectExists
            | ErrorKind::ChangedOnDisk
            | ErrorKind::FileExists
            | ErrorKind::RootsFixed => 409,
            ErrorKind::Internal => 500,
            _ => 422,
        }
    }
}

impl std::fmt::Display for StudioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

fn display(path: &std::path::Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

/// The error with the field the library names (`field()`), when it names
/// one.
fn with_field(error: StudioError, field: Option<String>) -> StudioError {
    match field {
        Some(field) => error.on_field(field),
        None => error,
    }
}

/// Each project error as its own kind, with the path it names and the field
/// the library names.
impl From<ProjectError> for StudioError {
    fn from(e: ProjectError) -> Self {
        let message = e.to_string();
        let field = e.field();
        let error = |kind| with_field(StudioError::new(kind, message.clone()), field.clone());
        match e {
            ProjectError::NotAProject { dir } => error(ErrorKind::NotAProject).at(display(&dir)),
            ProjectError::AmbiguousProject { dir } => {
                error(ErrorKind::AmbiguousProject).at(display(&dir))
            }
            ProjectError::ProjectExists { dir } => {
                error(ErrorKind::ProjectExists).at(display(&dir))
            }
            ProjectError::UnsupportedVersion { path, .. } => {
                error(ErrorKind::UnsupportedVersion).at(display(&path))
            }
            ProjectError::ChangedOnDisk { path } => {
                error(ErrorKind::ChangedOnDisk).at(display(&path))
            }
            ProjectError::Io { path, .. } => error(ErrorKind::Io).at(display(&path)),
            ProjectError::Parse { path, .. } => error(ErrorKind::Parse).at(display(&path)),
            ProjectError::Invalid { path, .. } => error(ErrorKind::Invalid).at(display(&path)),
            ProjectError::Manifest(e) => e.into(),
            ProjectError::Config(_) => error(ErrorKind::Config),
            ProjectError::Exclusion { .. } => error(ErrorKind::Exclusion),
            ProjectError::RunNotFound { .. } => error(ErrorKind::RunNotFound),
            ProjectError::ImportMismatch { .. } => error(ErrorKind::ImportMismatch),
            ProjectError::SeedUnresolved { path, .. } => error(ErrorKind::Seed).at(display(&path)),
            ProjectError::Calibrate(e) => e.into(),
            ProjectError::Cancelled => StudioError::cancelled(),
            // A variant of a later library version.
            _ => error(ErrorKind::Invalid),
        }
    }
}

impl From<ManifestError> for StudioError {
    fn from(e: ManifestError) -> Self {
        let error = with_field(
            StudioError::new(ErrorKind::Manifest, e.to_string()),
            e.field(),
        );
        match e {
            ManifestError::Io { path, .. } => StudioError {
                kind: ErrorKind::Io,
                ..error
            }
            .at(display(&path)),
            ManifestError::Parse { path, .. } => error.at(display(&path)),
            _ => error,
        }
    }
}

impl From<CalibrateDatasetError> for StudioError {
    fn from(e: CalibrateDatasetError) -> Self {
        let message = e.to_string();
        let field = e.field();
        let error = |kind| with_field(StudioError::new(kind, message.clone()), field.clone());
        match e {
            CalibrateDatasetError::Run(e) => e.into(),
            CalibrateDatasetError::Cancelled => StudioError::cancelled(),
            CalibrateDatasetError::ConfigTopologyMismatch { .. } => error(ErrorKind::Config),
            // The library names no field for a seed that does not fit; the
            // seed is the project's `seed`.
            CalibrateDatasetError::Seed(_) => error(ErrorKind::Seed).on_field("seed"),
            _ => error(ErrorKind::Calibration),
        }
    }
}

impl From<RunError> for StudioError {
    fn from(e: RunError) -> Self {
        let message = e.to_string();
        let field = e.field();
        let kind = |kind| with_field(StudioError::new(kind, message.clone()), field.clone());
        match e {
            RunError::AskUser {
                prompt,
                suggestions,
                ..
            } => StudioError {
                message: prompt,
                suggestions,
                ..kind(ErrorKind::NeedsInput)
            },
            RunError::Validation(_) => kind(ErrorKind::Manifest),
            RunError::Cancelled => StudioError::cancelled(),
            RunError::Detection { path, .. } | RunError::CornerCovariance { path, .. } => {
                kind(ErrorKind::Detection).at(display(&path))
            }
            RunError::LaserExtraction { .. } => kind(ErrorKind::Detection),
            // The cached features and a fresh detection of the image
            // disagree: the cache is not what this detector finds.
            RunError::DetailMismatch { path } => kind(ErrorKind::Detection).at(display(&path)),
            RunError::ImageNotInDataset { path, .. } => kind(ErrorKind::Invalid).at(display(&path)),
            RunError::Decode { path, .. } => kind(ErrorKind::Io).at(display(&path)),
            RunError::Io(_) => kind(ErrorKind::Io),
            _ => kind(ErrorKind::Calibration),
        }
    }
}

/// The command bodies of the viewers report a failure as a message: a
/// request the run's export cannot serve.
impl From<String> for StudioError {
    fn from(message: String) -> Self {
        Self::invalid(message)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn project_errors_map_to_their_own_kinds_with_path_and_field() {
        let cases = [
            (
                ProjectError::NotAProject {
                    dir: PathBuf::from("/p"),
                },
                "not_a_project",
                422,
            ),
            (
                ProjectError::ProjectExists {
                    dir: PathBuf::from("/p"),
                },
                "project_exists",
                409,
            ),
            (
                ProjectError::ChangedOnDisk {
                    path: PathBuf::from("/p/calibration-project.toml"),
                },
                "changed_on_disk",
                409,
            ),
            (
                ProjectError::RunNotFound { id: "run-9".into() },
                "run_not_found",
                404,
            ),
            (ProjectError::Cancelled, "cancelled", 422),
        ];
        for (error, wire, status) in cases {
            let e = StudioError::from(error);
            assert_eq!(serde_json::to_value(e.kind).unwrap(), wire);
            assert_eq!(e.http_status(), status, "{wire}");
        }
        let invalid = StudioError::from(ProjectError::Invalid {
            path: PathBuf::from("/p/calibration-project.toml"),
            field: "device.id".into(),
            message: "is empty".into(),
        });
        assert_eq!(invalid.kind, ErrorKind::Invalid);
        assert_eq!(invalid.field.as_deref(), Some("device.id"));
        assert_eq!(invalid.path.as_deref(), Some("/p/calibration-project.toml"));
    }

    #[test]
    fn an_open_choice_is_needs_input_with_its_field_and_suggestions() {
        let e = StudioError::from(RunError::AskUser {
            field: "pose_pairing".into(),
            prompt: "How are views paired?".into(),
            suggestions: vec!["by_index".into()],
        });
        assert_eq!(e.kind, ErrorKind::NeedsInput);
        assert_eq!(e.field.as_deref(), Some("pose_pairing"));
        assert_eq!(e.suggestions, ["by_index"]);
        assert_eq!(e.message, "How are views paired?");
    }

    #[test]
    fn a_seed_error_is_seed_on_the_seed_field() {
        let unresolved = StudioError::from(ProjectError::SeedUnresolved {
            path: PathBuf::from("/lib/a/runs/run-001"),
            message: "not a run".into(),
        });
        assert_eq!(unresolved.kind, ErrorKind::Seed);
        assert_eq!(unresolved.field.as_deref(), Some("seed"));
        assert_eq!(unresolved.path.as_deref(), Some("/lib/a/runs/run-001"));
        assert_eq!(unresolved.http_status(), 422);
        let misfit = StudioError::from(ProjectError::Calibrate(CalibrateDatasetError::Seed(
            vision_calibration::device_seed::DeviceSeedError::NoCameras,
        )));
        assert_eq!(misfit.kind, ErrorKind::Seed);
        assert_eq!(misfit.field.as_deref(), Some("seed"));
        assert_eq!(
            StudioError::new(ErrorKind::FileExists, "x").http_status(),
            409
        );
    }

    #[test]
    fn a_dataset_error_names_the_field_the_library_names() {
        use vision_calibration::dataset::ValidationError;

        let validation = || RunError::Validation(ValidationError::EmptyImagePattern("cam0".into()));
        let e = StudioError::from(validation());
        assert_eq!(e.kind, ErrorKind::Manifest);
        assert_eq!(e.field.as_deref(), Some("cameras[cam0].images"));
        // The same error through a project's run keeps its field.
        let e = StudioError::from(ProjectError::Calibrate(CalibrateDatasetError::Run(
            validation(),
        )));
        assert_eq!(e.field.as_deref(), Some("cameras[cam0].images"));
        // An error about an image names none.
        let e = StudioError::from(RunError::DetailMismatch {
            path: PathBuf::from("/d/a.png"),
        });
        assert_eq!((e.kind, e.field), (ErrorKind::Detection, None));
        assert_eq!(e.path.as_deref(), Some("/d/a.png"));
    }

    #[test]
    fn the_envelope_omits_what_it_lacks() {
        let e = StudioError::invalid("no such camera");
        assert_eq!(
            serde_json::to_value(&e).unwrap(),
            serde_json::json!({"kind": "invalid", "message": "no such camera"})
        );
        let e = StudioError::new(ErrorKind::ForbiddenPath, "out").at("/etc/passwd");
        assert_eq!(serde_json::to_value(&e).unwrap()["path"], "/etc/passwd");
        assert_eq!(e.http_status(), 403);
    }
}
