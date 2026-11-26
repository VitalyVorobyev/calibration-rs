//! One error envelope for every command, job and resource, on both
//! transports: the Tauri command rejects with it, the HTTP route answers it
//! as the body with a status from its kind, and a failed job carries it in
//! its terminal event.

use schemars::JsonSchema;
use serde::Serialize;

use crate::project::{ProjectError, ProjectErrorKind};

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
    /// No job with that id (it never existed or is no longer retained).
    JobNotFound,
    /// A failure of the back end itself.
    Internal,
    /// The folder has no project file.
    NotAProject,
    /// The folder already holds a project; creating one would overwrite it.
    ProjectExists,
    /// The project file has a version this app does not read.
    UnsupportedVersion,
    /// A run name that is empty or not a plain folder name.
    InvalidRunName,
    /// A run of that name is already saved; runs are never overwritten.
    RunExists,
    /// A file or folder could not be read or written.
    Io,
    /// A file did not parse.
    Parse,
    /// A dataset manifest that is not one.
    Manifest,
    /// A calibration config that is not valid for its workflow: a field it
    /// does not have, a value of the wrong type or out of range.
    Config,
    /// The request cannot be served as asked: no export is active, a camera
    /// index is out of range, the export lacks what the command needs.
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
        }
    }

    /// The error with the path it is about.
    pub fn at(mut self, path: impl Into<String>) -> Self {
        self.path = Some(path.into());
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
            ErrorKind::ForbiddenPath => 403,
            ErrorKind::UnknownCommand | ErrorKind::JobNotFound => 404,
            ErrorKind::InvalidArgs => 400,
            ErrorKind::Busy | ErrorKind::ProjectExists | ErrorKind::RunExists => 409,
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

impl From<ProjectError> for StudioError {
    fn from(e: ProjectError) -> Self {
        let kind = match e.kind {
            ProjectErrorKind::NotAProject => ErrorKind::NotAProject,
            ProjectErrorKind::ProjectExists => ErrorKind::ProjectExists,
            ProjectErrorKind::UnsupportedVersion => ErrorKind::UnsupportedVersion,
            ProjectErrorKind::InvalidRunName => ErrorKind::InvalidRunName,
            ProjectErrorKind::RunExists => ErrorKind::RunExists,
            ProjectErrorKind::Io => ErrorKind::Io,
            ProjectErrorKind::Parse => ErrorKind::Parse,
        };
        Self::new(kind, e.message)
    }
}

/// The command bodies shared with the viewer report a failure as a message:
/// a request the loaded export cannot serve.
impl From<String> for StudioError {
    fn from(message: String) -> Self {
        Self::invalid(message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_project_error_kind_maps_to_its_own_kind() {
        let kinds = [
            (ProjectErrorKind::NotAProject, "not_a_project", 422),
            (ProjectErrorKind::ProjectExists, "project_exists", 409),
            (
                ProjectErrorKind::UnsupportedVersion,
                "unsupported_version",
                422,
            ),
            (ProjectErrorKind::InvalidRunName, "invalid_run_name", 422),
            (ProjectErrorKind::RunExists, "run_exists", 409),
            (ProjectErrorKind::Io, "io", 422),
            (ProjectErrorKind::Parse, "parse", 422),
        ];
        for (kind, wire, status) in kinds {
            let e = StudioError::from(ProjectError::new(kind, "m"));
            assert_eq!(serde_json::to_value(e.kind).unwrap(), wire);
            assert_eq!(e.message, "m");
            assert_eq!(e.http_status(), status, "{wire}");
        }
    }

    #[test]
    fn the_envelope_omits_an_absent_path() {
        let e = StudioError::invalid("no export loaded yet");
        assert_eq!(
            serde_json::to_value(&e).unwrap(),
            serde_json::json!({"kind": "invalid", "message": "no export loaded yet"})
        );
        let e = StudioError::new(ErrorKind::ForbiddenPath, "out").at("/etc/passwd");
        assert_eq!(serde_json::to_value(&e).unwrap()["path"], "/etc/passwd");
        assert_eq!(e.http_status(), 403);
    }
}
