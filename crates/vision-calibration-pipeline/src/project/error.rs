//! [`ProjectError`]: why a project operation failed.

use std::path::{Path, PathBuf};

use thiserror::Error;
use vision_calibration_dataset::ManifestError;

use super::ExclusionFault;
use super::exclude::ResolveError;
use crate::dataset_runner::{CalibrateDatasetError, RunError};

/// Why a project operation failed.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ProjectError {
    /// The folder holds no project file.
    #[error(
        "{} is not a calibration project (it has no calibration-project.toml or \
         calibration-project.json)",
        dir.display()
    )]
    NotAProject {
        /// The folder.
        dir: PathBuf,
    },
    /// The folder holds both a `calibration-project.toml` and a
    /// `calibration-project.json`: which one defines the project is not
    /// guessed.
    #[error(
        "{} holds both calibration-project.toml and calibration-project.json; remove one",
        dir.display()
    )]
    AmbiguousProject {
        /// The folder.
        dir: PathBuf,
    },
    /// The folder already holds a project; creating one would overwrite it.
    #[error("{} already holds a calibration project", dir.display())]
    ProjectExists {
        /// The folder.
        dir: PathBuf,
    },
    /// The project file has a version this library does not read.
    #[error(
        "{}: project version {version}; this version reads versions 1 and 2",
        path.display()
    )]
    UnsupportedVersion {
        /// The project file.
        path: PathBuf,
        /// The version it states.
        version: u32,
    },
    /// The project file changed on disk since it was opened or last saved:
    /// saving would overwrite that change. Reload, apply the edit again and
    /// save.
    #[error(
        "{} changed on disk since it was read; reload it and apply the change again",
        path.display()
    )]
    ChangedOnDisk {
        /// The project file.
        path: PathBuf,
    },
    /// A file or folder could not be read or written.
    #[error("{}: {source}", path.display())]
    Io {
        /// The file or folder.
        path: PathBuf,
        /// The failure.
        #[source]
        source: std::io::Error,
    },
    /// A project or run file is not valid in its format.
    #[error("{}: {message}", path.display())]
    Parse {
        /// The file.
        path: PathBuf,
        /// The parser's message.
        message: String,
    },
    /// A field of the project holds a value it cannot have.
    #[error("{}: {field}: {message}", path.display())]
    Invalid {
        /// The project file.
        path: PathBuf,
        /// The field, as `device.id`.
        field: String,
        /// What is wrong with it.
        message: String,
    },
    /// The project's manifest file could not be read.
    #[error(transparent)]
    Manifest(#[from] ManifestError),
    /// The project's configuration changes do not make a valid
    /// configuration of its workflow.
    #[error("the project's configuration: {0}")]
    Config(crate::Error),
    /// An exclusion names something the dataset does not have.
    #[error("{entry}: {fault}; remove the entry or correct it")]
    Exclusion {
        /// The exclusion, as `exclude.<list>[<index>]`.
        entry: String,
        /// What is wrong with it.
        fault: ExclusionFault,
    },
    /// The project has no run of that id.
    #[error("the project has no run {id:?}")]
    RunNotFound {
        /// The id asked for.
        id: String,
    },
    /// An export does not belong to the project: another workflow, or
    /// another number of cameras.
    #[error("the export does not match the project: {message}")]
    ImportMismatch {
        /// How it differs.
        message: String,
    },
    /// The dataset's images could not be listed or paired, or the
    /// calibration failed. Never [`CalibrateDatasetError::Cancelled`]: that
    /// is [`ProjectError::Cancelled`].
    #[error(transparent)]
    Calibrate(CalibrateDatasetError),
    /// The run control's token was cancelled; nothing was written.
    #[error("cancelled")]
    Cancelled,
}

impl ProjectError {
    pub(crate) fn io(path: &Path, source: std::io::Error) -> Self {
        Self::Io {
            path: path.to_path_buf(),
            source,
        }
    }

    pub(crate) fn parse(path: &Path, message: impl std::fmt::Display) -> Self {
        Self::Parse {
            path: path.to_path_buf(),
            message: message.to_string(),
        }
    }

    pub(crate) fn invalid(path: &Path, field: &str, message: impl Into<String>) -> Self {
        Self::Invalid {
            path: path.to_path_buf(),
            field: field.to_string(),
            message: message.into(),
        }
    }
}

impl From<CalibrateDatasetError> for ProjectError {
    fn from(err: CalibrateDatasetError) -> Self {
        match err {
            CalibrateDatasetError::Cancelled => Self::Cancelled,
            other => Self::Calibrate(other),
        }
    }
}

impl From<RunError> for ProjectError {
    fn from(err: RunError) -> Self {
        CalibrateDatasetError::from(err).into()
    }
}

impl From<ResolveError> for ProjectError {
    fn from(err: ResolveError) -> Self {
        match err {
            ResolveError::Run(err) => err.into(),
            ResolveError::Exclusion { entry, fault } => Self::Exclusion { entry, fault },
        }
    }
}
