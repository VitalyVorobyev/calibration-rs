//! Typed error hierarchy for `vision-calibration-pipeline`.

use vision_calibration_core::Error as CoreError;
use vision_calibration_core::UncertaintyUnavailable;
use vision_calibration_linear::Error as LinearError;
use vision_calibration_optim::Error as OptimError;

/// Errors returned by `vision-calibration-pipeline` public APIs.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// An argument violates a documented precondition.
    #[error("invalid input: {reason}")]
    InvalidInput {
        /// Human-readable description of why the input was rejected.
        reason: String,
    },

    /// Not enough observations to proceed.
    #[error("insufficient data: need {need}, got {got}")]
    InsufficientData {
        /// Minimum number of observations required.
        need: usize,
        /// Actual number of observations supplied.
        got: usize,
    },

    /// A required resource (input, state, output) was not yet set in the session.
    #[error("session resource not available: {resource}")]
    NotAvailable {
        /// Name of the session resource that was expected but not yet populated.
        resource: &'static str,
    },

    /// The result has no parameter covariance, which the analysis needs.
    #[error("the result has no parameter covariance: {reason:?}")]
    CovarianceUnavailable {
        /// Why the result has none.
        reason: UncertaintyUnavailable,
    },

    /// A numerical failure (singular matrix, divergence, etc.).
    #[error("numerical failure: {0}")]
    Numerical(String),

    /// An export was written in a format version this library does not read.
    #[error("unsupported export schema_version {got} (this library reads {expected})")]
    UnsupportedExportVersion {
        /// The `schema_version` the export carries.
        got: u64,
        /// The version this library reads for that export type.
        expected: u32,
    },

    /// Error propagated from a JSON serialization/deserialization step.
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    /// Error propagated from the core crate.
    #[error(transparent)]
    Core(#[from] CoreError),

    /// Error propagated from the linear crate.
    #[error(transparent)]
    Linear(#[from] LinearError),

    /// Error propagated from the optim crate.
    #[error(transparent)]
    Optim(OptimError),

    /// The run control's token was cancelled: the step, or the run, stopped
    /// and returned no result.
    #[error("cancelled")]
    Cancelled,
}

/// An optim error, with a cancelled solve lifted to [`Error::Cancelled`]:
/// one variant answers "was this run cancelled?" whichever crate stopped it.
impl From<OptimError> for Error {
    fn from(err: OptimError) -> Self {
        match err {
            OptimError::Cancelled => Self::Cancelled,
            other => Self::Optim(other),
        }
    }
}

impl From<vision_calibration_core::Cancelled> for Error {
    fn from(_: vision_calibration_core::Cancelled) -> Self {
        Self::Cancelled
    }
}

impl Error {
    /// Construct an [`Error::InvalidInput`] with the given reason.
    pub(crate) fn invalid_input(reason: impl Into<String>) -> Self {
        Self::InvalidInput {
            reason: reason.into(),
        }
    }

    /// Construct an [`Error::Numerical`] with the given message.
    pub(crate) fn numerical(msg: impl Into<String>) -> Self {
        Self::Numerical(msg.into())
    }

    /// A linear initializer's failure, as [`Error::Numerical`] prefixed by
    /// `context` — except views that do not determine the intrinsics
    /// ([`LinearError::DegenerateViews`]), which stay typed as
    /// [`Error::Linear`] so that a caller can tell a capture to redo from a
    /// solver failure.
    pub(crate) fn linear_init(context: impl std::fmt::Display, err: LinearError) -> Self {
        match err {
            err @ LinearError::DegenerateViews { .. } => Self::Linear(err),
            other => Self::numerical(format!("{context}: {other}")),
        }
    }

    /// An optim failure, as [`Error::Numerical`] prefixed by `context` —
    /// except a cancelled solve, which stays [`Error::Cancelled`].
    pub(crate) fn optim(context: impl std::fmt::Display, err: OptimError) -> Self {
        match err {
            OptimError::Cancelled => Self::Cancelled,
            other => Self::numerical(format!("{context}: {other}")),
        }
    }

    /// Construct an [`Error::NotAvailable`] for a named resource.
    pub(crate) fn not_available(resource: &'static str) -> Self {
        Self::NotAvailable { resource }
    }
}
