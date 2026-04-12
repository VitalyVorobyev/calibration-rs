//! Typed error hierarchy for `vision-calibration-pipeline`.

use vision_calibration_core::Error as CoreError;
use vision_calibration_linear::Error as LinearError;
use vision_calibration_optim::Error as OptimError;

/// Errors returned by `vision-calibration-pipeline` public APIs.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// An argument violates a documented precondition.
    #[error("invalid input: {reason}")]
    InvalidInput { reason: String },

    /// Not enough observations to proceed.
    #[error("insufficient data: need {need}, got {got}")]
    InsufficientData { need: usize, got: usize },

    /// A required resource (input, state, output) was not yet set in the session.
    #[error("session resource not available: {resource}")]
    NotAvailable { resource: &'static str },

    /// A numerical failure (singular matrix, divergence, etc.).
    #[error("numerical failure: {0}")]
    Numerical(String),

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
    Optim(#[from] OptimError),
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

    /// Construct an [`Error::NotAvailable`] for a named resource.
    pub(crate) fn not_available(resource: &'static str) -> Self {
        Self::NotAvailable { resource }
    }
}

/// Allow `anyhow::Error` to be converted to a pipeline `Error` (catches any
/// internal `anyhow`-style failures at the session boundary).
impl From<anyhow::Error> for Error {
    fn from(e: anyhow::Error) -> Self {
        Self::Numerical(e.to_string())
    }
}
