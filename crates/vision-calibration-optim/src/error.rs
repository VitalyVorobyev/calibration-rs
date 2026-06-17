//! Typed error enum for `vision-calibration-optim`.

use vision_calibration_core::Error as CoreError;

/// Errors returned by public APIs in `vision-calibration-optim`.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Input data is invalid (e.g. mismatched lengths, wrong parameter count).
    #[error("invalid input: {reason}")]
    InvalidInput {
        /// Human-readable description of why the input was rejected.
        reason: String,
    },

    /// Not enough data to proceed.
    #[error("insufficient data: need {need}, got {got}")]
    InsufficientData {
        /// Minimum number of observations required.
        need: usize,
        /// Actual number of observations supplied.
        got: usize,
    },

    /// A matrix inversion or decomposition produced a degenerate result.
    #[error("singular matrix or degenerate configuration")]
    Singular,

    /// A numerical operation failed unexpectedly.
    #[error("numerical failure: {0}")]
    Numerical(String),

    /// Forwarded error from `vision-calibration-core`.
    #[error(transparent)]
    Core(#[from] CoreError),
}

impl Error {
    /// Convenience constructor for [`Error::InvalidInput`].
    pub(crate) fn invalid_input(reason: impl Into<String>) -> Self {
        Self::InvalidInput {
            reason: reason.into(),
        }
    }
}

impl From<anyhow::Error> for Error {
    fn from(e: anyhow::Error) -> Self {
        Self::Numerical(e.to_string())
    }
}
