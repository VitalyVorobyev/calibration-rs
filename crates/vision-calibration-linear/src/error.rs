//! Typed error enum for `vision-calibration-linear`.

use vision_calibration_core::Error as CoreError;
use vision_calibration_core::linalg::MathError;

/// Errors returned by public APIs in `vision-calibration-linear`.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Input data is invalid (e.g. mismatched lengths, wrong point count).
    #[error("invalid input: {reason}")]
    InvalidInput {
        /// Human-readable description of why the input was rejected.
        reason: String,
    },

    /// Not enough data to proceed (e.g. fewer correspondences than required).
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

impl From<MathError> for Error {
    /// Map a shared-solver failure into this crate's typed error, preserving the
    /// `Singular` variant so existing callers and tests that match on it keep
    /// working after the `math` primitives moved to `vision-calibration-core`.
    fn from(e: MathError) -> Self {
        match e {
            MathError::Singular => Self::Singular,
            // `MathError` is `#[non_exhaustive]`; degrade any future variant to a
            // generic numerical failure rather than breaking the build downstream.
            other => Self::Numerical(other.to_string()),
        }
    }
}

impl Error {
    /// Convenience constructor for [`Error::InvalidInput`].
    pub(crate) fn invalid_input(reason: impl Into<String>) -> Self {
        Self::InvalidInput {
            reason: reason.into(),
        }
    }

    /// Convenience constructor for [`Error::Numerical`].
    pub(crate) fn numerical(msg: impl Into<String>) -> Self {
        Self::Numerical(msg.into())
    }
}
