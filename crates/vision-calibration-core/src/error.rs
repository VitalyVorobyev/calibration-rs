//! Typed error enum for `vision-calibration-core`.

/// Errors returned by public APIs in `vision-calibration-core`.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Input data is invalid (e.g. mismatched lengths, negative weights).
    #[error("invalid input: {reason}")]
    InvalidInput { reason: String },

    /// Not enough data to proceed (e.g. fewer correspondences than required).
    #[error("insufficient data: need {need}, got {got}")]
    InsufficientData { need: usize, got: usize },

    /// A matrix inversion or decomposition produced a degenerate result.
    #[error("singular matrix or degenerate configuration")]
    Singular,

    /// A numerical operation failed unexpectedly.
    #[error("numerical failure: {0}")]
    Numerical(String),
}

impl Error {
    /// Convenience constructor for [`Error::InvalidInput`].
    pub(crate) fn invalid_input(reason: impl Into<String>) -> Self {
        Self::InvalidInput {
            reason: reason.into(),
        }
    }
}
