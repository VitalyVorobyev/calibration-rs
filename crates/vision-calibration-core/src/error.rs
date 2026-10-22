//! Typed error enum for `vision-calibration-core`.

/// Errors returned by public APIs in `vision-calibration-core`.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Input data is invalid (e.g. mismatched lengths, a covariance that is
    /// not positive definite).
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

    /// Some views state per-point covariances and others do not. A
    /// calibration input states them for every view or for none.
    #[error("views must all state per-point covariances or none: {with} do, {without} do not")]
    MixedCovariance {
        /// Views (with points) that state covariances.
        with: usize,
        /// Views (with points) that do not.
        without: usize,
    },

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
