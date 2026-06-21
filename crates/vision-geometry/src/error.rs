//! Typed error enum for `vision-geometry`.

use vision_calibration_core::Error as CoreError;
use vision_calibration_core::linalg::MathError;

/// Convenience alias for results returned by this crate.
pub type Result<T> = std::result::Result<T, GeometryError>;

/// Errors returned by public APIs in `vision-geometry`.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum GeometryError {
    /// Input data is invalid for reasons other than count (e.g. an intrinsics
    /// matrix that cannot be inverted).
    #[error("invalid input: {reason}")]
    InvalidInput {
        /// Human-readable description of why the input was rejected.
        reason: String,
    },

    /// Not enough data to proceed (e.g. fewer correspondences than a solver needs).
    #[error("insufficient data: need {need}, got {got}")]
    InsufficientData {
        /// Minimum number of correspondences required.
        need: usize,
        /// Actual number of correspondences supplied.
        got: usize,
    },

    /// Two input slices that must have equal length did not.
    #[error("count mismatch: expected {expected}, got {got}")]
    CountMismatch {
        /// Expected number of elements.
        expected: usize,
        /// Actual number of elements supplied.
        got: usize,
    },

    /// The input configuration is geometrically degenerate — collinear,
    /// coincident, or coplanar points; a zero baseline; or a rank-deficient
    /// design matrix.
    #[error("degenerate configuration: {reason}")]
    Degenerate {
        /// Human-readable description of the degeneracy.
        reason: String,
    },

    /// A matrix inversion or decomposition produced a singular result.
    #[error("singular matrix or degenerate configuration")]
    Singular,

    /// A robust estimator failed to find a consensus set.
    #[error("no consensus: robust estimation failed")]
    NoConsensus,

    /// A numerical operation failed unexpectedly (e.g. an SVD or polynomial solve).
    #[error("numerical failure: {0}")]
    Numerical(String),

    /// Forwarded error from `vision-calibration-core`.
    #[error(transparent)]
    Core(#[from] CoreError),
}

impl From<MathError> for GeometryError {
    /// Map a shared-solver failure into this crate's typed error, preserving the
    /// [`Singular`](GeometryError::Singular) variant so callers that match on it
    /// keep working after the `math` primitives moved to `vision-calibration-core`.
    fn from(e: MathError) -> Self {
        match e {
            MathError::Singular => Self::Singular,
            // `MathError` is `#[non_exhaustive]`; degrade any future variant to a
            // generic numerical failure rather than breaking the build downstream.
            other => Self::Numerical(other.to_string()),
        }
    }
}

impl GeometryError {
    /// Convenience constructor for [`GeometryError::Degenerate`].
    pub(crate) fn degenerate(reason: impl Into<String>) -> Self {
        Self::Degenerate {
            reason: reason.into(),
        }
    }

    /// Convenience constructor for [`GeometryError::Numerical`].
    pub(crate) fn numerical(msg: impl Into<String>) -> Self {
        Self::Numerical(msg.into())
    }
}
