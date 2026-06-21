//! Typed error enum for `vision-mvg`.

use vision_calibration_core::Error as CoreError;
use vision_calibration_core::linalg::MathError;
use vision_geometry::GeometryError;

/// Convenience alias for results returned by this crate.
pub type Result<T> = std::result::Result<T, MvgError>;

/// Errors returned by public APIs in `vision-mvg`.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum MvgError {
    /// Input data is invalid for reasons other than count.
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

    /// The input configuration is geometrically degenerate.
    #[error("degenerate configuration: {reason}")]
    Degenerate {
        /// Human-readable description of the degeneracy.
        reason: String,
    },

    /// A robust estimator failed to find a consensus set.
    #[error("no consensus: robust estimation failed")]
    NoConsensus,

    /// Pose recovery found no candidate that passes the cheirality (positive-depth) test.
    #[error("no valid pose: all candidates fail cheirality")]
    NoValidPose,

    /// A nonlinear refinement did not converge.
    #[error("refinement did not converge: {reason}")]
    NotConverged {
        /// Human-readable description of what failed to converge.
        reason: String,
    },

    /// A numerical operation failed unexpectedly (e.g. an SVD).
    #[error("numerical failure: {0}")]
    Numerical(String),

    /// Forwarded error from `vision-geometry`.
    #[error(transparent)]
    Geometry(#[from] GeometryError),

    /// Forwarded error from `vision-calibration-core`.
    #[error(transparent)]
    Core(#[from] CoreError),
}

impl From<MathError> for MvgError {
    /// Map a shared-solver failure into this crate's typed error.
    fn from(e: MathError) -> Self {
        // `MathError` is `#[non_exhaustive]`; degrade to a generic numerical
        // failure rather than breaking the build downstream.
        Self::Numerical(e.to_string())
    }
}

impl MvgError {
    /// Convenience constructor for [`MvgError::InvalidInput`].
    ///
    /// Currently only used by the feature-gated nonlinear refinement.
    #[cfg(feature = "refine")]
    pub(crate) fn invalid_input(reason: impl Into<String>) -> Self {
        Self::InvalidInput {
            reason: reason.into(),
        }
    }

    /// Convenience constructor for [`MvgError::Degenerate`].
    pub(crate) fn degenerate(reason: impl Into<String>) -> Self {
        Self::Degenerate {
            reason: reason.into(),
        }
    }

    /// Convenience constructor for [`MvgError::NotConverged`].
    ///
    /// Convergence failures only arise in the feature-gated nonlinear refinement.
    #[cfg(feature = "refine")]
    pub(crate) fn not_converged(reason: impl Into<String>) -> Self {
        Self::NotConverged {
            reason: reason.into(),
        }
    }

    /// Convenience constructor for [`MvgError::Numerical`].
    pub(crate) fn numerical(msg: impl Into<String>) -> Self {
        Self::Numerical(msg.into())
    }
}
