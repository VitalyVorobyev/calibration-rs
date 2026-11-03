//! Typed error enum for `vision-calibration-linear`.

use vision_calibration_core::Error as CoreError;
use vision_calibration_core::Real;
use vision_calibration_core::linalg::MathError;
use vision_geometry::GeometryError;

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

    /// The views do not determine the camera intrinsics: the board planes
    /// are parallel in every view (all fronto-parallel, or all tilted the
    /// same way), or Zhang's method finds no camera that fits them. Tilting
    /// the board about both of its axes, from view to view, fixes it.
    #[error("{}", degenerate_views_message(*views, *spread_deg))]
    DegenerateViews {
        /// Number of views given.
        views: usize,
        /// The largest angle, in degrees, between a view's board plane and
        /// the views' mean plane, when that spread is what rejected the
        /// views; `None` when Zhang's solution is no camera. Measured
        /// through a camera with square pixels whose focal length is the
        /// image extent the views cover.
        spread_deg: Option<Real>,
    },

    /// No motion pairs survived the min-rotation / axis-parallel filters
    /// during hand-eye initialization.
    #[error(
        "no valid hand-eye motion pairs after filtering (poses too similar or rotation axes near-parallel)"
    )]
    NoValidMotionPairs,

    /// Forwarded error from `vision-calibration-core`.
    #[error(transparent)]
    Core(#[from] CoreError),

    /// Forwarded error from `vision-geometry` (two-view geometric solvers).
    #[error(transparent)]
    Geometry(#[from] GeometryError),
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

/// The message of [`Error::DegenerateViews`].
fn degenerate_views_message(views: usize, spread_deg: Option<Real>) -> String {
    let evidence = match spread_deg {
        Some(spread) => format!(
            "the board planes are parallel to within {spread:.2}°, as when the board is \
             fronto-parallel or tilted the same way in every view"
        ),
        None => "Zhang's method finds no camera that fits them, as when the board planes \
                 are parallel (the board fronto-parallel or tilted the same way in every view)"
            .to_owned(),
    };
    format!(
        "the {views} views do not determine the camera intrinsics: {evidence}; \
         tilt the board about both of its axes, by 15° to 30°, from view to view"
    )
}
