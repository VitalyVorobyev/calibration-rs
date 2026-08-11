//! Typed error type for the calibration-target detectors.

/// Errors returned by the built-in detectors.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DetectError {
    /// The type-erased JSON config did not deserialize into the detector's
    /// config struct.
    #[error("invalid {detector} config: {source}")]
    Config {
        /// Detector name (`"charuco"`, `"chessboard"`, …).
        detector: &'static str,
        /// Underlying serde error.
        #[source]
        source: serde_json::Error,
    },
    /// A detector configuration or target-geometry constraint was violated.
    #[error("{0}")]
    InvalidConfig(String),
    /// The underlying detector backend failed for a reason other than "no
    /// target in this frame".
    ///
    /// A frame that simply contains no target is **not** an error — every
    /// detector reports that as an empty [`Feature`](crate::Feature) list.
    /// This variant is reserved for genuine backend failures, so a
    /// misconfigured board cannot masquerade as an unlucky image.
    ///
    /// The source is carried as a rendered string rather than a typed error:
    /// the backends are optional dependencies behind per-detector features,
    /// so a typed `#[source]` would leak them into this crate's public API
    /// and make the variant's shape depend on the enabled feature set.
    #[error("{detector} detector backend failed: {message}")]
    Backend {
        /// Detector name (`"ringgrid"`, `"chessboard"`, …).
        detector: &'static str,
        /// Rendered backend error.
        message: String,
    },
}
