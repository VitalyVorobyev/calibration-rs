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
}
