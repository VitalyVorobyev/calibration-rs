//! Calibration-target feature detectors plus a content-addressed
//! detection cache.
//!
//! Every detector exposes the same minimal interface:
//!
//! ```ignore
//! pub trait Detector {
//!     fn name(&self) -> &'static str;
//!     fn detect_json(&self, image: &DynamicImage, config: &Value) -> Result<Vec<Feature>>;
//! }
//! ```
//!
//! The dispatcher (in `vision-calibration-pipeline`) maps a
//! `vision_calibration_dataset::TargetSpec` to a `(name, config_json)`
//! pair and calls into a registered detector. Cached features are keyed
//! on `(image_content_hash, detector_name, canonical_config_hash)` so
//! re-running with the same images and same detector params is free
//! (filesystem read only, no detection re-run); changing detector
//! params auto-invalidates that detector's entries while leaving
//! others intact.
//!
//! See ADR 0017 for the cache contract.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod cache;
mod feature;

#[cfg(feature = "charuco")]
mod charuco;
#[cfg(any(feature = "charuco", feature = "chessboard"))]
mod chess_options;
#[cfg(feature = "chessboard")]
mod chessboard;

pub use cache::{
    CacheError, CacheKey, CachedFeatures, DetectionCache, FsDetectionCache, hash_image_bytes,
};
pub use feature::Feature;

#[cfg(feature = "charuco")]
pub use charuco::{CharucoConfig, CharucoDetector, validate_charuco_layout, validate_dictionary};
#[cfg(any(feature = "charuco", feature = "chessboard"))]
pub use chess_options::{ChessCornersConfig, ChessThresholdMode};
#[cfg(feature = "chessboard")]
pub use chessboard::{ChessboardConfig, ChessboardDetector};

use serde_json::Value;

/// Sealed-trait guard: prevents downstream crates from implementing
/// [`Detector`]. The detector set is closed and crate-owned.
mod sealed {
    /// Private supertrait of [`Detector`](super::Detector). Only types in
    /// this crate can name it, so only this crate can implement `Detector`.
    pub trait Sealed {}
}

/// Minimal detector interface used by the runner. Configs are passed
/// in type-erased JSON form so dispatch (and the cache key derivation)
/// can be data-driven from the dataset manifest without monomorphising
/// over every detector.
///
/// This trait is **sealed**: it is implemented only by the built-in
/// detectors in `vision-calibration-detect` and cannot be implemented
/// by downstream crates.
pub trait Detector: sealed::Sealed + Send + Sync {
    /// Stable, lowercase name (`"chessboard"`, `"charuco"`, …).
    fn name(&self) -> &'static str;

    /// Detect calibration-target features in `image`. The `config`
    /// must deserialize into the detector-specific config struct.
    fn detect_json(
        &self,
        image: &image::DynamicImage,
        config: &Value,
    ) -> anyhow::Result<Vec<Feature>>;
}
