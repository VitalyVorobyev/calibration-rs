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

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod cache;
mod error;
mod feature;

#[cfg(feature = "charuco")]
mod charuco;
#[cfg(any(feature = "charuco", feature = "chessboard"))]
mod chess_options;
#[cfg(feature = "chessboard")]
mod chessboard;
#[cfg(feature = "puzzleboard")]
mod puzzleboard;
#[cfg(feature = "ringgrid")]
mod ringgrid;

pub use cache::{CacheError, CacheKey, CachedFeatures, DetectionCache, FsDetectionCache};
pub use error::DetectError;
pub use feature::{Feature, reject_ambiguous_detection};

#[cfg(feature = "charuco")]
pub use charuco::{CharucoConfig, CharucoDetector, validate_charuco_layout, validate_dictionary};
#[cfg(any(feature = "charuco", feature = "chessboard"))]
pub use chess_options::{ChessCornersConfig, CornerCovariance, CornerStrategy, ImageNoise};
#[cfg(feature = "chessboard")]
pub use chessboard::{ChessboardConfig, ChessboardDetector};
#[cfg(feature = "puzzleboard")]
pub use puzzleboard::{PuzzleboardConfig, PuzzleboardDetector};
#[cfg(feature = "ringgrid")]
pub use ringgrid::{RinggridConfig, RinggridDetector};

use serde_json::Value;

/// Sealed-trait guard: prevents downstream crates from implementing
/// [`Detector`]. The detector set is closed and crate-owned.
mod sealed {
    /// Private supertrait of [`Detector`](super::Detector). Only types in
    /// this crate can name it, so only this crate can implement `Detector`.
    pub trait Sealed {
        /// The detection crate this detector wraps, with its version
        /// (one of the `crate::backend` constants). The cache key carries
        /// it, so upgrading that crate recomputes cached detections.
        fn backend(&self) -> &'static str;
    }
}

/// The wrapped detection crates, as `"<crate> <version>"` pairs joined by
/// `" / "`. Each must name the version the workspace lockfile resolves (a
/// test checks it), since even a patch release of a detection crate can move
/// corners. `calib-targets` stands for its whole family: the test also checks
/// that every `calib-targets-*` crate resolves to the same version, because
/// the detection logic lives in those sub-crates and updating the facade
/// alone does not move them.
pub(crate) mod backend {
    #[cfg(any(feature = "chessboard", feature = "charuco", feature = "puzzleboard"))]
    pub const CALIB_TARGETS: &str = "calib-targets 0.16.1 / chess-corners 2.1.0";
    #[cfg(feature = "ringgrid")]
    pub const RINGGRID: &str = "ringgrid 0.13.2";
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
    ) -> Result<Vec<Feature>, DetectError>;

    /// Revision of this detector's output, folded into its cache key by
    /// [`CacheKey::for_detector`]. The key already carries this crate's
    /// release and the wrapped detection crate's version; bump the revision
    /// when this crate changes the features returned for the same image and
    /// config between releases, so stale cache entries are not served.
    /// `None` (the default) adds nothing.
    fn output_revision(&self) -> Option<u32> {
        None
    }
}

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
