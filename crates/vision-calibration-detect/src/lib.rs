//! Calibration-target feature detectors plus a content-addressed
//! detection cache.
//!
//! A [`DetectorConfig`] names a target and its geometry (one variant per
//! detector: chessboard, ChArUco, PuzzleBoard, coded ring grid) and detects
//! it in an 8-bit grayscale [`GrayImageView`]:
//!
//! ```ignore
//! let config = DetectorConfig::Chessboard(ChessboardConfig { rows: 7, cols: 11, square_size_m: 0.03, chess_corners: None });
//! let features = config.detect(GrayImageView::new(width, height, &pixels)?)?;
//! ```
//!
//! Each [`Feature`] pairs a pixel with its point on the target, in metres.
//! Cached features are keyed on the image content, the detector and its
//! configuration ([`CacheKey::for_detection`]), so re-running with the same
//! images and parameters is a filesystem read, and changing a parameter
//! recomputes only that detector's entries.

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
pub use charuco::{CharucoConfig, validate_charuco_layout, validate_dictionary};
#[cfg(any(feature = "charuco", feature = "chessboard"))]
pub use chess_options::{ChessCornersConfig, CornerCovariance, CornerStrategy, ImageNoise};
#[cfg(feature = "chessboard")]
pub use chessboard::ChessboardConfig;
#[cfg(feature = "puzzleboard")]
pub use puzzleboard::PuzzleboardConfig;
#[cfg(feature = "ringgrid")]
pub use ringgrid::RinggridConfig;

mod image_view;

pub use image_view::GrayImageView;

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

/// A calibration target and the detector that finds it. The set is closed:
/// one variant per built-in detector, each carrying its typed config.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(tag = "detector", rename_all = "snake_case")]
#[non_exhaustive]
pub enum DetectorConfig {
    /// A plain chessboard (`chess-corners` + `calib-targets`).
    #[cfg(feature = "chessboard")]
    Chessboard(ChessboardConfig),
    /// A ChArUco board (`calib-targets`).
    #[cfg(feature = "charuco")]
    Charuco(CharucoConfig),
    /// A PuzzleBoard (`calib-targets`).
    #[cfg(feature = "puzzleboard")]
    Puzzleboard(PuzzleboardConfig),
    /// A hexagonal coded ring grid (`ringgrid`).
    #[cfg(feature = "ringgrid")]
    Ringgrid(RinggridConfig),
}

impl DetectorConfig {
    /// Stable, lowercase detector name (`"chessboard"`, `"charuco"`, …).
    pub fn name(&self) -> &'static str {
        match self {
            #[cfg(feature = "chessboard")]
            Self::Chessboard(_) => "chessboard",
            #[cfg(feature = "charuco")]
            Self::Charuco(_) => "charuco",
            #[cfg(feature = "puzzleboard")]
            Self::Puzzleboard(_) => "puzzleboard",
            #[cfg(feature = "ringgrid")]
            Self::Ringgrid(_) => "ringgrid",
        }
    }

    /// Detect the target in `image`. A frame without the target yields an
    /// empty list, not an error.
    ///
    /// # Errors
    ///
    /// [`DetectError::InvalidConfig`] for an impossible board,
    /// [`DetectError::Backend`] when the wrapped detector fails for another
    /// reason than a missing target.
    pub fn detect(&self, image: GrayImageView<'_>) -> Result<Vec<Feature>, DetectError> {
        let luma = image.to_gray_image();
        match self {
            #[cfg(feature = "chessboard")]
            Self::Chessboard(cfg) => chessboard::detect(cfg, &luma),
            #[cfg(feature = "charuco")]
            Self::Charuco(cfg) => charuco::detect(cfg, &luma),
            #[cfg(feature = "puzzleboard")]
            Self::Puzzleboard(cfg) => puzzleboard::detect(cfg, &luma),
            #[cfg(feature = "ringgrid")]
            Self::Ringgrid(cfg) => ringgrid::detect(cfg, &luma),
        }
    }

    /// Whether the config asks for each corner's pixel covariance (only the
    /// chessboard-like detectors can provide one).
    pub fn requests_covariance(&self) -> bool {
        match self {
            #[cfg(feature = "chessboard")]
            Self::Chessboard(cfg) => cfg.chess_corners.is_some_and(|c| c.covariance.is_some()),
            #[cfg(feature = "charuco")]
            Self::Charuco(cfg) => cfg.chess_corners.is_some_and(|c| c.covariance.is_some()),
            #[allow(unreachable_patterns)]
            _ => false,
        }
    }

    /// The detection crate this detector wraps, with its version. The cache
    /// key carries it, so upgrading that crate recomputes cached detections.
    pub(crate) fn backend(&self) -> &'static str {
        match self {
            #[cfg(feature = "ringgrid")]
            Self::Ringgrid(_) => backend::RINGGRID,
            #[cfg(any(feature = "chessboard", feature = "charuco", feature = "puzzleboard"))]
            #[allow(unreachable_patterns)]
            _ => backend::CALIB_TARGETS,
        }
    }

    /// Revision of this detector's output, folded into its cache key: bumped
    /// when this crate changes the features returned for the same image and
    /// config between releases, so stale cache entries are not served.
    pub(crate) fn output_revision(&self) -> Option<u32> {
        match self {
            // Labels are bounded by `cols` across and `rows` down.
            #[cfg(feature = "chessboard")]
            Self::Chessboard(_) => Some(2),
            // Corner re-detection runs with the `chess_corners` override.
            #[cfg(feature = "charuco")]
            Self::Charuco(_) => Some(1),
            #[allow(unreachable_patterns)]
            _ => None,
        }
    }
}

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
