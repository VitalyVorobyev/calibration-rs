//! Dataset-driven runner: takes a
//! [`DatasetSpec`] manifest plus a
//! per-problem `*Config` and produces the existing `*Input` IR by
//! running detection (cached) on the per-camera images.
//!
//! [`calibrate_dataset`] runs a manifest end to end: it picks the converter
//! and the problem type by the manifest's topology and returns the export as
//! a [`CalibrationExport`].
//! [`camera_image_sizes`] reads each camera's image size, which the
//! image-wide analyses take.
//!
//! Converters, for a run driven step by step:
//! - [`build_planar_input`] — `PlanarIntrinsics` (any camera model)
//!   (both consume a `PlanarDataset`).
//! - [`build_rig_extrinsics_input`] — `RigExtrinsics` (`RigDataset<NoMeta>`).
//! - [`build_rig_handeye_input`] — `RigHandeye`
//!   (`RigDataset<RobotPoseMeta>`, robot poses loaded from the manifest's
//!   [`RobotPoseSource`](vision_calibration_dataset::RobotPoseSource)).
//! - [`build_single_cam_handeye_input`] — `SingleCamHandeye`
//!   (`SingleCamHandeyeInput`, single camera + robot poses).
//! - [`build_laserline_device_input`] / [`build_rig_laserline_device_input`]
//!   — the laser topologies. Laser-line extraction is
//!   *injected* via the open [`LaserPixelExtractor`] trait because the
//!   reference implementation (`vision-metrology`) is not on crates.io.
//!
//! Every converter screens each target detection the same way, and lists
//! the ones it leaves out in its result's `rejected` field, each with a
//! [`RejectReason`]:
//! - a detection with fewer features than the per-view floor (4, or the
//!   manifest's `detector.min_features_per_view`) is left out;
//! - in a topology whose views share one target frame (single-camera
//!   hand-eye and every rig topology), a plain chessboard detection is kept
//!   only when it spans the whole board: corners on every edge. The
//!   detector labels a view from its own top-left corner, so a partial
//!   board would move the shared frame. Self-identifying targets (ChArUco,
//!   PuzzleBoard, ring grid) label partial views absolutely and are kept;
//! - in a rig, each camera's whole-board chessboard labels are turned to
//!   agree with the other cameras of the view, from the board's direction
//!   in each image and the cameras' relative turn, which the views they
//!   share establish. A detection whose turn cannot be matched is left out
//!   ([`RejectReason::AmbiguousChessboardTurn`]);
//! - where the target frame is fixed through the views (hand-eye and the
//!   laser rigs), the frame is the one camera 0 labels, and a board camera
//!   0 sees turned by 90° is left out ([`RejectReason::TurnedChessboard`]).
//!
//! A camera left with no usable view is an error before any solve:
//! [`RunError::NoWholeChessboardViews`] when a chessboard rule rejected any
//! of its views, [`RunError::InsufficientUsableViews`] otherwise.
//!
//! Any ambiguity that cannot be auto-resolved at
//! conversion time is surfaced as a [`RunError::AskUser`] event so
//! the UI can prompt the user instead of silently guessing.

use std::path::{Path, PathBuf};

use thiserror::Error;

use vision_calibration_core::{
    Cancelled, CorrespondenceView, NoMeta, PointCovariance, Progress, Pt2, Pt3, RunControl, View,
};
use vision_calibration_dataset::{
    ChessCornersDetectorSpec, CornerCovarianceSpec, CornerStrategySpec, DatasetSpec, DetectorSpec,
    ImageNoiseSpec, ImagePattern, TargetSpec, Topology, ValidationError,
};
use vision_calibration_detect::{
    CacheError, CacheKey, CachedFeatures, CharucoConfig, ChessCornersConfig, ChessboardConfig,
    CornerCovariance, CornerStrategy, DetectionCache, DetectorConfig, Feature, GrayImageView,
    ImageNoise, PuzzleboardConfig, RinggridConfig, validate_charuco_layout,
};

mod board_turn;
mod calibrate;
mod handeye;
mod laser;
mod pairing;
mod planar;
mod poses;
mod rig;
mod view_filter;

pub use calibrate::{
    CalibrateDatasetError, CalibrationConfig, CalibrationExport, DatasetCalibration,
    calibrate_dataset,
};
pub use handeye::{HandeyeRunResult, build_single_cam_handeye_input};
pub use laser::{
    LaserPixelExtractor, LaserlineRunResult, RigHandeyeLaserlineRunResult, RigLaserlineRunResult,
    build_laserline_device_input, build_rig_handeye_laserline_input,
    build_rig_laserline_device_input,
};
pub use pairing::PairedViews;
pub use planar::{PlanarRunResult, build_planar_input};
pub use rig::{RigRunResult, build_rig_extrinsics_input, build_rig_handeye_input};
pub use view_filter::{RejectReason, RejectedDetection};

/// Errors produced by the dataset-driven runner.
#[derive(Debug, Error)]
pub enum RunError {
    /// Manifest failed structural validation. Wraps the underlying
    /// [`ValidationError`].
    #[error("manifest validation failed: {0}")]
    Validation(#[from] ValidationError),

    /// Topology is not supported by this converter (e.g. a rig manifest
    /// handed to the planar converter, or a topology whose converter
    /// has not shipped yet).
    #[error("topology {topology:?} is not supported by this converter")]
    UnsupportedTopology {
        /// The offending topology.
        topology: Topology,
    },

    /// Target type is not supported by any registered detector.
    #[error("target type {kind} is not supported by any registered detector")]
    UnsupportedTarget {
        /// Discriminator from the manifest (e.g. `"ringgrid"`).
        kind: String,
    },

    /// Target config carries an invalid value (e.g. an unknown ArUco
    /// dictionary name). Caught before any image I/O so manifest typos
    /// fail fast.
    #[error("invalid target config: {message}")]
    InvalidTargetConfig {
        /// Human-readable description of the bad field/value.
        message: String,
    },

    /// Glob pattern produced no matches. The user almost certainly
    /// meant a different pattern; surface as a fail-fast event.
    #[error("camera {camera_id:?}: pattern {pattern:?} matched no files under {base}")]
    EmptyImageMatch {
        /// The offending camera id.
        camera_id: String,
        /// The pattern that matched nothing.
        pattern: String,
        /// The base directory the pattern was resolved against.
        base: PathBuf,
    },

    /// Glob crate refused to compile the user's pattern.
    #[error("camera {camera_id:?}: invalid glob pattern {pattern:?} ({source})")]
    BadGlob {
        /// The offending camera id.
        camera_id: String,
        /// The pattern that failed to parse.
        pattern: String,
        /// Underlying glob compile error.
        #[source]
        source: glob::PatternError,
    },

    /// Cameras expanded to different image counts under `by_index`
    /// pairing, where every camera must contribute one image per view.
    #[error("by_index pairing requires equal image counts per camera, got {counts:?}")]
    ViewCountMismatch {
        /// `(camera_id, image_count)` per camera, in manifest order.
        counts: Vec<(String, usize)>,
    },

    /// Robot pose count does not match the paired view count.
    #[error("pose file has {poses} poses but the cameras paired into {views} views")]
    PoseCountMismatch {
        /// Number of poses parsed from the pose file.
        poses: usize,
        /// Number of paired views.
        views: usize,
    },

    /// A row of the robot pose file failed to parse.
    #[error("pose file {path}, row {row}: {message}")]
    PoseParse {
        /// The pose file.
        path: PathBuf,
        /// Zero-based data-row index (header excluded for CSV).
        row: usize,
        /// What went wrong.
        message: String,
    },

    /// The `shared_filename_token` regex failed to compile.
    #[error("pose pairing regex {regex:?} failed to compile: {message}")]
    BadPairingRegex {
        /// The offending regex source.
        regex: String,
        /// Compile error text.
        message: String,
    },

    /// Filename-token pairing failed: a filename did not match the
    /// regex, the named group was missing, or token sets are unusable.
    #[error("filename-token pairing failed: {message}")]
    PairingTokenMismatch {
        /// Human-readable description (names the offending file/token).
        message: String,
    },

    /// I/O failure while reading/writing the cache or images.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Image decode failure.
    #[error("decoding image {path}: {source}")]
    Decode {
        /// File that failed to decode.
        path: PathBuf,
        /// Underlying error.
        #[source]
        source: image::ImageError,
    },

    /// A camera's images differ in size. One camera's intrinsics hold for
    /// one image size, so its images must share it.
    #[error(
        "camera {camera_id:?}: {path} is {size:?} pixels but {first} is {first_size:?}; \
         a camera's images must all have one size"
    )]
    ImageSizeMismatch {
        /// Camera whose images differ.
        camera_id: String,
        /// The camera's first image.
        first: PathBuf,
        /// Its `[width, height]` in pixels.
        first_size: [u32; 2],
        /// The first image of another size.
        path: PathBuf,
        /// Its `[width, height]` in pixels.
        size: [u32; 2],
    },

    /// Detection failure for a specific image.
    #[error("detector {detector} failed on {path}: {source}")]
    Detection {
        /// Detector name.
        detector: String,
        /// Offending image.
        path: PathBuf,
        /// Underlying error.
        #[source]
        source: vision_calibration_detect::DetectError,
    },

    /// Corner covariance was requested (`detector.chess_corners.covariance`)
    /// but the detector's output for this image does not carry a usable one.
    #[error("detector {detector} on {path}: {fault}")]
    CornerCovariance {
        /// Detector name.
        detector: String,
        /// Offending image.
        path: PathBuf,
        /// What is wrong with the reported covariance.
        fault: CovarianceFault,
    },

    /// The manifest asks for corner covariance on a target whose detector
    /// reports none.
    #[error(
        "detector.chess_corners.covariance is supported for chessboard and charuco \
         targets only, not {target}"
    )]
    CovarianceUnsupported {
        /// The manifest's target kind (e.g. `"puzzleboard"`).
        target: &'static str,
    },

    /// Cache backend reported an error.
    #[error("cache error: {0}")]
    Cache(#[from] vision_calibration_detect::CacheError),

    /// Runtime ambiguity that the runner refuses to guess.
    #[error("ask the user: {prompt}")]
    AskUser {
        /// Field path the runtime needs help with (e.g.
        /// `"pose_convention.transform"`).
        field: String,
        /// Human-friendly prompt for the modal.
        prompt: String,
        /// Optional preset suggestions.
        suggestions: Vec<String>,
    },

    /// Insufficient features after detection — calibration would fail
    /// downstream anyway. Surfaced here so the user gets an
    /// actionable error rather than a deep optim panic.
    #[error("camera {camera_id:?}: only {usable}/{total} views had enough features")]
    InsufficientUsableViews {
        /// Camera id.
        camera_id: String,
        /// Number of views with at least the per-view feature floor
        /// (4, or the manifest's `detector.min_features_per_view`).
        usable: usize,
        /// Total number of images attempted.
        total: usize,
    },

    /// Every view of a camera was rejected, at least one of them because a
    /// plain chessboard detection could not be given the shared labels.
    ///
    /// The views of this topology share one target frame, and a plain
    /// chessboard detection labels its corners from its own top-left
    /// corner, so only a detection that reaches every edge of the board
    /// places that frame, and in a rig only one whose turn matches the
    /// view's other cameras. A target with coded corners (ChArUco,
    /// PuzzleBoard) is labelled absolutely in partial views.
    #[error(
        "camera {camera_id:?}: no view shows the whole chessboard with labels the \
         other views agree with ({partial} partial, {turned} turned by 90°, \
         {ambiguous} with an ambiguous turn, of {total}); views of this topology \
         share one target frame, which a plain chessboard places only when every \
         edge of it is detected — capture the whole board, or use a ChArUco or \
         PuzzleBoard target for partial views"
    )]
    NoWholeChessboardViews {
        /// Camera id.
        camera_id: String,
        /// Views rejected because the board was partly out of view.
        partial: usize,
        /// Views rejected because the board was seen turned by 90° where the
        /// target frame is fixed through the views.
        turned: usize,
        /// Views rejected because the board's turn relative to the view's
        /// other cameras could not be told.
        ambiguous: usize,
        /// Total number of views attempted.
        total: usize,
    },

    /// Laser images could not be aligned with target views (count
    /// mismatch under `by_index`, or a laser filename token with no
    /// matching target view).
    #[error("laser pairing failed: {message}")]
    LaserPairing {
        /// Human-readable description (names the offending camera/file).
        message: String,
    },

    /// The injected laser extractor failed on a specific image.
    #[error("laser extraction failed on {path}: {source}")]
    LaserExtraction {
        /// Offending laser image.
        path: PathBuf,
        /// Underlying error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// A camera has no usable laser observations — its laser plane
    /// would be unconstrained.
    #[error("camera {camera_id:?}: only {usable}/{total} views had >= min_points laser pixels")]
    InsufficientLaserViews {
        /// Camera id.
        camera_id: String,
        /// Number of views meeting the `min_points` bar.
        usable: usize,
        /// Total number of paired views attempted.
        total: usize,
    },

    /// The frozen upstream rig hand-eye export could not be loaded or
    /// is inconsistent with this manifest.
    #[error("upstream calibration {path}: {message}")]
    UpstreamCalibration {
        /// The export file referenced by `upstream_calibration`.
        path: PathBuf,
        /// What went wrong (parse error, camera-count mismatch, …).
        message: String,
    },

    /// The run control's token was cancelled before the input was built.
    #[error("cancelled")]
    Cancelled,
}

impl From<Cancelled> for RunError {
    fn from(_: Cancelled) -> Self {
        Self::Cancelled
    }
}

/// Counts the images a converter reads for its run control: checks the
/// token before each image and reports each image done.
pub(super) struct ImageProgress<'a> {
    control: &'a RunControl,
    report: fn(usize, usize) -> Progress,
    done: usize,
    total: usize,
}

impl<'a> ImageProgress<'a> {
    /// A counter of `total` target images.
    pub(super) fn targets(control: &'a RunControl, total: usize) -> Self {
        Self {
            control,
            report: Progress::detection,
            done: 0,
            total,
        }
    }

    /// A counter of `total` laser images.
    pub(super) fn lasers(control: &'a RunControl, total: usize) -> Self {
        Self {
            control,
            report: Progress::laser_extraction,
            done: 0,
            total,
        }
    }

    /// Read one image with `read`, unless the run was cancelled.
    pub(super) fn read<T>(
        &mut self,
        read: impl FnOnce() -> Result<T, RunError>,
    ) -> Result<T, RunError> {
        self.control.check()?;
        let value = read()?;
        self.skip();
        Ok(value)
    }

    /// Count one image as done without reading it.
    pub(super) fn skip(&mut self) {
        self.done += 1;
        self.control.report((self.report)(self.done, self.total));
    }
}

/// Why a requested corner covariance is unusable; see
/// [`RunError::CornerCovariance`].
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum CovarianceFault {
    /// `missing` of the image's `total` corners carry no covariance.
    #[error(
        "the detector reported no covariance for {missing} of {total} corners in this \
         image; on JPEG or demosaiced input the per-image noise estimate declines — \
         set `image_noise = {{ known = σ }}` (gray levels) instead"
    )]
    Missing {
        /// Corners without a covariance.
        missing: usize,
        /// Corners in the image.
        total: usize,
    },
    /// A corner's covariance is not finite and positive definite.
    #[error("a reported corner covariance is not finite and positive definite: {message}")]
    NotPositiveDefinite {
        /// The offending matrix, as reported by the core check.
        message: String,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Lower the manifest's ChESS override to the detector's own option type.
///
/// The two structs are declared in sibling crates that cannot depend on each
/// other — `vision-calibration-dataset` is a dependency-free manifest schema,
/// and `vision-calibration-detect` pulls in `image` and the detector
/// backends. This is the single place their fields are mapped, and the
/// destructuring below is exhaustive on purpose: adding a knob to the
/// manifest without wiring it through here is a compile error, not a silently
/// ignored setting.
fn lower_chess_corners(spec: ChessCornersDetectorSpec) -> ChessCornersConfig {
    let ChessCornersDetectorSpec {
        strategy,
        threshold_value,
        min_corner_strength,
        covariance,
    } = spec;
    ChessCornersConfig {
        strategy: strategy.map(|s| match s {
            CornerStrategySpec::Chess => CornerStrategy::Chess,
            CornerStrategySpec::Radon => CornerStrategy::Radon,
        }),
        threshold_value,
        min_corner_strength,
        covariance: covariance.map(|CornerCovarianceSpec { image_noise }| CornerCovariance {
            image_noise: match image_noise {
                ImageNoiseSpec::Estimate => ImageNoise::Estimate,
                ImageNoiseSpec::Known(sigma) => ImageNoise::Known(sigma),
            },
        }),
    }
}

fn target_to_detector_config(spec: &DatasetSpec) -> Result<DetectorConfig, RunError> {
    detector_config(&spec.target, spec.detector.as_ref())
}

/// Detect a manifest's calibration target in one in-memory image.
///
/// The single-image form of what the dataset converters do per image file:
/// `target` and `detector` are the manifest's `target` and optional
/// `detector` sections, lowered to the matching detector exactly as a
/// manifest run lowers them. No cache is involved and no ROI is applied:
/// the features are in `image`'s own pixel frame, with the target points in
/// metres in the target frame. An image without a usable board yields an
/// empty list, not an error.
///
/// # Errors
///
/// [`RunError::InvalidTargetConfig`] for an impossible target (an unknown
/// ChArUco dictionary, a malformed PuzzleBoard layout), checked before
/// detection; [`RunError::Detection`] when the detector fails.
pub fn detect_target(
    image: GrayImageView<'_>,
    target: &TargetSpec,
    detector: Option<&DetectorSpec>,
) -> Result<Vec<Feature>, RunError> {
    let config = detector_config(target, detector)?;
    config.detect(image).map_err(|source| RunError::Detection {
        detector: config.name().to_string(),
        path: PathBuf::from("<in-memory image>"),
        source,
    })
}

/// The detector config for `target`, with the manifest's detector override
/// applied. The one place a manifest target is lowered to a detector.
fn detector_config(
    target: &TargetSpec,
    detector: Option<&DetectorSpec>,
) -> Result<DetectorConfig, RunError> {
    // An all-default override changes nothing; leaving it out keeps the
    // cache key of a run without one.
    let chess_corners = detector
        .and_then(|d| d.chess_corners)
        .map(lower_chess_corners)
        .filter(|c| *c != ChessCornersConfig::default());
    let refuse_chess_override = |name: &'static str| -> Result<(), RunError> {
        match chess_corners {
            Some(c) if c.covariance.is_some() => {
                Err(RunError::CovarianceUnsupported { target: name })
            }
            Some(_) => Err(RunError::InvalidTargetConfig {
                message: format!(
                    "detector.chess_corners applies to chessboard and charuco targets, not {name}"
                ),
            }),
            None => Ok(()),
        }
    };
    match target {
        TargetSpec::Chessboard {
            rows,
            cols,
            square_size_m,
        } => Ok(DetectorConfig::Chessboard(ChessboardConfig {
            rows: *rows,
            cols: *cols,
            square_size_m: *square_size_m,
            chess_corners,
        })),
        TargetSpec::Charuco {
            rows,
            cols,
            square_size_m,
            marker_size_m,
            dictionary,
        } => {
            // Fail fast before touching the filesystem on a manifest typo
            // (unknown dictionary) or an impossible board whose layout
            // needs more markers than the dictionary holds.
            validate_charuco_layout(*rows, *cols, dictionary).map_err(|e| {
                RunError::InvalidTargetConfig {
                    message: e.to_string(),
                }
            })?;
            Ok(DetectorConfig::Charuco(CharucoConfig {
                rows: *rows,
                cols: *cols,
                square_size_m: *square_size_m,
                marker_size_m: *marker_size_m,
                dictionary: dictionary.clone(),
                chess_corners,
            }))
        }
        TargetSpec::Puzzleboard {
            layout,
            cell_size_m,
        } => {
            refuse_chess_override("puzzleboard")?;
            // The manifest names a layout (`"puzzle_130x130"`); the
            // detector needs explicit dimensions. Resolve here so a typo
            // fails before any image I/O.
            let (rows, cols) = parse_puzzleboard_layout(layout)
                .map_err(|message| RunError::InvalidTargetConfig { message })?;
            Ok(DetectorConfig::Puzzleboard(PuzzleboardConfig {
                rows,
                cols,
                cell_size_m: *cell_size_m,
            }))
        }
        TargetSpec::Ringgrid {
            pitch_m,
            rows,
            long_row_cols,
            marker_outer_radius_m,
            marker_inner_radius_m,
            marker_ring_width_m,
        } => {
            refuse_chess_override("ringgrid")?;
            Ok(DetectorConfig::Ringgrid(RinggridConfig {
                pitch_m: *pitch_m,
                rows: *rows,
                long_row_cols: *long_row_cols,
                marker_outer_radius_m: *marker_outer_radius_m,
                marker_inner_radius_m: *marker_inner_radius_m,
                marker_ring_width_m: *marker_ring_width_m,
            }))
        }
    }
}

/// Resolve a PuzzleBoard `layout` name to `(rows, cols)`.
///
/// Accepts `"<rows>x<cols>"` with an optional `"puzzle_"` prefix and a
/// case-insensitive separator (e.g. `"puzzle_130x130"` or `"130X130"`).
/// PuzzleBoards are parametric in their cell count, so the dimensions are
/// the only thing the detector needs; encoding them in the name keeps the
/// manifest self-describing without a separate layout registry.
fn parse_puzzleboard_layout(layout: &str) -> Result<(u32, u32), String> {
    let body = layout
        .trim()
        .strip_prefix("puzzle_")
        .unwrap_or_else(|| layout.trim());
    let (rows, cols) = body
        .split_once(['x', 'X'])
        .ok_or_else(|| format!("puzzleboard layout {layout:?} must be \"<rows>x<cols>\""))?;
    let rows: u32 = rows
        .parse()
        .map_err(|_| format!("puzzleboard layout {layout:?} has a non-numeric rows count"))?;
    let cols: u32 = cols
        .parse()
        .map_err(|_| format!("puzzleboard layout {layout:?} has a non-numeric cols count"))?;
    if rows == 0 || cols == 0 {
        return Err(format!(
            "puzzleboard layout {layout:?} must have non-zero dimensions"
        ));
    }
    Ok((rows, cols))
}

/// Each camera's image size in pixels, `[width, height]`, in camera order:
/// the size of its region (`roi_xywh`) when it has one, else the size its
/// target images share. This is the `image_sizes` argument of
/// [`assess`](crate::analysis::assess) and the other image-wide analyses.
///
/// Reads only each image's header. Relative paths in the manifest resolve
/// against `base_dir`.
///
/// # Errors
///
/// [`RunError::Validation`] for an invalid manifest,
/// [`RunError::EmptyImageMatch`] or [`RunError::BadGlob`] when a camera's
/// images cannot be listed, [`RunError::Decode`] when an image header cannot
/// be read, and [`RunError::ImageSizeMismatch`] when one camera's images
/// differ in size.
pub fn camera_image_sizes(spec: &DatasetSpec, base_dir: &Path) -> Result<Vec<[u32; 2]>, RunError> {
    vision_calibration_dataset::validate(spec)?;
    spec.cameras
        .iter()
        .map(|camera| {
            if let Some([_, _, width, height]) = camera.roi_xywh {
                return Ok([width, height]);
            }
            let images = expand_camera_images(camera, base_dir)?;
            let mut first: Option<(&PathBuf, [u32; 2])> = None;
            for path in &images {
                let (width, height) =
                    image::image_dimensions(path).map_err(|source| RunError::Decode {
                        path: path.clone(),
                        source,
                    })?;
                let size = [width, height];
                match first {
                    None => first = Some((path, size)),
                    Some((first_path, first_size)) if first_size != size => {
                        return Err(RunError::ImageSizeMismatch {
                            camera_id: camera.id.clone(),
                            first: first_path.clone(),
                            first_size,
                            path: path.clone(),
                            size,
                        });
                    }
                    Some(_) => {}
                }
            }
            first
                .map(|(_, size)| size)
                .ok_or_else(|| RunError::EmptyImageMatch {
                    camera_id: camera.id.clone(),
                    pattern: pattern_repr(&camera.images),
                    base: base_dir.to_path_buf(),
                })
        })
        .collect()
}

fn expand_camera_images(
    camera: &vision_calibration_dataset::CameraSource,
    base_dir: &Path,
) -> Result<Vec<PathBuf>, RunError> {
    expand_image_pattern(&camera.images, &camera.id, base_dir)
}

/// Expand a glob/list image pattern into natural-sorted absolute paths.
/// Shared by the target (`images`) and laser (`laser_images`) sources.
fn expand_image_pattern(
    images: &ImagePattern,
    camera_id: &str,
    base_dir: &Path,
) -> Result<Vec<PathBuf>, RunError> {
    match images {
        ImagePattern::Glob { pattern } => {
            let resolved = if Path::new(pattern).is_absolute() {
                pattern.clone()
            } else {
                base_dir.join(pattern).to_string_lossy().to_string()
            };
            let entries = glob::glob(&resolved).map_err(|e| RunError::BadGlob {
                camera_id: camera_id.to_string(),
                pattern: pattern.clone(),
                source: e,
            })?;
            let mut paths: Vec<PathBuf> = entries
                .filter_map(|r| r.ok())
                .filter(|p| p.is_file())
                .collect();
            // Natural sort keeps `Im_2` before `Im_10` so `by_index`
            // pairing matches the acquisition order of non-zero-padded
            // filename schemes.
            paths.sort_by(|a, b| natural_cmp(&a.to_string_lossy(), &b.to_string_lossy()));
            Ok(paths)
        }
        ImagePattern::List { paths } => {
            let mut resolved: Vec<PathBuf> = Vec::with_capacity(paths.len());
            for p in paths {
                let abs: PathBuf = if p.is_absolute() {
                    p.clone()
                } else {
                    base_dir.join(p)
                };
                resolved.push(abs);
            }
            Ok(resolved)
        }
    }
}

/// Natural (numeric-aware) string comparison: splits each string into
/// runs of digits and non-digits, comparing digit runs by numeric
/// value. Keeps `Im_2` < `Im_10`.
fn natural_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let mut ai = a.chars().peekable();
    let mut bi = b.chars().peekable();
    loop {
        match (ai.peek().copied(), bi.peek().copied()) {
            (None, None) => return Ordering::Equal,
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (Some(ca), Some(cb)) => {
                if ca.is_ascii_digit() && cb.is_ascii_digit() {
                    let na: String = take_digits(&mut ai);
                    let nb: String = take_digits(&mut bi);
                    // Compare by numeric value; fall back to string
                    // length then lexically for runs too long for u128.
                    let cmp = match (na.parse::<u128>(), nb.parse::<u128>()) {
                        (Ok(x), Ok(y)) => x.cmp(&y),
                        _ => na.len().cmp(&nb.len()).then_with(|| na.cmp(&nb)),
                    };
                    if cmp != Ordering::Equal {
                        return cmp;
                    }
                } else {
                    let cmp = ca.cmp(&cb);
                    if cmp != Ordering::Equal {
                        return cmp;
                    }
                    ai.next();
                    bi.next();
                }
            }
        }
    }
}

fn take_digits(it: &mut std::iter::Peekable<std::str::Chars<'_>>) -> String {
    let mut s = String::new();
    while let Some(&c) = it.peek() {
        if c.is_ascii_digit() {
            s.push(c);
            it.next();
        } else {
            break;
        }
    }
    s
}

fn pattern_repr(p: &ImagePattern) -> String {
    match p {
        ImagePattern::Glob { pattern } => pattern.clone(),
        ImagePattern::List { paths } => format!("<list of {} paths>", paths.len()),
    }
}

/// Run one image through the cache-or-detect path shared by every
/// converter: read bytes → cache lookup → on miss, decode, crop to
/// ROI, detect in the camera pixel frame, store.
///
/// Returns the features plus whether they came from the cache. When the
/// detector config requests corner covariance, cached and fresh features
/// alike are checked by [`check_corner_covariances`].
fn detect_features(
    config: &DetectorConfig,
    roi: Option<[u32; 4]>,
    image_path: &Path,
    detection: Detection<'_>,
) -> Result<(Vec<Feature>, bool), RunError> {
    let (features, cache_hit) = cached_or_detected(config, roi, image_path, detection)?;
    if config.requests_covariance() {
        check_corner_covariances(&features, config.name(), image_path)?;
    }
    Ok((features, cache_hit))
}

/// Every corner of an image must carry a covariance that the calibration can
/// whiten with (finite, positive definite). No corner is dropped to make a
/// view pass: one corner without a covariance would otherwise force the whole
/// input back to unweighted, or be silently unweighted among weighted ones.
fn check_corner_covariances(
    features: &[Feature],
    detector: &str,
    image_path: &Path,
) -> Result<(), RunError> {
    let fault = |fault| RunError::CornerCovariance {
        detector: detector.to_string(),
        path: image_path.to_path_buf(),
        fault,
    };
    let missing = features.iter().filter(|f| f.cov_px.is_none()).count();
    if missing > 0 {
        return Err(fault(CovarianceFault::Missing {
            missing,
            total: features.len(),
        }));
    }
    for [xx, xy, yy] in features.iter().filter_map(|f| f.cov_px) {
        PointCovariance::new(xx, xy, yy).map_err(|e| {
            fault(CovarianceFault::NotPositiveDefinite {
                message: e.to_string(),
            })
        })?;
    }
    Ok(())
}

/// How the runner obtains each image's features.
#[derive(Clone, Copy)]
pub enum Detection<'a> {
    /// Reuse the cached features of an image the cache has seen with the
    /// same detector and config; detect and store the others.
    Cached(&'a dyn DetectionCache),
    /// Detect every image again and overwrite its cache entry: for a
    /// detector change the cache key does not capture.
    Refresh(&'a dyn DetectionCache),
    /// Detect every image; read and write no cache.
    Uncached,
}

impl Detection<'_> {
    /// The cached features for `key`, when this mode reads the cache.
    fn lookup(&self, key: &CacheKey) -> Result<Option<CachedFeatures>, CacheError> {
        match self {
            Detection::Cached(cache) => cache.get(key),
            Detection::Refresh(_) | Detection::Uncached => Ok(None),
        }
    }

    /// Store freshly detected features, when this mode writes the cache.
    fn store(&self, key: &CacheKey, features: &CachedFeatures) -> Result<(), CacheError> {
        match self {
            Detection::Cached(cache) | Detection::Refresh(cache) => cache.put(key, features),
            Detection::Uncached => Ok(()),
        }
    }
}

impl std::fmt::Debug for Detection<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Detection::Cached(_) => "Detection::Cached",
            Detection::Refresh(_) => "Detection::Refresh",
            Detection::Uncached => "Detection::Uncached",
        })
    }
}

fn cached_or_detected(
    config: &DetectorConfig,
    roi: Option<[u32; 4]>,
    image_path: &Path,
    detection: Detection<'_>,
) -> Result<(Vec<Feature>, bool), RunError> {
    let bytes = std::fs::read(image_path)?;
    let key = CacheKey::for_detection(&bytes, config, roi);

    let cached: Option<CachedFeatures> = detection.lookup(&key)?;
    if let Some(entry) = cached {
        return Ok((entry.features, true));
    }

    let gray = decode_gray(&bytes, image_path, roi)?;
    let view = GrayImageView::new(gray.width(), gray.height(), gray.as_raw())
        .expect("an image buffer holds width × height bytes");
    let detected = config.detect(view).map_err(|e| RunError::Detection {
        detector: config.name().to_string(),
        path: image_path.to_path_buf(),
        source: e,
    })?;
    detection.store(
        &key,
        &CachedFeatures {
            features: detected.clone(),
        },
    )?;
    Ok((detected, false))
}

/// Decode an image file's bytes to 8-bit grayscale, cropped to `roi`
/// (`[x, y, width, height]`) when given: the frame detectors and laser
/// extractors read, whose pixel coordinates are ROI-local.
pub(crate) fn decode_gray(
    bytes: &[u8],
    image_path: &Path,
    roi: Option<[u32; 4]>,
) -> Result<image::GrayImage, RunError> {
    let img = image::load_from_memory(bytes).map_err(|e| RunError::Decode {
        path: image_path.to_path_buf(),
        source: e,
    })?;
    let img = match roi {
        Some([x, y, w, h]) => img.crop_imm(x, y, w, h),
        None => img,
    };
    Ok(img.to_luma8())
}

fn features_to_view(features: &[Feature]) -> Result<View<NoMeta>, RunError> {
    Ok(View::without_meta(features_to_obs(features)))
}

/// The 2D-3D correspondences of one view, with each point's covariance when
/// the features carry them.
///
/// Covariances are all-or-none per view: [`detect_features`] rejects an image
/// where a requested covariance is missing or unusable, so the features here
/// either all carry a valid one or (the default) none does.
fn features_to_obs(features: &[Feature]) -> CorrespondenceView {
    let mut points_3d = Vec::with_capacity(features.len());
    let mut points_2d = Vec::with_capacity(features.len());
    for f in features {
        points_3d.push(Pt3::new(f.world_xyz[0], f.world_xyz[1], f.world_xyz[2]));
        points_2d.push(Pt2::new(f.image_xy[0], f.image_xy[1]));
    }
    let covariances: Option<Vec<PointCovariance>> = features
        .iter()
        .map(|f| {
            f.cov_px.map(|[xx, xy, yy]| {
                PointCovariance::new(xx, xy, yy).expect(
                    "covariances are validated by check_corner_covariances in detect_features",
                )
            })
        })
        .collect();
    let view = match covariances {
        Some(cov) if !cov.is_empty() => {
            CorrespondenceView::new_with_covariances(points_3d, points_2d, cov)
        }
        _ => CorrespondenceView::new(points_3d, points_2d),
    };
    view.expect(
        "Feature vectors are non-empty and equal-length by construction; \
         CorrespondenceView::new only fails on mismatched lengths.",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use vision_calibration_dataset::{
        CameraSource, ChessCornersDetectorSpec, DetectorSpec, ImagePattern,
    };

    fn spec_for_target(target: TargetSpec) -> DatasetSpec {
        DatasetSpec {
            version: 1,
            cameras: vec![CameraSource {
                id: "cam0".into(),
                images: ImagePattern::List { paths: vec![] },
                roi_xywh: None,
                laser_images: None,
            }],
            target,
            detector: None,
            robot_poses: None,
            laser: None,
            upstream_calibration: None,
            topology: Topology::PlanarIntrinsics,
            pose_pairing: None,
            pose_convention: None,
            unresolved: vec![],
            description: None,
        }
    }

    #[test]
    fn charuco_target_maps_to_detector_config() {
        // 12×12 needs 72 markers, so the dictionary must hold at least
        // that many (DICT_4X4_50 would be rejected — see the capacity test).
        let target = TargetSpec::Charuco {
            rows: 12,
            cols: 12,
            square_size_m: 0.020,
            marker_size_m: 0.015,
            dictionary: "DICT_4X4_100".into(),
        };
        let spec = spec_for_target(target);
        let DetectorConfig::Charuco(config) = target_to_detector_config(&spec).unwrap() else {
            panic!("a charuco target lowers to the charuco detector");
        };
        assert_eq!(config.dictionary, "DICT_4X4_100");
        assert_eq!(config.marker_size_m, 0.015);
    }

    /// A synthetic chessboard: `squares_x` × `squares_y` squares of `px`
    /// pixels on a white margin of one square.
    fn chessboard_image(squares_x: u32, squares_y: u32, px: u32) -> image::GrayImage {
        let (w, h) = ((squares_x + 2) * px, (squares_y + 2) * px);
        image::GrayImage::from_fn(w, h, |x, y| {
            let (cx, cy) = (x / px, y / px);
            let inside = (1..=squares_x).contains(&cx) && (1..=squares_y).contains(&cy);
            image::Luma([if inside && (cx + cy) % 2 == 0 { 0 } else { 255 }])
        })
    }

    fn view(image: &image::GrayImage) -> GrayImageView<'_> {
        GrayImageView::new(image.width(), image.height(), image.as_raw()).unwrap()
    }

    #[test]
    fn detect_target_finds_the_inner_corners_of_a_synthetic_board() {
        // 8×6 squares have 7×5 inner corners.
        let image = chessboard_image(8, 6, 40);
        let target = TargetSpec::Chessboard {
            rows: 5,
            cols: 7,
            square_size_m: 0.03,
        };
        let features = detect_target(view(&image), &target, None).unwrap();
        assert_eq!(features.len(), 35);
        for f in &features {
            assert_eq!(f.world_xyz[2], 0.0);
            // Board coordinates are multiples of the square size.
            for v in &f.world_xyz[..2] {
                let k = v / 0.03;
                assert!((k - k.round()).abs() < 1e-9, "{v} is off the grid");
            }
            // Corners sit on the square lattice of the image.
            for v in f.image_xy {
                let k = v / 40.0;
                assert!((k - k.round()).abs() < 0.05, "{v} is off the lattice");
            }
        }
    }

    #[test]
    fn detect_target_on_a_blank_image_is_empty() {
        let image = image::GrayImage::from_pixel(200, 150, image::Luma([128]));
        let target = TargetSpec::Chessboard {
            rows: 5,
            cols: 7,
            square_size_m: 0.03,
        };
        assert!(
            detect_target(view(&image), &target, None)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn detect_target_rejects_an_impossible_target_before_detection() {
        let image = chessboard_image(8, 6, 40);
        let target = TargetSpec::Charuco {
            rows: 12,
            cols: 12,
            square_size_m: 0.020,
            marker_size_m: 0.015,
            dictionary: "DICT_TYPO_99".into(),
        };
        let err = detect_target(view(&image), &target, None).unwrap_err();
        assert!(
            matches!(err, RunError::InvalidTargetConfig { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn charuco_bad_dictionary_fails_before_io() {
        let target = TargetSpec::Charuco {
            rows: 12,
            cols: 12,
            square_size_m: 0.020,
            marker_size_m: 0.015,
            dictionary: "DICT_TYPO_99".into(),
        };
        let spec = spec_for_target(target);
        let err = target_to_detector_config(&spec).unwrap_err();
        match err {
            RunError::InvalidTargetConfig { message } => {
                assert!(message.contains("DICT_TYPO_99"), "got: {message}");
            }
            other => panic!("expected InvalidTargetConfig, got {other:?}"),
        }
    }

    #[test]
    fn charuco_dictionary_too_small_for_board_fails_before_io() {
        // A 12×12 board needs 72 markers in the OpenCV layout; DICT_4X4_50
        // only holds 50, so the manifest is impossible and must be
        // rejected up front rather than reaching a guaranteed-empty detect.
        let target = TargetSpec::Charuco {
            rows: 12,
            cols: 12,
            square_size_m: 0.020,
            marker_size_m: 0.015,
            dictionary: "DICT_4X4_50".into(),
        };
        let spec = spec_for_target(target);
        let err = target_to_detector_config(&spec).unwrap_err();
        match err {
            RunError::InvalidTargetConfig { message } => {
                assert!(
                    message.contains("72") && message.contains("50"),
                    "expected a needs-72/has-50 capacity error, got: {message}"
                );
            }
            other => panic!("expected InvalidTargetConfig, got {other:?}"),
        }
    }

    #[test]
    fn detector_override_maps_to_chess_corners_config() {
        let mut spec = spec_for_target(TargetSpec::Charuco {
            rows: 8,
            cols: 8,
            square_size_m: 0.020,
            marker_size_m: 0.015,
            dictionary: "DICT_4X4_50".into(),
        });
        spec.detector = Some(DetectorSpec {
            chess_corners: Some(ChessCornersDetectorSpec {
                strategy: Some(CornerStrategySpec::Radon),
                threshold_value: Some(0.2),
                covariance: Some(CornerCovarianceSpec {
                    image_noise: ImageNoiseSpec::Known(2.0),
                }),
                ..ChessCornersDetectorSpec::default()
            }),
            min_features_per_view: None,
        });
        let DetectorConfig::Charuco(config) = target_to_detector_config(&spec).unwrap() else {
            panic!("a charuco target lowers to the charuco detector");
        };
        // `lower_chess_corners` destructures the manifest struct exhaustively,
        // so a new manifest knob that is not wired through fails to compile.
        let lowered = config.chess_corners.expect("the override is attached");
        assert_eq!(lowered.strategy, Some(CornerStrategy::Radon));
        assert_eq!(lowered.threshold_value, Some(0.2));
        assert_eq!(
            lowered.covariance,
            Some(CornerCovariance {
                image_noise: ImageNoise::Known(2.0)
            })
        );
        assert_eq!(
            lower_chess_corners(ChessCornersDetectorSpec {
                covariance: Some(CornerCovarianceSpec::default()),
                ..ChessCornersDetectorSpec::default()
            })
            .covariance,
            Some(CornerCovariance {
                image_noise: ImageNoise::Estimate
            })
        );
        assert_eq!(
            lowered,
            lower_chess_corners(spec.detector.unwrap().chess_corners.unwrap())
        );
    }

    #[test]
    fn default_detector_override_is_not_attached() {
        // An all-`None` override must not inject an empty `chess_corners`
        // object: that would change the detection-cache key for a manifest
        // that asked for nothing.
        let mut spec = spec_for_target(TargetSpec::Chessboard {
            rows: 9,
            cols: 6,
            square_size_m: 0.025,
        });
        spec.detector = Some(DetectorSpec::default());
        let DetectorConfig::Chessboard(config) = target_to_detector_config(&spec).unwrap() else {
            panic!("a chessboard target lowers to the chessboard detector");
        };
        assert!(config.chess_corners.is_none());
    }

    #[test]
    fn puzzleboard_target_maps_to_detector_config() {
        let target = TargetSpec::Puzzleboard {
            layout: "puzzle_130x130".into(),
            cell_size_m: 0.0052,
        };
        let spec = spec_for_target(target);
        let config = target_to_detector_config(&spec).unwrap();
        assert_eq!(
            config,
            DetectorConfig::Puzzleboard(PuzzleboardConfig {
                rows: 130,
                cols: 130,
                cell_size_m: 0.0052,
            })
        );
    }

    #[test]
    fn puzzleboard_bad_layout_fails_before_io() {
        let target = TargetSpec::Puzzleboard {
            layout: "not-a-grid".into(),
            cell_size_m: 0.0052,
        };
        let spec = spec_for_target(target);
        let err = target_to_detector_config(&spec).unwrap_err();
        match err {
            RunError::InvalidTargetConfig { message } => {
                assert!(message.contains("not-a-grid"), "got: {message}");
            }
            other => panic!("expected InvalidTargetConfig, got {other:?}"),
        }
    }

    #[test]
    fn ringgrid_target_maps_to_detector_config() {
        let target = TargetSpec::Ringgrid {
            pitch_m: 0.02,
            rows: 5,
            long_row_cols: 5,
            marker_outer_radius_m: 0.006,
            marker_inner_radius_m: 0.003,
            marker_ring_width_m: 0.001,
        };
        let spec = spec_for_target(target);
        let DetectorConfig::Ringgrid(config) = target_to_detector_config(&spec).unwrap() else {
            panic!("a ringgrid target lowers to the ringgrid detector");
        };
        assert_eq!(config.long_row_cols, 5);
        assert_eq!(config.marker_outer_radius_m, 0.006);
    }

    #[test]
    fn parse_puzzleboard_layout_accepts_known_forms() {
        assert_eq!(
            parse_puzzleboard_layout("puzzle_130x130").unwrap(),
            (130, 130)
        );
        assert_eq!(parse_puzzleboard_layout("130x130").unwrap(), (130, 130));
        assert_eq!(parse_puzzleboard_layout("12X9").unwrap(), (12, 9));
        assert_eq!(parse_puzzleboard_layout("  puzzle_8x8 ").unwrap(), (8, 8));
        assert!(parse_puzzleboard_layout("puzzle_0x10").is_err());
        assert!(parse_puzzleboard_layout("puzzle_10").is_err());
        assert!(parse_puzzleboard_layout("puzzle_axb").is_err());
    }

    #[test]
    fn natural_sort_orders_numeric_runs() {
        let mut names = vec!["Im_10.png", "Im_2.png", "Im_1.png", "Im_20.png"];
        names.sort_by(|a, b| natural_cmp(a, b));
        assert_eq!(
            names,
            vec!["Im_1.png", "Im_2.png", "Im_10.png", "Im_20.png"]
        );
    }

    #[test]
    fn the_roi_is_part_of_the_cache_key() {
        // Two ROIs over the same image with the same detector config
        // must hash to different cache keys; otherwise a user editing
        // an ROI would silently re-use stale detections.
        let config = DetectorConfig::Chessboard(ChessboardConfig {
            rows: 9,
            cols: 6,
            square_size_m: 0.025,
            chess_corners: None,
        });
        let key_no_roi = CacheKey::for_detection(b"img", &config, None);
        let key_roi_a = CacheKey::for_detection(b"img", &config, Some([10, 20, 100, 100]));
        let key_roi_b = CacheKey::for_detection(b"img", &config, Some([10, 20, 200, 100]));
        assert_ne!(key_no_roi, key_roi_a);
        assert_ne!(key_roi_a, key_roi_b);
        assert_ne!(key_no_roi, key_roi_b);
    }

    /// A one-camera planar spec over `paths`, with `roi`.
    fn sized_spec(paths: Vec<PathBuf>, roi: Option<[u32; 4]>) -> DatasetSpec {
        let mut spec = spec_for_target(TargetSpec::Chessboard {
            rows: 5,
            cols: 4,
            square_size_m: 0.03,
        });
        spec.cameras[0].images = ImagePattern::List { paths };
        spec.cameras[0].roi_xywh = roi;
        spec
    }

    fn write_blank(dir: &Path, name: &str, width: u32, height: u32) -> PathBuf {
        let path = dir.join(name);
        image::GrayImage::new(width, height).save(&path).unwrap();
        path
    }

    #[test]
    fn camera_image_sizes_reads_the_shared_size_or_the_region() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = vec![
            write_blank(tmp.path(), "a.png", 64, 48),
            write_blank(tmp.path(), "b.png", 64, 48),
        ];
        let spec = sized_spec(paths.clone(), None);
        assert_eq!(
            camera_image_sizes(&spec, tmp.path()).unwrap(),
            vec![[64, 48]]
        );

        // A region is the camera's image, whatever the files' size.
        let spec = sized_spec(paths, Some([8, 4, 32, 24]));
        assert_eq!(
            camera_image_sizes(&spec, tmp.path()).unwrap(),
            vec![[32, 24]]
        );
    }

    #[test]
    fn camera_image_sizes_rejects_a_camera_with_mixed_sizes() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = vec![
            write_blank(tmp.path(), "a.png", 64, 48),
            write_blank(tmp.path(), "b.png", 48, 64),
        ];
        let err = camera_image_sizes(&sized_spec(paths, None), tmp.path()).unwrap_err();
        let RunError::ImageSizeMismatch {
            camera_id,
            first_size,
            size,
            path,
            ..
        } = err
        else {
            panic!("expected ImageSizeMismatch, got {err:?}");
        };
        assert_eq!(camera_id, "cam0");
        assert_eq!((first_size, size), ([64, 48], [48, 64]));
        assert!(path.ends_with("b.png"));
    }

    #[test]
    fn camera_image_sizes_names_a_camera_without_images() {
        let tmp = tempfile::tempdir().unwrap();
        let mut spec = sized_spec(vec![], None);
        spec.cameras[0].images = ImagePattern::Glob {
            pattern: "missing/*.png".into(),
        };
        let err = camera_image_sizes(&spec, tmp.path()).unwrap_err();
        assert!(matches!(err, RunError::EmptyImageMatch { .. }), "{err:?}");
    }
}
