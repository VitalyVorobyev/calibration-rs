//! Dataset-driven runner: takes a
//! [`DatasetSpec`] manifest plus a
//! per-problem `*Config` and produces the existing `*Input` IR by
//! running detection (cached) on the per-camera images.
//!
//! [`calibrate_dataset`] runs a manifest end to end: it picks the converter
//! and the problem type by the manifest's topology and returns the export as
//! a [`CalibrationExport`]; [`calibrate_dataset_with_seed`] runs it from a
//! [`DeviceSeed`](crate::device_seed::DeviceSeed), a previous calibration
//! of the device or its datasheet.
//! [`camera_image_sizes`] reads each camera's image size, which the
//! image-wide analyses take; [`input_files`] lists every file a run reads.
//! [`detection_detail`] returns what a run's detection of one image looked
//! like, for review: the per-feature overlay data, the candidates left out
//! and the image's outcome.
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
    Cancelled, CorrespondenceView, NoMeta, PointBlur, PointCovariance, Progress, Pt2, Pt3,
    RunControl, View, lexical_absolute,
};
use vision_calibration_dataset::{
    ChessCornersDetectorSpec, CornerCovarianceSpec, CornerStrategySpec, DatasetSpec, DetectorSpec,
    ImageNoiseSpec, ImagePattern, ProblemKind, RinggridCenterSpec, RinggridCodingSpec,
    RinggridLatticeSpec, TargetSpec, ValidationError,
};
use vision_calibration_detect::{
    CacheError, CacheKey, CachedFeatures, CharucoConfig, ChessCornersConfig, ChessboardConfig,
    CornerCovariance, CornerStrategy, DetectionCache, DetectionDetail, DetectorConfig, Feature,
    GrayImageView, ImageNoise, PuzzleboardConfig, RinggridCenterMethod, RinggridCoding,
    RinggridConfig, RinggridLattice, validate_charuco_layout,
};

mod board_turn;
mod calibrate;
mod config_patch;
mod convert_options;
mod handeye;
mod laser;
mod pairing;
mod planar;
mod poses;
mod rig;
mod seed;
mod view_filter;

pub(crate) use calibrate::calibrate_dataset_with;
pub use calibrate::{
    CalibrateDatasetError, CalibrationConfig, CalibrationExport, DatasetCalibration,
    calibrate_dataset, calibrate_dataset_with_seed,
};
pub use config_patch::ConfigPatch;
pub(crate) use convert_options::ConvertOptions;
pub use handeye::{HandeyeRunResult, build_single_cam_handeye_input};
pub use laser::{
    LaserPixelExtractor, LaserlineRunResult, RigHandeyeLaserlineRunResult, RigLaserlineRunResult,
    build_laserline_device_input, build_rig_handeye_laserline_input,
    build_rig_laserline_device_input,
};
pub use pairing::{PairedViews, PoseRow, UnpairedLaserFrame, ViewLaserFrames, ViewPoses};
pub(crate) use pairing::{list_laser_frames, list_view_poses, pair_dataset_views};
pub use planar::{PlanarRunResult, build_planar_input};
pub use rig::{RigRunResult, build_rig_extrinsics_input, build_rig_handeye_input};
pub use view_filter::{RejectReason, RejectedDetection};

/// Errors produced by the dataset-driven runner.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RunError {
    /// Manifest failed structural validation. Wraps the underlying
    /// [`ValidationError`].
    #[error("manifest validation failed: {0}")]
    Validation(#[from] ValidationError),

    /// ProblemKind is not supported by this converter (e.g. a rig manifest
    /// handed to the planar converter, or a topology whose converter
    /// has not shipped yet).
    #[error("topology {topology:?} is not supported by this converter")]
    UnsupportedTopology {
        /// The offending topology.
        topology: ProblemKind,
    },

    /// A manifest option has a value this version does not support (one
    /// that a newer manifest format added).
    #[error("{option} {value} is not supported by this version")]
    UnsupportedOption {
        /// The manifest option.
        option: &'static str,
        /// Its value, as the manifest states it.
        value: String,
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

    /// A camera's image pattern is not a valid glob.
    #[error("camera {camera_id:?}: invalid glob pattern {pattern:?} ({message})")]
    BadGlob {
        /// The offending camera id.
        camera_id: String,
        /// The pattern that failed to parse.
        pattern: String,
        /// Why the pattern is invalid.
        message: String,
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
    #[error("decoding image {path}: {message}")]
    Decode {
        /// File that failed to decode.
        path: PathBuf,
        /// What the decoder reported.
        message: String,
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

    /// A camera's region of interest (`roi_xywh`) does not fit inside its
    /// images.
    #[error(
        "{path} is {size:?} pixels, but its region of interest [x, y, width, height] \
         = {roi:?} runs past it"
    )]
    RoiOutOfBounds {
        /// The image.
        path: PathBuf,
        /// Its `[width, height]` in pixels.
        size: [u32; 2],
        /// The region, `[x, y, width, height]`.
        roi: [u32; 4],
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

    /// A view of a rig laser run matches no view of the upstream
    /// calibration, or more than one, by its images (path or path within
    /// the dataset folder) or its pairing token, as the upstream export's
    /// image manifest records them. No view matching means the upstream run
    /// left that view out, so it has no board pose for it.
    #[error(
        "upstream calibration {path}: view {token:?} matches {matches} of its views \
         (by image or pairing token); each laser view must be exactly one view of the \
         upstream run"
    )]
    UpstreamViewMatch {
        /// The export file referenced by `upstream_calibration`.
        path: PathBuf,
        /// The view's pairing token (its index under `by_index` pairing).
        token: String,
        /// How many upstream views it matches: 0, or 2 and more.
        matches: usize,
    },

    /// The run control's token was cancelled before the input was built.
    #[error("cancelled")]
    Cancelled,

    /// An image named by camera and path is not one of the manifest's: the
    /// camera index is out of range, or the path is not among that camera's
    /// target images.
    #[error("{path} is not a target image of camera {camera} of the manifest")]
    ImageNotInDataset {
        /// Index of the camera in the manifest's `cameras`.
        camera: usize,
        /// The image, as given.
        path: PathBuf,
    },

    /// The cached features of an image differ from a fresh detection of it
    /// with the same detector and config, so its detail would show a
    /// detection the run did not use. Detect the dataset again
    /// ([`Detection::Refresh`]) to bring the cache up to date.
    #[error(
        "the cached features of {path} differ from a fresh detection; \
         refresh the detection cache"
    )]
    DetailMismatch {
        /// The image.
        path: PathBuf,
    },
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
         set `image_noise = {{ stated = σ }}` (gray levels) instead"
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
/// backends. This is the single place their fields are mapped. The manifest
/// types can grow, so the destructuring below cannot be exhaustive; the
/// `lowering_covers_every_manifest_field` test fails instead when a knob is
/// added to the manifest without being wired through here.
fn lower_chess_corners(spec: ChessCornersDetectorSpec) -> Result<ChessCornersConfig, RunError> {
    let ChessCornersDetectorSpec {
        strategy,
        threshold_value,
        min_corner_strength,
        covariance,
        ..
    } = spec;
    let unsupported =
        |option: &'static str, value: String| RunError::UnsupportedOption { option, value };
    let strategy = strategy
        .map(|s| match s {
            CornerStrategySpec::Chess => Ok(CornerStrategy::Chess),
            CornerStrategySpec::Radon => Ok(CornerStrategy::Radon),
            other => Err(unsupported(
                "detector.chess_corners.strategy",
                format!("{other:?}"),
            )),
        })
        .transpose()?;
    let covariance = covariance
        .map(|CornerCovarianceSpec { image_noise, .. }| {
            let image_noise = match image_noise {
                ImageNoiseSpec::Estimate => ImageNoise::Estimate,
                ImageNoiseSpec::Stated(sigma) => ImageNoise::Stated(sigma),
                other => {
                    return Err(unsupported(
                        "detector.chess_corners.covariance.image_noise",
                        format!("{other:?}"),
                    ));
                }
            };
            Ok(CornerCovariance::new(image_noise))
        })
        .transpose()?;
    let mut config = ChessCornersConfig::default();
    config.strategy = strategy;
    config.threshold_value = threshold_value;
    config.min_corner_strength = min_corner_strength;
    config.covariance = covariance;
    Ok(config)
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

/// [`detect_target`], with the detection's detail: the target-specific
/// overlay data of each feature, the candidates the detector left out with
/// their reasons, and the image's outcome. `detail.features` equals what
/// [`detect_target`] returns for the same image.
///
/// # Errors
///
/// As [`detect_target`].
pub fn detect_target_detail(
    image: GrayImageView<'_>,
    target: &TargetSpec,
    detector: Option<&DetectorSpec>,
) -> Result<DetectionDetail, RunError> {
    let config = detector_config(target, detector)?;
    config
        .detect_detailed(image)
        .map_err(|source| RunError::Detection {
            detector: config.name().to_string(),
            path: PathBuf::from("<in-memory image>"),
            source,
        })
}

/// The detail of the detection a manifest run makes of one image: camera
/// `camera`'s target image `image` (relative paths resolve against
/// `base_dir`), detected with the detector, region of interest and cache key
/// the converters use, so its features are the ones a run of `spec` with
/// `detection` reads.
///
/// With a cache ([`Detection::Cached`]) a stored detail is returned when its
/// features equal the image's cached features. Otherwise the image is
/// detected, its features stored when the cache had none, and its detail
/// stored beside them ([`DetectionCache::put_detail`]). [`Detection::Refresh`]
/// detects again and overwrites both; [`Detection::Uncached`] reads and
/// writes no cache. A calibration never reads the detail.
///
/// The detail is in the pixels of the camera's region of interest when the
/// manifest sets one, as the features are.
///
/// # Errors
///
/// [`RunError::Validation`] for an invalid manifest;
/// [`RunError::ImageNotInDataset`] for a camera index out of range or an
/// image not among that camera's; I/O, decode and detection errors as a run
/// reports them; [`RunError::DetailMismatch`] when the image's cached
/// features differ from a fresh detection of it.
pub fn detection_detail(
    spec: &DatasetSpec,
    base_dir: &Path,
    camera: usize,
    image: &Path,
    detection: Detection<'_>,
) -> Result<DetectionDetail, RunError> {
    let config = target_to_detector_config(spec)?;
    vision_calibration_dataset::validate(spec)?;
    let not_in_dataset = || RunError::ImageNotInDataset {
        camera,
        path: image.to_path_buf(),
    };
    let source = spec.cameras.get(camera).ok_or_else(not_in_dataset)?;
    let wanted = lexical_absolute(&base_dir.join(image));
    let path = expand_camera_images(source, base_dir)?
        .into_iter()
        .find(|path| lexical_absolute(path) == wanted)
        .ok_or_else(not_in_dataset)?;
    cached_or_detected_detail(&config, source.roi_xywh, &path, detection)
}

/// [`cached_or_detected`] for the detail: the stored detail when it matches
/// the stored features, else a fresh one, checked against the stored
/// features and stored.
fn cached_or_detected_detail(
    config: &DetectorConfig,
    roi: Option<[u32; 4]>,
    image_path: &Path,
    detection: Detection<'_>,
) -> Result<DetectionDetail, RunError> {
    let bytes = std::fs::read(image_path)?;
    let key = CacheKey::for_detection(&bytes, config, roi);
    let cached = detection.lookup(&key)?;
    if let Some(entry) = &cached
        && let Some(detail) = detection.lookup_detail(&key)?
        && detail.features == entry.features
    {
        return Ok(detail);
    }

    let gray = decode_gray(&bytes, image_path, roi)?;
    let view = GrayImageView::new(gray.width(), gray.height(), gray.as_raw())
        .expect("an image buffer holds width × height bytes");
    let detail = config
        .detect_detailed(view)
        .map_err(|source| RunError::Detection {
            detector: config.name().to_string(),
            path: image_path.to_path_buf(),
            source,
        })?;
    match cached {
        Some(entry) if entry.features != detail.features => {
            return Err(RunError::DetailMismatch {
                path: image_path.to_path_buf(),
            });
        }
        Some(_) => {}
        None => detection.store(&key, &CachedFeatures::new(detail.features.clone()))?,
    }
    detection.store_detail(&key, &detail)?;
    Ok(detail)
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
        .transpose()?
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
            measure_blur,
            ..
        } => {
            let mut config = ChessboardConfig::new(*rows, *cols, *square_size_m)
                .with_measure_blur(*measure_blur);
            config.chess_corners = chess_corners;
            Ok(DetectorConfig::Chessboard(config))
        }
        TargetSpec::Charuco {
            rows,
            cols,
            square_size_m,
            marker_size_m,
            dictionary,
            ..
        } => {
            // Fail fast before touching the filesystem on a manifest typo
            // (unknown dictionary) or an impossible board whose layout
            // needs more markers than the dictionary holds.
            validate_charuco_layout(*rows, *cols, dictionary).map_err(|e| {
                RunError::InvalidTargetConfig {
                    message: e.to_string(),
                }
            })?;
            let mut config = CharucoConfig::new(
                *rows,
                *cols,
                *square_size_m,
                *marker_size_m,
                dictionary.clone(),
            );
            config.chess_corners = chess_corners;
            Ok(DetectorConfig::Charuco(config))
        }
        TargetSpec::Puzzleboard {
            layout,
            cell_size_m,
            measure_blur,
            ..
        } => {
            refuse_chess_override("puzzleboard")?;
            // The manifest names a layout (`"puzzle_130x130"`); the
            // detector needs explicit dimensions. Resolve here so a typo
            // fails before any image I/O.
            let (rows, cols) = parse_puzzleboard_layout(layout)
                .map_err(|message| RunError::InvalidTargetConfig { message })?;
            Ok(DetectorConfig::Puzzleboard(
                PuzzleboardConfig::new(rows, cols, *cell_size_m).with_measure_blur(*measure_blur),
            ))
        }
        TargetSpec::Ringgrid {
            pitch_m,
            rows,
            long_row_cols,
            marker_outer_radius_m,
            marker_inner_radius_m,
            marker_ring_width_m,
            center_method,
            lattice,
            coding,
            min_confidence,
            ..
        } => {
            refuse_chess_override("ringgrid")?;
            if let Some(c) = min_confidence
                && !(c.is_finite() && (0.0..=1.0).contains(c))
            {
                return Err(RunError::InvalidTargetConfig {
                    message: format!("target.min_confidence must be finite and in [0, 1], got {c}"),
                });
            }
            let center_method = match center_method {
                RinggridCenterSpec::Ellipse => RinggridCenterMethod::Ellipse,
                RinggridCenterSpec::Photometric => RinggridCenterMethod::Photometric,
                other => {
                    return Err(RunError::UnsupportedOption {
                        option: "target.center_method",
                        value: format!("{other:?}"),
                    });
                }
            };
            let lattice = match lattice {
                RinggridLatticeSpec::Hex => RinggridLattice::Hex,
                RinggridLatticeSpec::Rect => RinggridLattice::Rect,
                other => {
                    return Err(RunError::UnsupportedOption {
                        option: "target.lattice",
                        value: format!("{other:?}"),
                    });
                }
            };
            let coding = match coding {
                RinggridCodingSpec::Coded => RinggridCoding::Coded,
                RinggridCodingSpec::Plain { origin_dots: true } => {
                    RinggridCoding::Plain { origin_dots: true }
                }
                // Its views would carry no board coordinates: refuse it
                // here rather than drop every view after detection.
                RinggridCodingSpec::Plain { origin_dots: false } => {
                    return Err(RunError::InvalidTargetConfig {
                        message: "a plain ring grid without origin dots \
                                  (target.coding.plain.origin_dots = false) is not supported"
                            .to_string(),
                    });
                }
                other => {
                    return Err(RunError::UnsupportedOption {
                        option: "target.coding",
                        value: format!("{other:?}"),
                    });
                }
            };
            let mut config = RinggridConfig::new(
                *pitch_m,
                *rows,
                *long_row_cols,
                *marker_outer_radius_m,
                *marker_inner_radius_m,
                *marker_ring_width_m,
            );
            config.center_method = center_method;
            config.lattice = lattice;
            config.coding = coding;
            config.min_confidence = *min_confidence;
            Ok(DetectorConfig::Ringgrid(config))
        }
        other => Err(RunError::UnsupportedTarget {
            kind: format!("{other:?}"),
        }),
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
/// be read, [`RunError::ImageSizeMismatch`] when one camera's images
/// differ in size, and [`RunError::RoiOutOfBounds`] when a camera's region
/// does not fit inside its images.
pub fn camera_image_sizes(spec: &DatasetSpec, base_dir: &Path) -> Result<Vec<[u32; 2]>, RunError> {
    vision_calibration_dataset::validate(spec)?;
    spec.cameras
        .iter()
        .map(|camera| {
            let images = expand_camera_images(camera, base_dir)?;
            let mut first: Option<(&PathBuf, [u32; 2])> = None;
            for path in &images {
                let (width, height) =
                    image::image_dimensions(path).map_err(|e| RunError::Decode {
                        path: path.clone(),
                        message: e.to_string(),
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
            let (path, size) = first.ok_or_else(|| RunError::EmptyImageMatch {
                camera_id: camera.id.clone(),
                pattern: pattern_repr(&camera.images),
                base: base_dir.to_path_buf(),
            })?;
            match camera.roi_xywh {
                Some(roi) => {
                    check_roi(roi, size, path)?;
                    Ok([roi[2], roi[3]])
                }
                None => Ok(size),
            }
        })
        .collect()
}

/// Every file a run of the manifest reads, each once and sorted: each
/// camera's target images and laser images, the robot pose file and the
/// upstream calibration. Relative paths resolve against `base_dir`.
///
/// Lists the image folders and reads no file, so a caller can check that a
/// run stays inside the folders it may read before it starts one. A
/// camera's pattern that matches nothing lists nothing; the run itself
/// fails on it.
///
/// # Errors
///
/// [`RunError::BadGlob`] for an image pattern that is not a valid glob,
/// [`RunError::UnsupportedOption`] for a pattern kind this version does not
/// read.
pub fn input_files(spec: &DatasetSpec, base_dir: &Path) -> Result<Vec<PathBuf>, RunError> {
    let mut files = std::collections::BTreeSet::new();
    for camera in &spec.cameras {
        files.extend(expand_camera_images(camera, base_dir)?);
        if let Some(laser) = &camera.laser_images {
            files.extend(expand_image_pattern(laser, &camera.id, base_dir)?);
        }
    }
    if let Some(poses) = &spec.robot_poses {
        files.insert(base_dir.join(&poses.path));
    }
    if let Some(upstream) = &spec.upstream_calibration {
        files.insert(base_dir.join(upstream));
    }
    Ok(files.into_iter().collect())
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
                message: e.to_string(),
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
        other => Err(RunError::UnsupportedOption {
            option: "images",
            value: format!("{other:?}"),
        }),
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
            (None, None) => return a.cmp(b),
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
        other => format!("{other:?}"),
    }
}

/// Run one image through the cache-or-detect path shared by every
/// converter: read bytes → cache lookup → on miss, decode, crop to
/// ROI, detect in the camera pixel frame, store.
///
/// Returns the features plus whether they came from the cache. When the
/// detector config requests corner covariance, cached and fresh features
/// alike are checked by [`check_corner_covariances`]; a feature's measured
/// blur, when it has one, by [`check_feature_blur`].
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
    check_feature_blur(&features, config.name(), image_path)?;
    Ok((features, cache_hit))
}

/// Every blur a feature carries must be one the observations can hold
/// ([`PointBlur::new`]). The detectors only report valid ones, so this
/// catches a cache entry written by other means; such an entry fails the
/// image rather than losing its blur silently.
fn check_feature_blur(
    features: &[Feature],
    detector: &'static str,
    image_path: &Path,
) -> Result<(), RunError> {
    for (i, f) in features.iter().enumerate() {
        if let Some(blur) = f.blur {
            PointBlur::new(blur.psf_px, blur.cov).map_err(|e| RunError::Detection {
                detector: detector.to_string(),
                path: image_path.to_path_buf(),
                source: vision_calibration_detect::DetectError::Backend {
                    detector,
                    message: format!("feature {i}: {e}"),
                },
            })?;
        }
    }
    Ok(())
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
#[non_exhaustive]
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

    /// The cached detail for `key`, when this mode reads the cache. Only
    /// [`detection_detail`] reads it; a calibration never does.
    fn lookup_detail(&self, key: &CacheKey) -> Result<Option<DetectionDetail>, CacheError> {
        match self {
            Detection::Cached(cache) => cache.get_detail(key),
            Detection::Refresh(_) | Detection::Uncached => Ok(None),
        }
    }

    /// Store a fresh detail, when this mode writes the cache.
    fn store_detail(&self, key: &CacheKey, detail: &DetectionDetail) -> Result<(), CacheError> {
        match self {
            Detection::Cached(cache) | Detection::Refresh(cache) => cache.put_detail(key, detail),
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
    detection.store(&key, &CachedFeatures::new(detected.clone()))?;
    Ok((detected, false))
}

/// The manifest's `pose_pairing`, or [`RunError::AskUser`] for a topology
/// that needs one and leaves it open. One place for the question, so every
/// converter and [`calibrate_dataset`](crate::dataset_runner::calibrate_dataset)
/// asks it the same way, before any image is read.
pub(crate) fn require_pose_pairing(
    spec: &DatasetSpec,
) -> Result<Option<&vision_calibration_dataset::PosePairing>, RunError> {
    let prompt = match spec.topology {
        ProblemKind::SingleCamHandeye => {
            "Hand-eye topologies need to know how images align with robot poses. \
             Pick by_index (pose row i pairs with the i-th image in natural-sort \
             order) or shared_filename_token (a regex extracts the pose id from \
             filenames)."
        }
        ProblemKind::LaserlineDevice => {
            "Laser topologies need to know how laser frames align with target \
             frames. Pick by_index (the i-th laser image pairs with the i-th \
             target image in natural-sort order) or shared_filename_token (a \
             regex extracts a shared view token from both filename sets)."
        }
        ProblemKind::RigExtrinsics
        | ProblemKind::RigHandeye
        | ProblemKind::RigLaserlineDevice
        | ProblemKind::RigHandeyeLaserline => {
            "Rig topologies need to know how per-camera images align into views. \
             Pick by_index (same count and order everywhere) or \
             shared_filename_token (a regex extracts a view token from filenames)."
        }
        // Planar intrinsics pairs nothing; a future topology states its own
        // question here.
        _ => return Ok(spec.pose_pairing.as_ref()),
    };
    match &spec.pose_pairing {
        Some(pairing) => Ok(Some(pairing)),
        None => Err(RunError::AskUser {
            field: "pose_pairing".into(),
            prompt: prompt.into(),
            suggestions: vec!["by_index".into(), "shared_filename_token".into()],
        }),
    }
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
        message: e.to_string(),
    })?;
    let img = match roi {
        Some(roi) => {
            check_roi(roi, [img.width(), img.height()], image_path)?;
            let [x, y, w, h] = roi;
            img.crop_imm(x, y, w, h)
        }
        None => img,
    };
    Ok(img.to_luma8())
}

/// The region `roi` (`[x, y, width, height]`) lies inside an image of
/// `size` (`[width, height]`): cropping would otherwise clip it silently,
/// and the detections would be reported in a smaller frame than the
/// manifest states.
fn check_roi(roi: [u32; 4], size: [u32; 2], path: &Path) -> Result<(), RunError> {
    let [x, y, w, h] = roi;
    let fits = x.checked_add(w).is_some_and(|right| right <= size[0])
        && y.checked_add(h).is_some_and(|bottom| bottom <= size[1]);
    if fits {
        Ok(())
    } else {
        Err(RunError::RoiOutOfBounds {
            path: path.to_path_buf(),
            size,
            roi,
        })
    }
}

fn features_to_view(features: &[Feature]) -> Result<View<NoMeta>, RunError> {
    Ok(View::without_meta(features_to_obs(features)))
}

/// The 2D-3D correspondences of one view, with each point's covariance when
/// the features carry them, and each point's measured blur.
///
/// Covariances are all-or-none per view: [`detect_features`] rejects an image
/// where a requested covariance is missing or unusable, so the features here
/// either all carry a valid one or (the default) none does. Blur is per
/// feature, and [`detect_features`] has checked each one.
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
    let blur: Vec<Option<PointBlur>> = features
        .iter()
        .map(|f| {
            f.blur.map(|b| {
                PointBlur::new(b.psf_px, b.cov)
                    .expect("blurs are validated by check_feature_blur in detect_features")
            })
        })
        .collect();
    let view = match covariances {
        Some(cov) if !cov.is_empty() => {
            CorrespondenceView::new_with_covariances(points_3d, points_2d, cov)
        }
        _ => CorrespondenceView::new(points_3d, points_2d),
    };
    view.and_then(|v| v.with_blur(blur)).expect(
        "Feature vectors are non-empty and equal-length by construction; \
         CorrespondenceView::new and with_blur only fail on mismatched lengths.",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use vision_calibration_dataset::{
        CameraSource, ChessCornersDetectorSpec, DetectorSpec, ImagePattern,
    };

    fn spec_for_target(target: TargetSpec) -> DatasetSpec {
        DatasetSpec::new(
            vec![CameraSource::new(
                "cam0",
                ImagePattern::List { paths: vec![] },
            )],
            target,
            ProblemKind::PlanarIntrinsics,
        )
    }

    #[test]
    fn charuco_target_maps_to_detector_config() {
        // 12×12 needs 72 markers, so the dictionary must hold at least
        // that many (DICT_4X4_50 would be rejected — see the capacity test).
        let target = TargetSpec::charuco(12, 12, 0.020, 0.015, "DICT_4X4_100");
        let spec = spec_for_target(target);
        let DetectorConfig::Charuco(config) = target_to_detector_config(&spec).unwrap() else {
            panic!("a charuco target lowers to the charuco detector");
        };
        assert_eq!(config.dictionary, "DICT_4X4_100");
        assert_eq!(config.marker_size_m, 0.015);
    }

    /// The targets' `measure_blur` reaches the chessboard and PuzzleBoard
    /// detector configs, and is off when left out.
    #[test]
    fn measure_blur_maps_to_the_detector_config() {
        let lower = |target: TargetSpec| target_to_detector_config(&spec_for_target(target));
        let mut chessboard = TargetSpec::chessboard(5, 7, 0.03);
        let Ok(DetectorConfig::Chessboard(config)) = lower(chessboard.clone()) else {
            panic!("a chessboard target lowers to the chessboard detector");
        };
        assert!(!config.measure_blur);
        if let TargetSpec::Chessboard { measure_blur, .. } = &mut chessboard {
            *measure_blur = true;
        }
        let Ok(DetectorConfig::Chessboard(config)) = lower(chessboard) else {
            panic!("a chessboard target lowers to the chessboard detector");
        };
        assert!(config.measure_blur);

        let mut puzzle = TargetSpec::puzzleboard("puzzle_12x12", 0.012);
        let Ok(DetectorConfig::Puzzleboard(config)) = lower(puzzle.clone()) else {
            panic!("a puzzleboard target lowers to the puzzleboard detector");
        };
        assert!(!config.measure_blur);
        if let TargetSpec::Puzzleboard { measure_blur, .. } = &mut puzzle {
            *measure_blur = true;
        }
        let Ok(DetectorConfig::Puzzleboard(config)) = lower(puzzle) else {
            panic!("a puzzleboard target lowers to the puzzleboard detector");
        };
        assert!(config.measure_blur);
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
        let target = TargetSpec::chessboard(5, 7, 0.03);
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
        let target = TargetSpec::chessboard(5, 7, 0.03);
        assert!(
            detect_target(view(&image), &target, None)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn detect_target_rejects_an_impossible_target_before_detection() {
        let image = chessboard_image(8, 6, 40);
        let target = TargetSpec::charuco(12, 12, 0.020, 0.015, "DICT_TYPO_99");
        let err = detect_target(view(&image), &target, None).unwrap_err();
        assert!(
            matches!(err, RunError::InvalidTargetConfig { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn charuco_bad_dictionary_fails_before_io() {
        let target = TargetSpec::charuco(12, 12, 0.020, 0.015, "DICT_TYPO_99");
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
        let target = TargetSpec::charuco(12, 12, 0.020, 0.015, "DICT_4X4_50");
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
        let mut spec = spec_for_target(TargetSpec::charuco(8, 8, 0.020, 0.015, "DICT_4X4_50"));
        spec.detector = Some({
            let mut value = DetectorSpec::default();
            value.chess_corners = Some({
                let mut value = ChessCornersDetectorSpec::default();
                value.strategy = Some(CornerStrategySpec::Radon);
                value.threshold_value = Some(0.2);
                value.covariance = Some(CornerCovarianceSpec::new(ImageNoiseSpec::Stated(2.0)));
                value
            });
            value.min_features_per_view = None;
            value
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
            Some(CornerCovariance::new(ImageNoise::Stated(2.0)))
        );
        assert_eq!(
            lower_chess_corners({
                let mut value = ChessCornersDetectorSpec::default();
                value.covariance = Some(CornerCovarianceSpec::default());
                value
            })
            .unwrap()
            .covariance,
            Some(CornerCovariance::new(ImageNoise::Estimate))
        );
        assert_eq!(
            lowered,
            lower_chess_corners(spec.detector.unwrap().chess_corners.unwrap()).unwrap()
        );
    }

    #[test]
    fn default_detector_override_is_not_attached() {
        // An all-`None` override must not inject an empty `chess_corners`
        // object: that would change the detection-cache key for a manifest
        // that asked for nothing.
        let mut spec = spec_for_target(TargetSpec::chessboard(9, 6, 0.025));
        spec.detector = Some(DetectorSpec::default());
        let DetectorConfig::Chessboard(config) = target_to_detector_config(&spec).unwrap() else {
            panic!("a chessboard target lowers to the chessboard detector");
        };
        assert!(config.chess_corners.is_none());
    }

    #[test]
    fn puzzleboard_target_maps_to_detector_config() {
        let target = TargetSpec::puzzleboard("puzzle_130x130", 0.0052);
        let spec = spec_for_target(target);
        let config = target_to_detector_config(&spec).unwrap();
        assert_eq!(
            config,
            DetectorConfig::Puzzleboard(PuzzleboardConfig::new(130, 130, 0.0052))
        );
    }

    #[test]
    fn puzzleboard_bad_layout_fails_before_io() {
        let target = TargetSpec::puzzleboard("not-a-grid", 0.0052);
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
        let mut target = TargetSpec::ringgrid(0.02, 5, 5, 0.006, 0.003, 0.001);
        if let TargetSpec::Ringgrid { center_method, .. } = &mut target {
            *center_method = RinggridCenterSpec::Photometric;
        }
        let spec = spec_for_target(target);
        let DetectorConfig::Ringgrid(config) = target_to_detector_config(&spec).unwrap() else {
            panic!("a ringgrid target lowers to the ringgrid detector");
        };
        assert_eq!(config.center_method, RinggridCenterMethod::Photometric);
        assert_eq!(config.long_row_cols, 5);
        assert_eq!(config.marker_outer_radius_m, 0.006);
        assert_eq!(config.lattice, RinggridLattice::Hex);
        assert_eq!(config.coding, RinggridCoding::Coded);
        assert_eq!(config.min_confidence, None);
    }

    /// The target's confidence floor reaches the detector config; one out of
    /// `[0, 1]` is refused before any detection.
    #[test]
    fn ringgrid_min_confidence_maps_and_is_checked() {
        let with = |c: f64| {
            let mut target = TargetSpec::ringgrid(0.02, 5, 5, 0.006, 0.003, 0.001);
            if let TargetSpec::Ringgrid { min_confidence, .. } = &mut target {
                *min_confidence = Some(c);
            }
            target_to_detector_config(&spec_for_target(target))
        };
        let Ok(DetectorConfig::Ringgrid(config)) = with(0.17) else {
            panic!("a ringgrid target lowers to the ringgrid detector");
        };
        assert_eq!(config.min_confidence, Some(0.17));
        for bad in [-0.5, 1.2, f64::INFINITY] {
            match with(bad) {
                Err(RunError::InvalidTargetConfig { message }) => {
                    assert!(message.contains("min_confidence"), "{message}");
                }
                other => panic!("expected InvalidTargetConfig, got {other:?}"),
            }
        }
    }

    #[test]
    fn plain_rect_ringgrid_target_maps_to_detector_config() {
        let target =
            TargetSpec::ringgrid_plain(RinggridLatticeSpec::Rect, 0.002, 58, 58, 0.0008, 0.0004);
        let spec = spec_for_target(target);
        let DetectorConfig::Ringgrid(config) = target_to_detector_config(&spec).unwrap() else {
            panic!("a ringgrid target lowers to the ringgrid detector");
        };
        assert_eq!(config.lattice, RinggridLattice::Rect);
        assert_eq!(config.coding, RinggridCoding::Plain { origin_dots: true });
        assert_eq!((config.rows, config.long_row_cols), (58, 58));
        assert_eq!(config.pitch_m, 0.002);

        let coded_rect = TargetSpec::ringgrid_rect(0.014, 20, 20, 0.0048, 0.0032, 0.001152);
        let DetectorConfig::Ringgrid(config) =
            target_to_detector_config(&spec_for_target(coded_rect)).unwrap()
        else {
            panic!("a ringgrid target lowers to the ringgrid detector");
        };
        assert_eq!(config.lattice, RinggridLattice::Rect);
        assert_eq!(config.coding, RinggridCoding::Coded);
        assert_eq!(config.marker_ring_width_m, 0.001152);
    }

    #[test]
    fn plain_ringgrid_without_origin_dots_is_refused_before_detection() {
        let mut target =
            TargetSpec::ringgrid_plain(RinggridLatticeSpec::Hex, 0.02, 5, 5, 0.006, 0.003);
        if let TargetSpec::Ringgrid { coding, .. } = &mut target {
            *coding = RinggridCodingSpec::Plain { origin_dots: false };
        }
        match target_to_detector_config(&spec_for_target(target)) {
            Err(RunError::InvalidTargetConfig { message }) => {
                assert!(message.contains("origin dots"), "got: {message}");
            }
            other => panic!("expected InvalidTargetConfig, got {other:?}"),
        }
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
        let config = DetectorConfig::Chessboard(ChessboardConfig::new(9, 6, 0.025));
        let key_no_roi = CacheKey::for_detection(b"img", &config, None);
        let key_roi_a = CacheKey::for_detection(b"img", &config, Some([10, 20, 100, 100]));
        let key_roi_b = CacheKey::for_detection(b"img", &config, Some([10, 20, 200, 100]));
        assert_ne!(key_no_roi, key_roi_a);
        assert_ne!(key_roi_a, key_roi_b);
        assert_ne!(key_no_roi, key_roi_b);
    }

    /// A one-camera planar spec over `paths`, with `roi`.
    fn sized_spec(paths: Vec<PathBuf>, roi: Option<[u32; 4]>) -> DatasetSpec {
        let mut spec = spec_for_target(TargetSpec::chessboard(5, 4, 0.03));
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

        // A region inside the images is the camera's image.
        let spec = sized_spec(paths, Some([8, 4, 32, 24]));
        assert_eq!(
            camera_image_sizes(&spec, tmp.path()).unwrap(),
            vec![[32, 24]]
        );
    }

    /// Every file a run reads, sorted and listed once, without reading any:
    /// the images are files that are not images.
    #[test]
    fn input_files_lists_images_laser_frames_and_side_files() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        for name in ["t/b.png", "t/a.png", "l/a.png"] {
            std::fs::create_dir_all(dir.join(name).parent().unwrap()).unwrap();
            std::fs::write(dir.join(name), b"-").unwrap();
        }
        let mut spec = sized_spec(vec![], None);
        spec.cameras[0].images = ImagePattern::Glob {
            pattern: "t/*.png".into(),
        };
        spec.cameras[0].laser_images = Some(ImagePattern::List {
            paths: vec!["l/a.png".into(), "t/a.png".into()],
        });
        spec.robot_poses = Some(vision_calibration_dataset::RobotPoseSource::new(
            "poses.csv",
            vision_calibration_dataset::RobotPoseFormat::Rowmajor4x4,
        ));
        // An absolute path on every platform: a second folder's file.
        let elsewhere = tempfile::tempdir().unwrap();
        let upstream = elsewhere.path().join("export.json");
        spec.upstream_calibration = Some(upstream.clone());
        let files = input_files(&spec, dir).unwrap();
        let expected: Vec<PathBuf> = vec![
            upstream,
            dir.join("l/a.png"),
            dir.join("poses.csv"),
            dir.join("t/a.png"),
            dir.join("t/b.png"),
        ];
        let mut sorted = expected.clone();
        sorted.sort();
        assert_eq!(files, sorted);
    }

    #[test]
    fn a_region_that_runs_past_the_images_is_an_error() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = vec![write_blank(tmp.path(), "a.png", 64, 48)];
        let spec = sized_spec(paths.clone(), Some([40, 0, 32, 24]));
        let err = camera_image_sizes(&spec, tmp.path()).unwrap_err();
        assert!(
            matches!(err, RunError::RoiOutOfBounds { size: [64, 48], .. }),
            "{err:?}"
        );
        // Decoding a frame refuses it too, rather than cropping it short.
        let bytes = std::fs::read(&paths[0]).unwrap();
        let err = decode_gray(&bytes, &paths[0], Some([40, 0, 32, 24])).unwrap_err();
        assert!(matches!(err, RunError::RoiOutOfBounds { .. }), "{err:?}");
        // And a camera whose region names no image is still an error.
        let mut spec = sized_spec(vec![], Some([0, 0, 8, 8]));
        spec.cameras[0].images = ImagePattern::Glob {
            pattern: "missing/*.png".into(),
        };
        let err = camera_image_sizes(&spec, tmp.path()).unwrap_err();
        assert!(matches!(err, RunError::EmptyImageMatch { .. }), "{err:?}");
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

    /// Every key of `T`'s JSON schema that names a field: the `properties`
    /// of each object it describes, its own and its parts'.
    #[cfg(feature = "schemars")]
    fn schema_fields<T: schemars::JsonSchema>() -> std::collections::BTreeSet<String> {
        fn collect(v: &serde_json::Value, out: &mut std::collections::BTreeSet<String>) {
            match v {
                serde_json::Value::Object(map) => {
                    if let Some(serde_json::Value::Object(props)) = map.get("properties") {
                        out.extend(props.keys().cloned());
                    }
                    map.values().for_each(|v| collect(v, out));
                }
                serde_json::Value::Array(items) => items.iter().for_each(|v| collect(v, out)),
                _ => {}
            }
        }
        let mut out = std::collections::BTreeSet::new();
        collect(
            &serde_json::to_value(schemars::schema_for!(T)).unwrap(),
            &mut out,
        );
        out
    }

    /// The manifest's target and chess-corner types can grow, so their
    /// lowering to the detector configs destructures them with `..`. A field
    /// added to the manifest must be wired through `lower_chess_corners` and
    /// `target_to_detector_config` and then listed here.
    #[cfg(feature = "schemars")]
    #[test]
    fn lowering_covers_every_manifest_field() {
        let set = |names: &[&str]| names.iter().map(|n| n.to_string()).collect();
        assert_eq!(
            schema_fields::<ChessCornersDetectorSpec>(),
            set(&[
                "covariance",
                "image_noise",
                "min_corner_strength",
                "stated",
                "strategy",
                "threshold_value",
            ])
        );
        assert_eq!(
            schema_fields::<TargetSpec>(),
            set(&[
                "cell_size_m",
                "center_method",
                "coding",
                "cols",
                "dictionary",
                "kind",
                "lattice",
                "layout",
                "long_row_cols",
                "marker_inner_radius_m",
                "marker_outer_radius_m",
                "marker_ring_width_m",
                "marker_size_m",
                "measure_blur",
                "min_confidence",
                "origin_dots",
                "pitch_m",
                "plain",
                "rows",
                "square_size_m",
            ])
        );
    }

    /// The corner producer keeps a blur only when the calibration accepts
    /// it: `FeatureBlur::is_valid` (detect, without the calibration types)
    /// implies `PointBlur::new` (core), on semi-definite covariances of every
    /// rank perturbed from far inside to far outside the tolerance and
    /// rounded to `f32` as the corner detector delivers them. The two also
    /// agree away from the boundary.
    #[test]
    fn producer_blur_check_matches_the_observation_check() {
        use nalgebra::{Matrix3, Vector3};
        use vision_calibration_detect::FeatureBlur;

        // SplitMix64, uniform in [-1, 1).
        let mut state = 0x5EED_u64;
        let mut uniform = move || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            2.0 * ((z >> 11) as f64 / (1u64 << 53) as f64) - 1.0
        };
        let (mut agree, mut checked, mut kept) = (0, 0, 0);
        for trial in 0..4000 {
            let rank = 1 + trial % 3;
            let magnitude = 10f64.powf(-6.0 + 4.0 * uniform().abs());
            let mut c = Matrix3::zeros();
            for _ in 0..rank {
                let v = Vector3::new(uniform(), uniform(), uniform());
                c += v * v.transpose();
            }
            c *= magnitude;
            // Move the smallest eigenvalue by a multiple of the largest
            // variance spanning the tolerance (−1e-9) on both sides.
            let scale = c.diagonal().max();
            let shift = [
                0.0, 1e-12, -1e-11, -3e-10, -6e-10, -9e-10, -1.1e-9, -2e-9, -1e-6,
            ][trial % 9]
                * scale;
            let e = c.symmetric_eigen();
            let u = e.eigenvectors.column(e.eigenvalues.imin()).into_owned();
            c += u * u.transpose() * shift;
            for to_f32 in [false, true] {
                let cov: [[f64; 3]; 3] = std::array::from_fn(|r| {
                    std::array::from_fn(|k| {
                        let v = c[(r, k)];
                        if to_f32 { f64::from(v as f32) } else { v }
                    })
                });
                let produced = FeatureBlur::new([1.5, 0.1, 1.2], cov).is_valid();
                let observed = PointBlur::new([1.5, 0.1, 1.2], cov).is_ok();
                assert!(!produced || observed, "the producer keeps {cov:?}");
                let s = (0..3).map(|i| cov[i][i]).fold(0.0, f64::max);
                let sym = Matrix3::from_fn(|r, k| 0.5 * (cov[r][k] + cov[k][r]));
                let lambda = sym.symmetric_eigenvalues().min() / s;
                if !(-1.1e-9..=-0.4e-9).contains(&lambda) {
                    assert_eq!(
                        produced, observed,
                        "λ_min {lambda:e} of the largest variance: {cov:?}"
                    );
                    agree += 1;
                    kept += usize::from(produced);
                }
                checked += 1;
            }
        }
        // Most cases lie away from the boundary band.
        assert!(agree * 10 >= checked * 6, "{agree} of {checked}");
        // Both outcomes occur among them.
        assert!(
            kept * 5 >= agree && kept * 5 <= agree * 4,
            "{kept} of {agree} kept"
        );
    }

    /// A corner whose covariance is slightly indefinite (as `f32` rounding
    /// of a rank-deficient one can leave it) is dropped by the producer's
    /// check, so the image passes; kept, it would fail the image.
    #[test]
    fn an_indefinite_corner_blur_is_dropped_not_failing_the_image() {
        use vision_calibration_detect::FeatureBlur;

        let off = f64::from(-1.000_001e-3_f32);
        let indefinite = [[1e-3, 0.0, off], [0.0, 1e-4, 0.0], [off, 0.0, 1e-3]];
        let blur = FeatureBlur::new([1.0, 0.0, 1.0], indefinite);
        assert!(!blur.is_valid());
        let corner = Feature::new([10.0, 20.0], [0.0, 0.0, 0.0]);
        let path = Path::new("01.png");
        // What the producer emits: the corner without a blur.
        let emitted = [corner.with_blur(Some(blur).filter(FeatureBlur::is_valid))];
        assert!(check_feature_blur(&emitted, "chessboard", path).is_ok());
        let kept = [corner.with_blur(Some(blur))];
        assert!(check_feature_blur(&kept, "chessboard", path).is_err());
    }
}
