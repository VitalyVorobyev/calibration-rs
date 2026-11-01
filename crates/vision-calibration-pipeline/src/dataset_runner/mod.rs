//! Dataset-driven runner: takes a
//! [`DatasetSpec`] manifest plus a
//! per-problem `*Config` and produces the existing `*Input` IR by
//! running detection (cached) on the per-camera images.
//!
//! Converters:
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
//!   only when it spans the whole board as declared: corners on every edge,
//!   `cols` across and `rows` down. The detector labels a view from its own
//!   top-left corner, so a partial board, or a board seen turned by 90°,
//!   would move the shared frame. Self-identifying targets (ChArUco,
//!   PuzzleBoard, ring grid) label partial views absolutely and are kept.
//!
//! A camera left with no usable view is an error before any solve:
//! [`RunError::NoWholeChessboardViews`] when the whole-board rule rejected
//! any of its views, [`RunError::InsufficientUsableViews`] otherwise.
//!
//! Any ambiguity that cannot be auto-resolved at
//! conversion time is surfaced as a [`RunError::AskUser`] event so
//! the UI can prompt the user instead of silently guessing.

use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use thiserror::Error;

use vision_calibration_core::{CorrespondenceView, NoMeta, PointCovariance, Pt2, Pt3, View};
use vision_calibration_dataset::{
    ChessCornersDetectorSpec, CornerCovarianceSpec, CornerStrategySpec, DatasetSpec, DetectorSpec,
    ImageNoiseSpec, ImagePattern, TargetSpec, Topology, ValidationError,
};
use vision_calibration_detect::{
    CacheError, CacheKey, CachedFeatures, CharucoDetector, ChessCornersConfig, ChessboardDetector,
    CornerCovariance, CornerStrategy, DetectionCache, Detector, Feature, ImageNoise,
    PuzzleboardDetector, RinggridDetector, validate_charuco_layout,
};

mod handeye;
mod laser;
mod pairing;
mod planar;
mod poses;
mod rig;
mod view_filter;

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
    /// plain chessboard was not seen whole in its declared orientation.
    ///
    /// The views of this topology share one target frame, and a plain
    /// chessboard detection labels its corners from its own top-left
    /// corner, so only a detection that reaches every edge of the board,
    /// seen as declared (`cols` corners across, `rows` down), places that
    /// frame. A target with coded corners (ChArUco, PuzzleBoard) is labelled
    /// absolutely in partial views.
    #[error(
        "camera {camera_id:?}: no view shows the whole chessboard as declared \
         ({partial} partial, {turned} turned by 90°, of {total}); views of this \
         topology share one target frame, which a plain chessboard places only \
         when every edge of it is detected — capture the whole board, or use a \
         ChArUco or PuzzleBoard target for partial views"
    )]
    NoWholeChessboardViews {
        /// Camera id.
        camera_id: String,
        /// Views rejected because the board was partly out of view.
        partial: usize,
        /// Views rejected because the board was seen turned by 90°.
        turned: usize,
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

fn target_to_detector_config(spec: &DatasetSpec) -> Result<(&'static str, Value), RunError> {
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
/// detection; [`RunError::Detection`] when the detector rejects its config.
pub fn detect_target(
    image: &image::DynamicImage,
    target: &TargetSpec,
    detector: Option<&DetectorSpec>,
) -> Result<Vec<Feature>, RunError> {
    let (name, config) = detector_config(target, detector)?;
    pick_detector(name)?
        .detect_json(image, &config)
        .map_err(|source| RunError::Detection {
            detector: name.to_string(),
            path: PathBuf::from("<in-memory image>"),
            source,
        })
}

/// The detector name and config JSON for `target`, with the manifest's
/// detector override applied. The one place a manifest target is lowered to
/// a detector config.
fn detector_config(
    target: &TargetSpec,
    detector: Option<&DetectorSpec>,
) -> Result<(&'static str, Value), RunError> {
    let (name, mut config) = detector_config_for_target(target)?;
    if let Some(chess) = detector.and_then(|d| d.chess_corners)
        && chess.covariance.is_some()
        && !matches!(name, "chessboard" | "charuco")
    {
        return Err(RunError::CovarianceUnsupported { target: name });
    }
    if let Some(chess) = detector.and_then(|d| d.chess_corners)
        && let Value::Object(map) = &mut config
    {
        let chess = lower_chess_corners(chess);
        // `skip_serializing_if` drops unset knobs, so an all-default override
        // serializes to `{}`; attaching it would only bloat the cache key.
        let chess_json = serde_json::to_value(chess)
            .expect("ChessCornersConfig is a plain record of options; cannot fail");
        if chess_json.as_object().is_some_and(|o| !o.is_empty()) {
            map.insert("chess_corners".to_string(), chess_json);
        }
    }
    Ok((name, config))
}

fn detector_config_for_target(target: &TargetSpec) -> Result<(&'static str, Value), RunError> {
    match target {
        TargetSpec::Chessboard {
            rows,
            cols,
            square_size_m,
        } => Ok((
            "chessboard",
            json!({
                "rows": *rows,
                "cols": *cols,
                "square_size_m": *square_size_m,
            }),
        )),
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
            Ok((
                "charuco",
                json!({
                    "rows": *rows,
                    "cols": *cols,
                    "square_size_m": *square_size_m,
                    "marker_size_m": *marker_size_m,
                    "dictionary": dictionary,
                }),
            ))
        }
        TargetSpec::Puzzleboard {
            layout,
            cell_size_m,
        } => {
            // The manifest names a layout (`"puzzle_130x130"`); the
            // detector needs explicit dimensions. Resolve here so a typo
            // fails before any image I/O (ADR 0016).
            let (rows, cols) = parse_puzzleboard_layout(layout)
                .map_err(|message| RunError::InvalidTargetConfig { message })?;
            Ok((
                "puzzleboard",
                json!({
                    "rows": rows,
                    "cols": cols,
                    "cell_size_m": *cell_size_m,
                }),
            ))
        }
        TargetSpec::Ringgrid {
            pitch_m,
            rows,
            long_row_cols,
            marker_outer_radius_m,
            marker_inner_radius_m,
            marker_ring_width_m,
        } => Ok((
            "ringgrid",
            json!({
                "pitch_m": *pitch_m,
                "rows": *rows,
                "long_row_cols": *long_row_cols,
                "marker_outer_radius_m": *marker_outer_radius_m,
                "marker_inner_radius_m": *marker_inner_radius_m,
                "marker_ring_width_m": *marker_ring_width_m,
            }),
        )),
    }
}

fn pick_detector(name: &str) -> Result<Box<dyn Detector>, RunError> {
    match name {
        "chessboard" => Ok(Box::new(ChessboardDetector)),
        "charuco" => Ok(Box::new(CharucoDetector)),
        "puzzleboard" => Ok(Box::new(PuzzleboardDetector)),
        "ringgrid" => Ok(Box::new(RinggridDetector)),
        other => Err(RunError::UnsupportedTarget {
            kind: other.to_string(),
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

/// Splice ROI-only metadata into the detector's canonical config JSON
/// so the cache key changes whenever the user edits the ROI or the
/// ROI-local coordinate contract changes. The detector itself never
/// sees these fields — the runner crops the image upstream.
fn augment_config_with_roi(detector_config: &Value, roi: Option<[u32; 4]>) -> Value {
    let Some(roi) = roi else {
        return detector_config.clone();
    };
    let mut copy = detector_config.clone();
    if let Value::Object(map) = &mut copy {
        map.insert("_roi".to_string(), json!([roi[0], roi[1], roi[2], roi[3]]));
        map.insert("_coord_frame".to_string(), json!("roi_local_v2"));
    }
    copy
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
#[allow(clippy::too_many_arguments)]
fn detect_features(
    detector: &dyn Detector,
    detector_config: &Value,
    key_config: &Value,
    roi: Option<[u32; 4]>,
    image_path: &Path,
    detection: Detection<'_>,
) -> Result<(Vec<Feature>, bool), RunError> {
    let (features, cache_hit) = cached_or_detected(
        detector,
        detector_config,
        key_config,
        roi,
        image_path,
        detection,
    )?;
    if covariance_requested(detector_config) {
        check_corner_covariances(&features, detector.name(), image_path)?;
    }
    Ok((features, cache_hit))
}

/// Whether the lowered detector config asks for per-corner covariance.
fn covariance_requested(detector_config: &Value) -> bool {
    detector_config
        .get("chess_corners")
        .and_then(|c| c.get("covariance"))
        .is_some()
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

#[allow(clippy::too_many_arguments)]
fn cached_or_detected(
    detector: &dyn Detector,
    detector_config: &Value,
    key_config: &Value,
    roi: Option<[u32; 4]>,
    image_path: &Path,
    detection: Detection<'_>,
) -> Result<(Vec<Feature>, bool), RunError> {
    let bytes = std::fs::read(image_path)?;
    let key = CacheKey::for_detector(&bytes, detector, key_config);

    let cached: Option<CachedFeatures> = detection.lookup(&key)?;
    if let Some(entry) = cached {
        return Ok((entry.features, true));
    }

    let img = image::load_from_memory(&bytes).map_err(|e| RunError::Decode {
        path: image_path.to_path_buf(),
        source: e,
    })?;
    let img_for_detect = if let Some([x, y, w, h]) = roi {
        img.crop_imm(x, y, w, h)
    } else {
        img
    };
    let detected = detector
        .detect_json(&img_for_detect, detector_config)
        .map_err(|e| RunError::Detection {
            detector: detector.name().to_string(),
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
        let (name, config) = target_to_detector_config(&spec).unwrap();
        assert_eq!(name, "charuco");
        assert_eq!(config["dictionary"], "DICT_4X4_100");
        assert_eq!(config["marker_size_m"], 0.015);
        // The mapped config must deserialize into the detector's own
        // config struct — guards the field-name contract between the
        // manifest and the detect crate.
        pick_detector(name).unwrap();
    }

    /// A synthetic chessboard: `squares_x` × `squares_y` squares of `px`
    /// pixels on a white margin of one square.
    fn chessboard_image(squares_x: u32, squares_y: u32, px: u32) -> image::DynamicImage {
        let (w, h) = ((squares_x + 2) * px, (squares_y + 2) * px);
        let img = image::GrayImage::from_fn(w, h, |x, y| {
            let (cx, cy) = (x / px, y / px);
            let inside = (1..=squares_x).contains(&cx) && (1..=squares_y).contains(&cy);
            image::Luma([if inside && (cx + cy) % 2 == 0 { 0 } else { 255 }])
        });
        image::DynamicImage::ImageLuma8(img)
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
        let features = detect_target(&image, &target, None).unwrap();
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
        let image = image::DynamicImage::ImageLuma8(image::GrayImage::from_pixel(
            200,
            150,
            image::Luma([128]),
        ));
        let target = TargetSpec::Chessboard {
            rows: 5,
            cols: 7,
            square_size_m: 0.03,
        };
        assert!(detect_target(&image, &target, None).unwrap().is_empty());
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
        let err = detect_target(&image, &target, None).unwrap_err();
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
        let (_name, config) = target_to_detector_config(&spec).unwrap();
        assert_eq!(config["chess_corners"]["strategy"], "radon");
        assert_eq!(
            config["chess_corners"]["covariance"]["image_noise"]["known"],
            2.0
        );
        assert_eq!(config["chess_corners"]["threshold_value"], 0.2f32 as f64);

        // `lower_chess_corners` destructures the manifest struct exhaustively,
        // so a new manifest knob that is not wired through fails to compile.
        // This checks the other direction: that what we emit still round-trips
        // into the detector's own `deny_unknown_fields` config.
        let lowered: ChessCornersConfig =
            serde_json::from_value(config["chess_corners"].clone()).unwrap();
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
        let (_name, config) = target_to_detector_config(&spec).unwrap();
        assert!(config.get("chess_corners").is_none());
    }

    #[test]
    fn puzzleboard_target_maps_to_detector_config() {
        let target = TargetSpec::Puzzleboard {
            layout: "puzzle_130x130".into(),
            cell_size_m: 0.0052,
        };
        let spec = spec_for_target(target);
        let (name, config) = target_to_detector_config(&spec).unwrap();
        assert_eq!(name, "puzzleboard");
        assert_eq!(config["rows"], 130);
        assert_eq!(config["cols"], 130);
        assert_eq!(config["cell_size_m"], 0.0052);
        // The mapped config must deserialize into the detector's own
        // config struct — guards the field-name contract between the
        // manifest dispatcher and the detect crate.
        let _cfg: vision_calibration_detect::PuzzleboardConfig =
            serde_json::from_value(config).unwrap();
        pick_detector(name).unwrap();
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
        let (name, config) = target_to_detector_config(&spec).unwrap();
        assert_eq!(name, "ringgrid");
        assert_eq!(config["long_row_cols"], 5);
        assert_eq!(config["marker_outer_radius_m"], 0.006);
        // Field-name contract guard between the dispatcher and the detect crate.
        let _cfg: vision_calibration_detect::RinggridConfig =
            serde_json::from_value(config).unwrap();
        pick_detector(name).unwrap();
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
    fn augment_config_with_roi_changes_cache_key() {
        // Two ROIs over the same image with the same detector config
        // must hash to different cache keys; otherwise a user editing
        // an ROI would silently re-use stale detections.
        let detector_config = json!({"rows": 9, "cols": 6, "square_size_m": 0.025});
        let key_no_roi = CacheKey::from_inputs(
            b"img",
            "chessboard",
            &augment_config_with_roi(&detector_config, None),
        );
        let key_roi_a = CacheKey::from_inputs(
            b"img",
            "chessboard",
            &augment_config_with_roi(&detector_config, Some([10, 20, 100, 100])),
        );
        let key_roi_b = CacheKey::from_inputs(
            b"img",
            "chessboard",
            &augment_config_with_roi(&detector_config, Some([10, 20, 200, 100])),
        );
        assert_ne!(key_no_roi, key_roi_a);
        assert_ne!(key_roi_a, key_roi_b);
        assert_ne!(key_no_roi, key_roi_b);
    }

    #[test]
    fn detector_config_unchanged_by_roi_augmentation() {
        // Splicing _roi into the *cache-key* config must not pollute
        // the *detector-side* config — the detector is ROI-agnostic
        // and would reject unknown fields with `deny_unknown_fields`.
        let detector_config = json!({"rows": 9, "cols": 6, "square_size_m": 0.025});
        let augmented = augment_config_with_roi(&detector_config, Some([0, 0, 64, 64]));
        assert_ne!(detector_config, augmented);
        assert!(augmented.get("_roi").is_some());
        assert_eq!(augmented["_coord_frame"], "roi_local_v2");
        assert!(detector_config.get("_roi").is_none());
        assert!(detector_config.get("_coord_frame").is_none());
    }
}
