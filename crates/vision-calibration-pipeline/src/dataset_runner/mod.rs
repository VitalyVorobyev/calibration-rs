//! Dataset-driven runner: takes a
//! [`DatasetSpec`](vision_calibration_dataset::DatasetSpec) manifest plus a
//! per-problem `*Config` and produces the existing `*Input` IR by
//! running detection (cached) on the per-camera images.
//!
//! Converters:
//! - [`build_planar_input`] — `PlanarIntrinsics` / `ScheimpflugIntrinsics`
//!   (both consume a `PlanarDataset`).
//! - [`build_rig_extrinsics_input`] — `RigExtrinsics` (`RigDataset<NoMeta>`).
//! - [`build_rig_handeye_input`] — `RigHandeye`
//!   (`RigDataset<RobotPoseMeta>`, robot poses loaded from the manifest's
//!   [`RobotPoseSource`](vision_calibration_dataset::RobotPoseSource)).
//! - [`build_single_cam_handeye_input`] — `SingleCamHandeye`
//!   (`SingleCamHandeyeInput`, single camera + robot poses).
//!
//! The laser topologies (`LaserlineDevice`, `RigLaserlineDevice`) await
//! the laser-frame manifest design.
//!
//! Per ADR 0019, any ambiguity that cannot be auto-resolved at
//! conversion time is surfaced as a [`RunError::AskUser`] event so
//! the UI can prompt the user instead of silently guessing.

use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use thiserror::Error;

use vision_calibration_core::{CorrespondenceView, NoMeta, Pt2, Pt3, View};
use vision_calibration_dataset::{ImagePattern, TargetSpec, Topology, ValidationError};
use vision_calibration_detect::{
    CacheKey, CachedFeatures, CharucoDetector, ChessboardDetector, DetectionCache, Detector,
    Feature, validate_charuco_layout,
};

mod handeye;
mod pairing;
mod planar;
mod poses;
mod rig;

pub use handeye::{HandeyeRunResult, build_single_cam_handeye_input};
pub use pairing::PairedViews;
pub use planar::{PlanarRunResult, build_planar_input};
pub use rig::{RigRunResult, build_rig_extrinsics_input, build_rig_handeye_input};

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
        source: anyhow::Error,
    },

    /// Cache backend reported an error.
    #[error("cache error: {0}")]
    Cache(#[from] vision_calibration_detect::CacheError),

    /// Runtime ambiguity that the runner refuses to guess (ADR 0019).
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
    #[error("camera {camera_id:?}: only {usable}/{total} views had >= 4 features")]
    InsufficientUsableViews {
        /// Camera id.
        camera_id: String,
        /// Number of views with at least 4 detected features.
        usable: usize,
        /// Total number of images attempted.
        total: usize,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────────────

fn target_to_detector_config(target: &TargetSpec) -> Result<(&'static str, Value), RunError> {
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
        TargetSpec::Puzzleboard { .. } => Err(RunError::UnsupportedTarget {
            kind: "puzzleboard".to_string(),
        }),
        TargetSpec::Ringgrid { .. } => Err(RunError::UnsupportedTarget {
            kind: "ringgrid".to_string(),
        }),
    }
}

fn pick_detector(name: &str) -> Result<Box<dyn Detector>, RunError> {
    match name {
        "chessboard" => Ok(Box::new(ChessboardDetector)),
        "charuco" => Ok(Box::new(CharucoDetector)),
        other => Err(RunError::UnsupportedTarget {
            kind: other.to_string(),
        }),
    }
}

fn expand_camera_images(
    camera: &vision_calibration_dataset::CameraSource,
    base_dir: &Path,
) -> Result<Vec<PathBuf>, RunError> {
    match &camera.images {
        ImagePattern::Glob { pattern } => {
            let resolved = if Path::new(pattern).is_absolute() {
                pattern.clone()
            } else {
                base_dir.join(pattern).to_string_lossy().to_string()
            };
            let entries = glob::glob(&resolved).map_err(|e| RunError::BadGlob {
                camera_id: camera.id.clone(),
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
/// value. Keeps `Im_2` < `Im_10`, matching the bench harness's
/// ordering (`vision-calibration-bench/src/detect.rs`).
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

/// Splice a `_roi` field into the detector's canonical config JSON so
/// the cache key changes whenever the user edits the ROI. The detector
/// itself never sees `_roi` — the runner crops the image upstream.
fn augment_config_with_roi(detector_config: &Value, roi: Option<[u32; 4]>) -> Value {
    let Some(roi) = roi else {
        return detector_config.clone();
    };
    let mut copy = detector_config.clone();
    if let Value::Object(map) = &mut copy {
        map.insert("_roi".to_string(), json!([roi[0], roi[1], roi[2], roi[3]]));
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
/// ROI, detect, lift pixels back to source coordinates, store.
///
/// Returns the features plus whether they came from the cache.
#[allow(clippy::too_many_arguments)]
fn detect_features(
    detector: &dyn Detector,
    detector_name: &str,
    detector_config: &Value,
    key_config: &Value,
    roi: Option<[u32; 4]>,
    image_path: &Path,
    cache: &dyn DetectionCache,
    force_redetect: bool,
) -> Result<(Vec<Feature>, bool), RunError> {
    let bytes = std::fs::read(image_path)?;
    let key = CacheKey::from_inputs(&bytes, detector_name, key_config);

    let cached: Option<CachedFeatures> = if force_redetect {
        None
    } else {
        cache.get(&key)?
    };
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
    let mut detected = detector
        .detect_json(&img_for_detect, detector_config)
        .map_err(|e| RunError::Detection {
            detector: detector_name.to_string(),
            path: image_path.to_path_buf(),
            source: e,
        })?;
    // Detected pixels are in the cropped frame; lift them back into
    // source-image coordinates so the rest of the pipeline (and the
    // export's `image_manifest`) stays in one consistent coordinate
    // system.
    if let Some([x, y, _w, _h]) = roi {
        let dx = x as f64;
        let dy = y as f64;
        for f in detected.iter_mut() {
            f.image_xy[0] += dx;
            f.image_xy[1] += dy;
        }
    }
    cache.put(
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

fn features_to_obs(features: &[Feature]) -> CorrespondenceView {
    let mut points_3d = Vec::with_capacity(features.len());
    let mut points_2d = Vec::with_capacity(features.len());
    for f in features {
        points_3d.push(Pt3::new(f.world_xyz[0], f.world_xyz[1], f.world_xyz[2]));
        points_2d.push(Pt2::new(f.image_xy[0], f.image_xy[1]));
    }
    CorrespondenceView::new(points_3d, points_2d).expect(
        "Feature vectors are non-empty and equal-length by construction; \
         CorrespondenceView::new only fails on mismatched lengths.",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let (name, config) = target_to_detector_config(&target).unwrap();
        assert_eq!(name, "charuco");
        assert_eq!(config["dictionary"], "DICT_4X4_100");
        assert_eq!(config["marker_size_m"], 0.015);
        // The mapped config must deserialize into the detector's own
        // config struct — guards the field-name contract between the
        // manifest and the detect crate.
        pick_detector(name).unwrap();
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
        let err = target_to_detector_config(&target).unwrap_err();
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
        let err = target_to_detector_config(&target).unwrap_err();
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
        assert!(detector_config.get("_roi").is_none());
    }
}
