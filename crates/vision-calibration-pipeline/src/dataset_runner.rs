//! Dataset-driven runner: takes a [`DatasetSpec`] manifest plus a
//! per-problem `*Config` and produces the existing `*Input` IR by
//! running detection (cached) on the per-camera images.
//!
//! PR 1 wires up `PlanarIntrinsics` only. PR 2 extends to the other
//! seven problem types via the same `(spec, config, &cache) → *Input`
//! shape.
//!
//! Per ADR 0019, any ambiguity that cannot be auto-resolved at
//! conversion time is surfaced as a [`RunError::AskUser`] event so
//! the UI can prompt the user instead of silently guessing.

use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use thiserror::Error;

use vision_calibration_core::{CorrespondenceView, NoMeta, PlanarDataset, Pt2, Pt3, View};
use vision_calibration_dataset::{
    DatasetSpec, ImagePattern, TargetSpec, Topology, ValidationError, validate,
};
use vision_calibration_detect::{
    CacheKey, CachedFeatures, ChessboardDetector, DetectionCache, Detector, Feature,
};

/// Errors produced by the dataset-driven runner.
#[derive(Debug, Error)]
pub enum RunError {
    /// Manifest failed structural validation. Wraps the underlying
    /// [`ValidationError`].
    #[error("manifest validation failed: {0}")]
    Validation(#[from] ValidationError),

    /// Topology is not yet supported by the runner. PR 1 ships with
    /// `PlanarIntrinsics` only; PR 2 lifts this restriction.
    #[error("topology {topology:?} is not yet supported by the dataset runner")]
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
        /// Camera id (single camera for `PlanarIntrinsics`).
        camera_id: String,
        /// Number of views with at least 4 detected features.
        usable: usize,
        /// Total number of images attempted.
        total: usize,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a Planar manifest + config + cache into the existing
/// [`PlanarDataset`] IR. Returns the dataset plus a per-view list of
/// the source image paths (for downstream `image_manifest`
/// population).
pub fn build_planar_input(
    spec: &DatasetSpec,
    base_dir: &Path,
    cache: &dyn DetectionCache,
    force_redetect: bool,
) -> Result<PlanarRunResult, RunError> {
    // Order matters: gate on topology + target before touching the
    // filesystem so users with broken globs still see helpful
    // "wrong topology" or "wrong target" errors first.
    if spec.topology != Topology::PlanarIntrinsics {
        return Err(RunError::UnsupportedTopology {
            topology: spec.topology,
        });
    }
    let (detector_name, detector_config) = target_to_detector_config(&spec.target)?;
    let detector = pick_detector(detector_name)?;
    validate(spec)?;

    let camera = spec
        .cameras
        .first()
        .expect("validate guarantees at least one camera for PlanarIntrinsics");
    let images = expand_camera_images(camera, base_dir)?;
    if images.is_empty() {
        return Err(RunError::EmptyImageMatch {
            camera_id: camera.id.clone(),
            pattern: pattern_repr(&camera.images),
            base: base_dir.to_path_buf(),
        });
    }

    let mut views: Vec<View<NoMeta>> = Vec::new();
    let mut view_paths: Vec<PathBuf> = Vec::new();
    let mut usable = 0usize;
    let total = images.len();

    for image_path in &images {
        let bytes = std::fs::read(image_path)?;
        let key = CacheKey::from_inputs(&bytes, detector_name, &detector_config);

        let cached: Option<CachedFeatures> = if force_redetect {
            None
        } else {
            cache.get(&key)?
        };

        let features = match cached {
            Some(entry) => entry.features,
            None => {
                let img = image::load_from_memory(&bytes).map_err(|e| RunError::Decode {
                    path: image_path.clone(),
                    source: e,
                })?;
                let detected = detector.detect_json(&img, &detector_config).map_err(|e| {
                    RunError::Detection {
                        detector: detector_name.to_string(),
                        path: image_path.clone(),
                        source: e,
                    }
                })?;
                cache.put(
                    &key,
                    &CachedFeatures {
                        features: detected.clone(),
                    },
                )?;
                detected
            }
        };

        if features.len() < 4 {
            // Too few features for a homography — drop the view but
            // keep walking so we can surface a meaningful error if
            // _everything_ failed.
            continue;
        }
        usable += 1;
        views.push(features_to_view(&features)?);
        view_paths.push(image_path.clone());
    }

    if views.is_empty() {
        return Err(RunError::InsufficientUsableViews {
            camera_id: camera.id.clone(),
            usable,
            total,
        });
    }

    let dataset = PlanarDataset::new(views).expect(
        "PlanarDataset::new failed after we already filtered to >=4 \
         features per view; this indicates a regression in the core \
         crate's invariants",
    );

    Ok(PlanarRunResult {
        dataset,
        view_paths,
        usable_views: usable,
        total_views: total,
    })
}

/// Result of a Planar dataset conversion.
#[derive(Debug)]
pub struct PlanarRunResult {
    /// The IR ready to feed into `CalibrationSession::set_input`.
    pub dataset: PlanarDataset,
    /// One source image path per accepted view, in the same order.
    /// Used to populate the export's `image_manifest` after the solve.
    pub view_paths: Vec<PathBuf>,
    /// Number of views where detection produced ≥4 features.
    pub usable_views: usize,
    /// Total number of images attempted.
    pub total_views: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
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
        TargetSpec::Charuco { .. } => Err(RunError::UnsupportedTarget {
            kind: "charuco".to_string(),
        }),
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
            paths.sort();
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

fn pattern_repr(p: &ImagePattern) -> String {
    match p {
        ImagePattern::Glob { pattern } => pattern.clone(),
        ImagePattern::List { paths } => format!("<list of {} paths>", paths.len()),
    }
}

fn features_to_view(features: &[Feature]) -> Result<View<NoMeta>, RunError> {
    let mut points_3d = Vec::with_capacity(features.len());
    let mut points_2d = Vec::with_capacity(features.len());
    for f in features {
        points_3d.push(Pt3::new(f.world_xyz[0], f.world_xyz[1], f.world_xyz[2]));
        points_2d.push(Pt2::new(f.image_xy[0], f.image_xy[1]));
    }
    let cv = CorrespondenceView::new(points_3d, points_2d).expect(
        "Feature vectors are non-empty and equal-length by construction; \
         CorrespondenceView::new only fails on mismatched lengths.",
    );
    Ok(View::without_meta(cv))
}

#[cfg(test)]
mod tests {
    use super::*;
    use vision_calibration_dataset::{CameraSource, ImagePattern, TargetSpec, Topology};
    use vision_calibration_detect::FsDetectionCache;

    fn planar_spec_for_globless_test() -> DatasetSpec {
        DatasetSpec {
            version: 1,
            cameras: vec![CameraSource {
                id: "cam0".into(),
                images: ImagePattern::List { paths: vec![] },
                roi_xywh: None,
            }],
            target: TargetSpec::Chessboard {
                rows: 9,
                cols: 6,
                square_size_m: 0.025,
            },
            robot_poses: None,
            topology: Topology::PlanarIntrinsics,
            pose_pairing: None,
            pose_convention: None,
            unresolved: vec![],
            description: None,
        }
    }

    #[test]
    fn rejects_non_planar_topology() {
        let mut spec = planar_spec_for_globless_test();
        spec.topology = Topology::RigExtrinsics;
        spec.cameras.push(CameraSource {
            id: "cam1".into(),
            images: ImagePattern::List { paths: vec![] },
            roi_xywh: None,
        });
        let cache = FsDetectionCache::new(std::env::temp_dir().join("calib-test-cache"));
        let err = build_planar_input(&spec, Path::new("/tmp"), &cache, false).unwrap_err();
        assert!(matches!(err, RunError::UnsupportedTopology { .. }));
    }

    #[test]
    fn rejects_unsupported_target() {
        let mut spec = planar_spec_for_globless_test();
        spec.target = TargetSpec::Charuco {
            rows: 9,
            cols: 6,
            square_size_m: 0.025,
            marker_size_m: 0.018,
            dictionary: "DICT_4X4_50".into(),
        };
        spec.cameras[0].images = ImagePattern::Glob {
            pattern: "*.png".into(),
        };
        let cache = FsDetectionCache::new(std::env::temp_dir().join("calib-test-cache"));
        let err = build_planar_input(&spec, Path::new("/tmp"), &cache, false).unwrap_err();
        assert!(matches!(err, RunError::UnsupportedTarget { .. }));
    }

    #[test]
    fn empty_image_match_is_actionable() {
        let mut spec = planar_spec_for_globless_test();
        spec.cameras[0].images = ImagePattern::Glob {
            pattern: "**/no_such_pattern_should_match_*.png".into(),
        };
        let tmp = tempdir_or_skip();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        let err = build_planar_input(&spec, tmp.path(), &cache, false).unwrap_err();
        match err {
            RunError::EmptyImageMatch { camera_id, .. } => assert_eq!(camera_id, "cam0"),
            other => panic!("expected EmptyImageMatch, got {other:?}"),
        }
    }

    fn tempdir_or_skip() -> tempfile::TempDir {
        tempfile::tempdir().expect("tempdir")
    }
}
