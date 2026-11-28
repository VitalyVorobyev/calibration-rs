//! `DatasetSpec → LaserlineDeviceInput / RigLaserlineDeviceInput`
//! converters for the laser topologies.
//!
//! Laser-line extraction is *injected* via [`LaserPixelExtractor`]:
//! the reference implementation wraps `vision-metrology`, which is not
//! on crates.io, so published crates cannot depend on it. The pipeline
//! owns everything around the extractor call — cache lookup, decode,
//! ROI crop, coordinate lift, cache store — mirroring the target
//! detection path (`detect_features`).
//!
//! Laser frames pair with target views through the manifest's
//! `pose_pairing` (no second pairing config): `by_index` requires one
//! laser image per paired view; `shared_filename_token` applies the
//! same regex to laser filenames. A laser token with no target view is
//! an error (the regex is almost certainly wrong); a target view
//! without a laser image is a gap — `None` laser slot for rigs, a
//! dropped view for the single-camera topology.

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use serde_json::Value;

use vision_calibration_core::{Iso3, Pt2, RunControl, View, data_relative_path, lexical_absolute};
use vision_calibration_dataset::{
    DatasetSpec, ImagePattern, LaserExtractionSpec, PosePairing, ProblemKind, validate,
};
use vision_calibration_detect::{CacheKey, CachedFeatures, Feature, GrayImageView};
use vision_calibration_optim::{
    LaserlineDataset, LaserlineMeta, RigHandeyeLaserlineView, RigLaserlineDataset,
    RigLaserlineView, RobotPoseMeta, handeye_observer_se3_target,
};

use crate::common::read_export;
use crate::laserline_device::LaserlineDeviceInput;
use crate::rig_handeye::RigHandeyeExport;
use crate::rig_handeye_laserline::RigHandeyeLaserlineInput;
use crate::rig_laserline_device::RigLaserlineDeviceInput;

use super::convert_options::{ConvertOptions, kept};
use super::pairing::pair_views;
use super::poses::{load_manifest_robot_poses, match_poses_to_tokens};
use super::require_pose_pairing;
use super::{
    Detection, ImageProgress, RejectReason, RejectedDetection, RunError, decode_gray,
    detect_features, expand_camera_images, expand_image_pattern, features_to_obs, pattern_repr,
    rig::{RigCore, build_rig_core},
    target_to_detector_config,
    view_filter::ViewFilter,
};

/// Injected laser-line extractor.
///
/// Unlike the target detectors, a closed
/// [`DetectorConfig`](vision_calibration_detect::DetectorConfig) set, this
/// trait is **open**: an application supplies the extractor (the desktop
/// app wraps `vision-metrology`, which is not on crates.io). Implementations only extract; caching, ROI
/// cropping, and coordinate lifting are handled by the runner.
pub trait LaserPixelExtractor: Send + Sync {
    /// Stable implementation id. Part of the cache key
    /// (`laser:<name>`), so switching implementations re-extracts
    /// instead of serving another extractor's pixels.
    fn name(&self) -> &str;

    /// Extract subpixel laser-line points `[x, y]` from `image`, the
    /// laser image in 8-bit grayscale (already ROI-cropped by the runner;
    /// coordinates are in the cropped frame).
    fn extract(
        &self,
        image: GrayImageView<'_>,
        spec: &LaserExtractionSpec,
    ) -> Result<Vec<[f64; 2]>, Box<dyn std::error::Error + Send + Sync>>;
}

/// Result of a single-camera laserline dataset conversion.
#[derive(Debug)]
#[non_exhaustive]
pub struct LaserlineRunResult {
    /// The IR ready to feed into `CalibrationSession::set_input`.
    pub input: LaserlineDeviceInput,
    /// One *target* image path per accepted view, in the same order.
    /// Used to populate the export's `image_manifest` after the solve.
    pub view_paths: Vec<PathBuf>,
    /// One *laser* image path per accepted view, aligned with
    /// `view_paths`. Not yet rendered anywhere.
    pub laser_paths: Vec<PathBuf>,
    /// Number of views with an accepted target detection and
    /// ≥`min_points` laser pixels.
    pub usable_views: usize,
    /// Total number of paired views attempted.
    pub total_views: usize,
    /// The target detections left out, in view order, each with its
    /// reason. Views dropped for their laser frame are not listed.
    pub rejected: Vec<RejectedDetection>,
}

/// Result of a rig laserline dataset conversion.
#[derive(Debug)]
#[non_exhaustive]
pub struct RigLaserlineRunResult {
    /// The IR (dataset + frozen upstream calibration) ready to feed
    /// into `CalibrationSession::set_input`.
    pub input: RigLaserlineDeviceInput,
    /// `view_paths[view][camera]` — *target* image path per kept view
    /// and camera; `None` where the camera contributed no usable
    /// target observation.
    pub view_paths: Vec<Vec<Option<PathBuf>>>,
    /// `laser_paths[view][camera]` — *laser* image path per kept view
    /// and camera, aligned with `view_paths`; `None` where the camera
    /// contributed no usable laser observation.
    pub laser_paths: Vec<Vec<Option<PathBuf>>>,
    /// Number of kept views (≥1 camera with an accepted target detection).
    pub usable_views: usize,
    /// Total number of paired views attempted.
    pub total_views: usize,
    /// The target detections left out, in view order, each with its
    /// reason. Laser frames below `min_points` are not listed.
    pub rejected: Vec<RejectedDetection>,
}

/// Result of a joint rig hand-eye laserline dataset conversion.
#[derive(Debug)]
#[non_exhaustive]
pub struct RigHandeyeLaserlineRunResult {
    /// The IR ready to feed into `RigHandeyeLaserlineProblem`.
    pub input: RigHandeyeLaserlineInput,
    /// `view_paths[view][camera]` — target image path per kept view.
    pub view_paths: Vec<Vec<Option<PathBuf>>>,
    /// `laser_paths[view][camera]` — laser image path per kept view.
    pub laser_paths: Vec<Vec<Option<PathBuf>>>,
    /// Number of kept views (≥1 camera with an accepted target detection).
    pub usable_views: usize,
    /// Total number of paired views attempted.
    pub total_views: usize,
    /// The target detections left out, in view order, each with its
    /// reason. Laser frames below `min_points` are not listed.
    pub rejected: Vec<RejectedDetection>,
}

/// Convert a single-camera laserline manifest + cache into a
/// [`LaserlineDeviceInput`] for `LaserlineDevice`.
///
/// `control` as for [`build_planar_input`](super::build_planar_input), and
/// told each laser image done ([`Progress::LaserExtraction`](vision_calibration_core::Progress)).
pub fn build_laserline_device_input(
    spec: &DatasetSpec,
    base_dir: &Path,
    detection: Detection<'_>,
    laser_extractor: &dyn LaserPixelExtractor,
    control: &RunControl,
) -> Result<LaserlineRunResult, RunError> {
    build_laserline_device_input_with(
        spec,
        base_dir,
        detection,
        laser_extractor,
        control,
        &ConvertOptions::default(),
    )
}

/// [`build_laserline_device_input`], leaving out what `options` names. A
/// view left out leaves its laser frame out with it.
pub(crate) fn build_laserline_device_input_with(
    spec: &DatasetSpec,
    base_dir: &Path,
    detection: Detection<'_>,
    laser_extractor: &dyn LaserPixelExtractor,
    control: &RunControl,
    options: &ConvertOptions,
) -> Result<LaserlineRunResult, RunError> {
    // Same gate ordering as the other converters: topology + target +
    // validation + pairing config before any filesystem access.
    if spec.topology != ProblemKind::LaserlineDevice {
        return Err(RunError::UnsupportedTopology {
            topology: spec.topology,
        });
    }
    let detector_config = target_to_detector_config(spec)?;
    validate(spec)?;

    let pairing = require_pose_pairing(spec)?
        .expect("every rig and laser topology needs a pairing; require_pose_pairing asks for it");

    // The validator enforces exactly one camera with laser_images.
    let camera = spec
        .cameras
        .first()
        .expect("validate guarantees exactly one camera for LaserlineDevice");
    let images = expand_camera_images(camera, base_dir)?;
    if images.is_empty() {
        return Err(RunError::EmptyImageMatch {
            camera_id: camera.id.clone(),
            pattern: pattern_repr(&camera.images),
            base: base_dir.to_path_buf(),
        });
    }
    let paired = pair_views(
        std::slice::from_ref(&images),
        std::slice::from_ref(&camera.id),
        pairing,
    )?;
    let total_views = paired.paths.len();

    let laser_images = camera
        .laser_images
        .as_ref()
        .expect("validate guarantees laser_images for LaserlineDevice");
    let laser_by_token =
        laser_paths_by_token(laser_images, &camera.id, pairing, &paired.tokens, base_dir)?;

    let laser_spec = spec.laser.unwrap_or_default();
    let roi = camera.roi_xywh;
    let laser_key_config = laser_key_config(&laser_spec, roi);
    let filter = ViewFilter::for_spec(spec);

    let mut rejected: Vec<RejectedDetection> = Vec::new();
    let mut views: Vec<View<LaserlineMeta>> = Vec::new();
    let mut view_paths: Vec<PathBuf> = Vec::new();
    let mut laser_paths: Vec<PathBuf> = Vec::new();
    let mut targets = ImageProgress::targets(control, paired.paths.len());
    let laser_count = paired
        .tokens
        .iter()
        .filter(|token| laser_by_token.contains_key(*token))
        .count();
    let mut lasers = ImageProgress::lasers(control, laser_count);
    for (paths, token) in paired.paths.iter().zip(&paired.tokens) {
        let path = paths[0]
            .as_ref()
            .expect("single-camera pairing never produces gap slots");
        let (features, screened) = if options.excludes(0, path) {
            targets.skip();
            (Vec::new(), Err(RejectReason::Excluded))
        } else {
            let (mut features, _cache_hit) =
                targets.read(|| detect_features(&detector_config, roi, path, detection))?;
            let mut screened = filter.check(&features);
            if screened.is_ok()
                && let Some(keep) = options.kept_features(0, path, &features)
            {
                features = kept(&features, &keep);
                screened = filter.check_masked(&features);
            }
            (features, screened)
        };
        if let Err(reason) = screened {
            rejected.push(RejectedDetection {
                camera: 0,
                path: path.clone(),
                reason,
            });
            if laser_by_token.contains_key(token) {
                lasers.skip();
            }
            continue;
        }
        // The single-camera problem needs laser pixels in *every* view
        // (LaserlineMeta validates non-empty), so a view without a
        // usable laser frame is dropped rather than gapped.
        let Some(laser_path) = laser_by_token.get(token) else {
            continue;
        };
        let pixels = lasers.read(|| {
            extract_laser_pixels(
                laser_extractor,
                &laser_spec,
                &laser_key_config,
                roi,
                laser_path,
                detection,
            )
        })?;
        if pixels.len() < laser_spec.min_points as usize {
            continue;
        }
        views.push(View::new(
            features_to_obs(&features),
            LaserlineMeta::new(pixels),
        ));
        view_paths.push(path.clone());
        laser_paths.push(laser_path.clone());
    }

    let usable_views = views.len();
    // The problem type needs ≥3 views; fail here with view counts
    // rather than letting set_input produce a less actionable error.
    if usable_views < 3 {
        return Err(RunError::InsufficientUsableViews {
            camera_id: camera.id.clone(),
            usable: usable_views,
            total: total_views,
        });
    }

    let input = LaserlineDataset::new(views).expect(
        "LaserlineDataset::new failed after the runner kept only views with >=4 \
         features and >=2 laser pixels; this indicates a regression in the \
         optim crate's invariants",
    );

    Ok(LaserlineRunResult {
        input,
        view_paths,
        laser_paths,
        usable_views,
        total_views,
        rejected,
    })
}

/// Convert a rig laserline manifest + cache + frozen upstream rig
/// hand-eye export into a [`RigLaserlineDeviceInput`] for
/// `RigLaserlineDevice`.
///
/// `control` as for [`build_planar_input`](super::build_planar_input), and
/// told each laser image done ([`Progress::LaserExtraction`](vision_calibration_core::Progress)).
pub fn build_rig_laserline_device_input(
    spec: &DatasetSpec,
    base_dir: &Path,
    detection: Detection<'_>,
    laser_extractor: &dyn LaserPixelExtractor,
    control: &RunControl,
) -> Result<RigLaserlineRunResult, RunError> {
    build_rig_laserline_device_input_with(
        spec,
        base_dir,
        detection,
        laser_extractor,
        control,
        &ConvertOptions::default(),
    )
}

/// [`build_rig_laserline_device_input`], leaving out what `options` names.
/// A view left out leaves its laser frames out with it.
pub(crate) fn build_rig_laserline_device_input_with(
    spec: &DatasetSpec,
    base_dir: &Path,
    detection: Detection<'_>,
    laser_extractor: &dyn LaserPixelExtractor,
    control: &RunControl,
    options: &ConvertOptions,
) -> Result<RigLaserlineRunResult, RunError> {
    if spec.topology != ProblemKind::RigLaserlineDevice {
        return Err(RunError::UnsupportedTopology {
            topology: spec.topology,
        });
    }
    // The robot poses and the upstream calibration load before any
    // detection, so a file that does not parse fails the run first.
    // `validate` guarantees pose_pairing, robot_poses, pose_convention,
    // upstream_calibration, and laser_images on every camera for this
    // topology.
    validate(spec)?;
    let poses = load_manifest_robot_poses(spec, base_dir)?;
    let upstream_path = spec
        .upstream_calibration
        .as_ref()
        .expect("validate guarantees upstream_calibration for RigLaserlineDevice");
    let upstream_path = if upstream_path.is_absolute() {
        upstream_path.clone()
    } else {
        base_dir.join(upstream_path)
    };
    let export = load_upstream_export(&upstream_path)?;

    let core = build_rig_core(spec, base_dir, detection, control, options)?;
    let num_cameras = spec.cameras.len();
    let laser = extract_rig_laser(spec, base_dir, &core, detection, laser_extractor, control)?;

    // Robot poses → per-kept-view T_B_G, matched by pairing token.
    let kept_tokens: Vec<String> = core.views.iter().map(|(_, t)| t.clone()).collect();
    let matched = match_poses_to_tokens(&kept_tokens, &poses, spec, core.total_views)?;

    // Frozen upstream rig hand-eye export → per-view rig_se3_target.
    // Prefer the upstream export's optimized target poses; older exports
    // fall back to the hand-eye chain.
    if export.cameras.len() != num_cameras {
        return Err(RunError::UpstreamCalibration {
            path: upstream_path,
            message: format!(
                "export has {} cameras but the manifest declares {num_cameras}",
                export.cameras.len()
            ),
        });
    }
    // Each kept view's place among the upstream run's views, when the
    // upstream export records its views' images.
    let upstream_views = match_upstream_views(&export, &upstream_path, spec, base_dir, &core)?;
    let rig_se3_target: Vec<Iso3> = matched
        .iter()
        .zip(kept_tokens.iter())
        .enumerate()
        .map(|(view_idx, (base_se3_gripper, token))| {
            let view = match &upstream_views {
                Some(indices) => UpstreamView::Recorded(indices[view_idx]),
                None => UpstreamView::Unrecorded {
                    token,
                    kept_idx: view_idx,
                    kept_views: core.views.len(),
                    total_views: core.total_views,
                },
            };
            rig_se3_target_from_upstream(&export, base_se3_gripper, view, &upstream_path)
        })
        .collect::<Result<_, _>>()?;
    let upstream = export
        .to_upstream_calibration(rig_se3_target)
        .map_err(|e| RunError::UpstreamCalibration {
            path: upstream_path,
            message: e.to_string(),
        })?;

    let views: Vec<RigLaserlineView> = core
        .views
        .into_iter()
        .zip(laser.pixels)
        .map(|((obs, _token), laser_pixels)| RigLaserlineView::new(obs.cameras, laser_pixels))
        .collect();
    let usable_views = views.len();
    let dataset = RigLaserlineDataset::new(views, num_cameras).expect(
        "RigLaserlineDataset::new failed after the runner enforced non-empty views \
         and per-view camera counts; this indicates an optim-crate regression",
    );

    Ok(RigLaserlineRunResult {
        input: RigLaserlineDeviceInput { dataset, upstream },
        view_paths: core.view_paths,
        laser_paths: laser.paths,
        usable_views,
        total_views: core.total_views,
        rejected: core.rejected,
    })
}

/// Convert a manifest into a joint rig hand-eye laserline input.
///
/// `control` as for [`build_planar_input`](super::build_planar_input), and
/// told each laser image done ([`Progress::LaserExtraction`](vision_calibration_core::Progress)).
pub fn build_rig_handeye_laserline_input(
    spec: &DatasetSpec,
    base_dir: &Path,
    detection: Detection<'_>,
    laser_extractor: &dyn LaserPixelExtractor,
    control: &RunControl,
) -> Result<RigHandeyeLaserlineRunResult, RunError> {
    build_rig_handeye_laserline_input_with(
        spec,
        base_dir,
        detection,
        laser_extractor,
        control,
        &ConvertOptions::default(),
    )
}

/// [`build_rig_handeye_laserline_input`], leaving out what `options` names.
/// A view left out leaves its robot pose and laser frames out with it.
pub(crate) fn build_rig_handeye_laserline_input_with(
    spec: &DatasetSpec,
    base_dir: &Path,
    detection: Detection<'_>,
    laser_extractor: &dyn LaserPixelExtractor,
    control: &RunControl,
    options: &ConvertOptions,
) -> Result<RigHandeyeLaserlineRunResult, RunError> {
    if spec.topology != ProblemKind::RigHandeyeLaserline {
        return Err(RunError::UnsupportedTopology {
            topology: spec.topology,
        });
    }
    // The robot poses load before any detection, so a pose file that does
    // not parse fails the run first.
    validate(spec)?;
    let poses = load_manifest_robot_poses(spec, base_dir)?;

    let core = build_rig_core(spec, base_dir, detection, control, options)?;
    let num_cameras = spec.cameras.len();
    let laser = extract_rig_laser(spec, base_dir, &core, detection, laser_extractor, control)?;

    let kept_tokens: Vec<String> = core.views.iter().map(|(_, t)| t.clone()).collect();
    let matched = match_poses_to_tokens(&kept_tokens, &poses, spec, core.total_views)?;

    let views: Vec<RigHandeyeLaserlineView> = core
        .views
        .into_iter()
        .zip(laser.pixels)
        .zip(matched)
        .map(|(((obs, _token), laser_pixels), base_se3_gripper)| {
            RigHandeyeLaserlineView::new(
                RigLaserlineView::new(obs.cameras, laser_pixels),
                RobotPoseMeta::new(base_se3_gripper),
            )
        })
        .collect();
    let usable_views = views.len();
    Ok(RigHandeyeLaserlineRunResult {
        input: RigHandeyeLaserlineInput { views, num_cameras },
        view_paths: core.view_paths,
        laser_paths: laser.paths,
        usable_views,
        total_views: core.total_views,
        rejected: core.rejected,
    })
}

/// A rig's laser frames over its kept views, `[view][camera]`: the laser
/// pixels and the laser image each came from. A slot is `None` where the
/// camera has no laser image for the view or fewer than `min_points` pixels.
struct RigLaser {
    pixels: Vec<Vec<Option<Vec<Pt2>>>>,
    paths: Vec<Vec<Option<PathBuf>>>,
}

/// Pair every camera's laser images with the rig's views and extract the
/// laser pixels of the kept views.
///
/// Laser frames pair against *all* view tokens (pre-drop): a laser image
/// for a view that was later dropped is not an error. A camera left with no
/// usable laser frame fails, since its plane would be unconstrained.
fn extract_rig_laser(
    spec: &DatasetSpec,
    base_dir: &Path,
    core: &RigCore,
    detection: Detection<'_>,
    laser_extractor: &dyn LaserPixelExtractor,
    control: &RunControl,
) -> Result<RigLaser, RunError> {
    let num_cameras = spec.cameras.len();
    let pairing = spec
        .pose_pairing
        .as_ref()
        .expect("build_rig_core checked pose_pairing");

    let laser_spec = spec.laser.unwrap_or_default();
    let mut laser_by_token: Vec<HashMap<String, PathBuf>> = Vec::with_capacity(num_cameras);
    for camera in &spec.cameras {
        let laser_images = camera
            .laser_images
            .as_ref()
            .expect("validate guarantees laser_images on every camera of a rig laser topology");
        laser_by_token.push(laser_paths_by_token(
            laser_images,
            &camera.id,
            pairing,
            &core.all_tokens,
            base_dir,
        )?);
    }

    let laser_key_configs: Vec<Value> = spec
        .cameras
        .iter()
        .map(|c| laser_key_config(&laser_spec, c.roi_xywh))
        .collect();
    let mut pixels: Vec<Vec<Option<Vec<Pt2>>>> = Vec::with_capacity(core.views.len());
    let mut paths: Vec<Vec<Option<PathBuf>>> = Vec::with_capacity(core.views.len());
    let mut per_camera_usable = vec![0usize; num_cameras];
    let laser_count = core
        .views
        .iter()
        .map(|(_obs, token)| {
            laser_by_token
                .iter()
                .filter(|by| by.contains_key(token))
                .count()
        })
        .sum();
    let mut lasers = ImageProgress::lasers(control, laser_count);
    for (_obs, token) in &core.views {
        let mut slots: Vec<Option<Vec<Pt2>>> = Vec::with_capacity(num_cameras);
        let mut path_slots: Vec<Option<PathBuf>> = Vec::with_capacity(num_cameras);
        for cam_idx in 0..num_cameras {
            let Some(laser_path) = laser_by_token[cam_idx].get(token) else {
                slots.push(None);
                path_slots.push(None);
                continue;
            };
            let view_pixels = lasers.read(|| {
                extract_laser_pixels(
                    laser_extractor,
                    &laser_spec,
                    &laser_key_configs[cam_idx],
                    spec.cameras[cam_idx].roi_xywh,
                    laser_path,
                    detection,
                )
            })?;
            if view_pixels.len() < laser_spec.min_points as usize {
                slots.push(None);
                path_slots.push(None);
                continue;
            }
            per_camera_usable[cam_idx] += 1;
            slots.push(Some(view_pixels));
            path_slots.push(Some(laser_path.clone()));
        }
        pixels.push(slots);
        paths.push(path_slots);
    }
    for (cam_idx, usable) in per_camera_usable.iter().enumerate() {
        if *usable == 0 {
            return Err(RunError::InsufficientLaserViews {
                camera_id: spec.cameras[cam_idx].id.clone(),
                usable: 0,
                total: core.views.len(),
            });
        }
    }
    Ok(RigLaser { pixels, paths })
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Cache-key config for laser extraction: the canonical
/// `LaserExtractionSpec` JSON plus the crop the pixels were extracted in.
fn laser_key_config(laser_spec: &LaserExtractionSpec, roi: Option<[u32; 4]>) -> Value {
    let mut config =
        serde_json::to_value(laser_spec).expect("LaserExtractionSpec serializes to a JSON object");
    if let (Some(roi), Value::Object(map)) = (roi, &mut config) {
        map.insert("_roi".to_string(), serde_json::json!(roi));
        map.insert(
            "_coord_frame".to_string(),
            serde_json::json!("roi_local_v2"),
        );
    }
    config
}

/// Map each view token to the camera's laser image path, using the
/// same pairing strategy as the target frames.
fn laser_paths_by_token(
    laser_images: &ImagePattern,
    camera_id: &str,
    pairing: &PosePairing,
    view_tokens: &[String],
    base_dir: &Path,
) -> Result<HashMap<String, PathBuf>, RunError> {
    let paths = expand_image_pattern(laser_images, camera_id, base_dir)?;
    if paths.is_empty() {
        return Err(RunError::LaserPairing {
            message: format!(
                "camera {camera_id:?}: laser_images {} matched no files under {}",
                pattern_repr(laser_images),
                base_dir.display()
            ),
        });
    }
    match pairing {
        PosePairing::ByIndex => {
            if paths.len() != view_tokens.len() {
                return Err(RunError::LaserPairing {
                    message: format!(
                        "camera {camera_id:?}: by_index pairing requires one laser image \
                         per view, got {} laser images for {} views",
                        paths.len(),
                        view_tokens.len()
                    ),
                });
            }
            Ok(view_tokens.iter().cloned().zip(paths).collect())
        }
        PosePairing::SharedFilenameToken { regex, group } => {
            let re = regex::Regex::new(regex).map_err(|e| RunError::BadPairingRegex {
                regex: regex.clone(),
                message: e.to_string(),
            })?;
            let known: std::collections::HashSet<&str> =
                view_tokens.iter().map(String::as_str).collect();
            let mut map = HashMap::new();
            for path in paths {
                let name = path
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default();
                let token = re
                    .captures(&name)
                    .and_then(|c| c.name(group))
                    .map(|m| m.as_str().to_string())
                    .ok_or_else(|| RunError::LaserPairing {
                        message: format!(
                            "camera {camera_id:?}: laser filename {name:?} does not match \
                             the pairing regex (or group {group:?} did not participate)"
                        ),
                    })?;
                if !known.contains(token.as_str()) {
                    return Err(RunError::LaserPairing {
                        message: format!(
                            "camera {camera_id:?}: laser token {token:?} (from {name:?}) \
                             matches no target view — the pairing regex is probably wrong"
                        ),
                    });
                }
                if let Some(previous) = map.insert(token.clone(), path.clone()) {
                    return Err(RunError::LaserPairing {
                        message: format!(
                            "camera {camera_id:?}: laser token {token:?} extracted from two \
                             files ({} and {})",
                            previous.display(),
                            path.display()
                        ),
                    });
                }
            }
            Ok(map)
        }
        other => Err(RunError::UnsupportedOption {
            option: "pairing",
            value: format!("{other:?}"),
        }),
    }
}

/// Run one laser image through the cache-or-extract path: read bytes →
/// cache lookup → on miss, decode, crop to ROI, extract in the camera
/// pixel frame, store. Mirrors `detect_features`; laser pixels are
/// cached as `Feature`s with a zero-filled (meaningless) `world_xyz`.
fn extract_laser_pixels(
    extractor: &dyn LaserPixelExtractor,
    laser_spec: &LaserExtractionSpec,
    key_config: &Value,
    roi: Option<[u32; 4]>,
    image_path: &Path,
    detection: Detection<'_>,
) -> Result<Vec<Pt2>, RunError> {
    let bytes = std::fs::read(image_path)?;
    let detector_name = format!("laser:{}", extractor.name());
    let key = CacheKey::from_inputs(&bytes, &detector_name, key_config);
    detection.note_image(image_path, &key);

    let cached: Option<CachedFeatures> = detection.lookup(&key)?;
    if let Some(entry) = cached {
        return Ok(entry
            .features
            .iter()
            .map(|f| Pt2::new(f.image_xy[0], f.image_xy[1]))
            .collect());
    }

    let gray = decode_gray(&bytes, image_path, roi)?;
    let view = GrayImageView::new(gray.width(), gray.height(), gray.as_raw())
        .expect("an image buffer holds width × height bytes");
    let points = extractor
        .extract(view, laser_spec)
        .map_err(|e| RunError::LaserExtraction {
            path: image_path.to_path_buf(),
            source: e,
        })?;
    detection.store(
        &key,
        &CachedFeatures::new(
            points
                .iter()
                .map(|p| Feature::new(*p, [0.0, 0.0, 0.0]))
                .collect(),
        ),
    )?;
    Ok(points.into_iter().map(|p| Pt2::new(p[0], p[1])).collect())
}

fn load_upstream_export(path: &Path) -> Result<RigHandeyeExport, RunError> {
    let text = std::fs::read_to_string(path).map_err(|e| RunError::UpstreamCalibration {
        path: path.to_path_buf(),
        message: format!("read failed: {e}"),
    })?;
    let not_an_export = |e: &dyn std::fmt::Display| RunError::UpstreamCalibration {
        path: path.to_path_buf(),
        message: format!("not a rig hand-eye export: {e}"),
    };
    read_export(&text).map_err(|e| not_an_export(&e))
}

/// Where a laser view sits among the upstream run's views.
#[derive(Debug, Clone, Copy)]
enum UpstreamView<'a> {
    /// The upstream export records its views' images, and this view is
    /// its view at this index.
    Recorded(usize),
    /// An upstream export without an image manifest: the view is placed by
    /// its numeric token among all paired views, or by its position among
    /// the kept views, whichever the export's lengths allow.
    Unrecorded {
        token: &'a str,
        kept_idx: usize,
        kept_views: usize,
        total_views: usize,
    },
}

/// Each kept view's index among the upstream run's views, matched through
/// the upstream export's image manifest; `None` when the export has none.
///
/// A view matches an upstream view that shows one of its target images
/// (the same file, or the same path within the dataset folder) or, under
/// `shared_filename_token` pairing, whose images carry its token. It must
/// match exactly one: none means the upstream run left the view out, two
/// or more that the match is ambiguous; both are
/// [`RunError::UpstreamViewMatch`].
fn match_upstream_views(
    export: &RigHandeyeExport,
    upstream_path: &Path,
    spec: &DatasetSpec,
    base_dir: &Path,
    core: &RigCore,
) -> Result<Option<Vec<usize>>, RunError> {
    let Some(manifest) = export.diagnostics.image_manifest.as_ref() else {
        return Ok(None);
    };
    if manifest.target_frames().next().is_none() {
        return Ok(None);
    }
    let upstream_dir = upstream_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let upstream_root = upstream_dir.join(&manifest.root);
    let token_regex = match spec.pose_pairing.as_ref() {
        Some(PosePairing::SharedFilenameToken { regex, group }) => Some((
            regex::Regex::new(regex).map_err(|e| RunError::BadPairingRegex {
                regex: regex.clone(),
                message: e.to_string(),
            })?,
            group.as_str(),
        )),
        _ => None,
    };
    let token_of = |path: &Path| -> Option<String> {
        let (re, group) = token_regex.as_ref()?;
        let name = path.file_name()?.to_string_lossy();
        re.captures(&name)
            .and_then(|c| c.name(group))
            .map(|m| m.as_str().to_string())
    };

    // Every way an upstream view is recognised, each with the views it
    // names.
    let mut by_file: HashMap<PathBuf, BTreeSet<usize>> = HashMap::new();
    let mut by_data_path: HashMap<PathBuf, BTreeSet<usize>> = HashMap::new();
    let mut by_token: HashMap<String, BTreeSet<usize>> = HashMap::new();
    for frame in manifest.target_frames() {
        by_file
            .entry(lexical_absolute(&upstream_root.join(&frame.path)))
            .or_default()
            .insert(frame.view);
        by_data_path
            .entry(frame.path.clone())
            .or_default()
            .insert(frame.view);
        if let Some(token) = token_of(&frame.path) {
            by_token.entry(token).or_default().insert(frame.view);
        }
    }

    let paired: HashMap<&str, &Vec<Option<PathBuf>>> = core
        .all_tokens
        .iter()
        .map(String::as_str)
        .zip(&core.all_paths)
        .collect();
    core.views
        .iter()
        .map(|(_, token)| {
            let mut matches: BTreeSet<usize> = BTreeSet::new();
            let images = paired
                .get(token.as_str())
                .into_iter()
                .flat_map(|paths| paths.iter().flatten());
            for image in images {
                let keys = [
                    by_file.get(&lexical_absolute(image)),
                    by_data_path.get(&data_relative_path(base_dir, image)),
                ];
                matches.extend(keys.into_iter().flatten().flatten());
            }
            if token_regex.is_some() {
                matches.extend(by_token.get(token).into_iter().flatten());
            }
            match (matches.len(), matches.first()) {
                (1, Some(&view)) => Ok(view),
                (n, _) => Err(RunError::UpstreamViewMatch {
                    path: upstream_path.to_path_buf(),
                    token: token.clone(),
                    matches: n,
                }),
            }
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

/// Per-view rig→target pose: the upstream export's own board pose of the
/// view when it lists one per view, otherwise through the frozen hand-eye
/// chain:
///
/// - `EyeInHand`:  `T_R_T = T_G_R⁻¹ · T_B_G⁻¹ · T_B_T`
/// - `EyeToHand`:  `T_R_T = T_R_B · T_B_G · T_G_T`
fn rig_se3_target_from_upstream(
    export: &RigHandeyeExport,
    base_se3_gripper: &Iso3,
    view: UpstreamView<'_>,
    path: &Path,
) -> Result<Iso3, RunError> {
    if !export.rig_se3_target.is_empty() {
        let poses = &export.rig_se3_target;
        let index = upstream_index(view, poses.len(), "rig_se3_target", path)?;
        return poses
            .get(index)
            .copied()
            .ok_or_else(|| RunError::UpstreamCalibration {
                path: path.to_path_buf(),
                message: format!(
                    "view index {index} is outside upstream rig_se3_target length {}",
                    poses.len()
                ),
            });
    }

    let delta = robot_delta_for_view(export, view, path)?;
    let chain = handeye_observer_se3_target(
        export.handeye.mode(),
        export.handeye.handeye().pose(),
        export.handeye.target_pose(),
        std::slice::from_ref(base_se3_gripper),
        delta.as_ref().map(std::slice::from_ref),
    );
    Ok(chain[0])
}

fn robot_delta_for_view(
    export: &RigHandeyeExport,
    view: UpstreamView<'_>,
    path: &Path,
) -> Result<Option<[f64; 6]>, RunError> {
    let Some(deltas) = export.robot_refinement.as_ref().map(|r| &r.deltas) else {
        return Ok(None);
    };
    let index = upstream_index(view, deltas.len(), "robot_refinement.deltas", path)?;
    deltas
        .get(index)
        .copied()
        .map(Some)
        .ok_or_else(|| RunError::UpstreamCalibration {
            path: path.to_path_buf(),
            message: format!(
                "view index {index} is outside upstream robot_refinement.deltas length {}",
                deltas.len()
            ),
        })
}

/// The index of `view` into an upstream per-view list of `len` entries
/// named `field`. A recorded view is its own index; an unrecorded one is
/// placed by its numeric token when the list covers every paired view, by
/// its position when it covers the kept views.
fn upstream_index(
    view: UpstreamView<'_>,
    len: usize,
    field: &str,
    path: &Path,
) -> Result<usize, RunError> {
    let (token, kept_idx, kept_views, total_views) = match view {
        UpstreamView::Recorded(index) => return Ok(index),
        UpstreamView::Unrecorded {
            token,
            kept_idx,
            kept_views,
            total_views,
        } => (token, kept_idx, kept_views, total_views),
    };
    if len == total_views {
        return token
            .parse::<usize>()
            .map_err(|_| RunError::UpstreamCalibration {
                path: path.to_path_buf(),
                message: format!(
                    "upstream {field} has {total_views} entries but view token {token:?} \
                     is not an index"
                ),
            });
    }
    if len == kept_views {
        return Ok(kept_idx);
    }
    Err(RunError::UpstreamCalibration {
        path: path.to_path_buf(),
        message: format!(
            "upstream {field} has {len} entries, but this run has {kept_views} kept \
             views and {total_views} total paired views"
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{HandeyeSolution, ProblemKind, VersionedExport};
    use std::io::Write;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use vision_calibration_core::{
        BrownConrady5, FxFyCxCy, make_pinhole_camera, pinhole_camera_params,
    };
    use vision_calibration_dataset::{
        CameraSource, PoseConvention, RobotPoseFormat, RobotPoseSource, RotationFormat, TargetSpec,
        TransformConvention, TranslationUnits,
    };
    use vision_calibration_detect::DetectionCache;
    use vision_calibration_detect::FsDetectionCache;

    /// Deterministic extractor: `n` points along a horizontal line.
    /// Counts calls so tests can assert the cache-hit path.
    struct FakeLaser {
        n: usize,
        calls: AtomicUsize,
    }

    impl FakeLaser {
        fn new(n: usize) -> Self {
            Self {
                n,
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl LaserPixelExtractor for FakeLaser {
        fn name(&self) -> &str {
            "fake"
        }
        fn extract(
            &self,
            _image: GrayImageView<'_>,
            _spec: &LaserExtractionSpec,
        ) -> Result<Vec<[f64; 2]>, Box<dyn std::error::Error + Send + Sync>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok((0..self.n).map(|i| [i as f64, 100.0]).collect())
        }
    }

    /// Every corner of the 9-row × 6-column test board — a detection that
    /// spans the whole board as declared.
    fn board_features() -> Vec<Feature> {
        (0..9u32)
            .flat_map(|v| (0..6u32).map(move |u| (u, v)))
            .map(|(u, v)| {
                Feature::new(
                    [100.0 + 40.0 * f64::from(u), 50.0 + 40.0 * f64::from(v)],
                    [0.025 * f64::from(u), 0.025 * f64::from(v), 0.0],
                )
            })
            .collect()
    }

    fn detector_config() -> vision_calibration_detect::DetectorConfig {
        vision_calibration_detect::DetectorConfig::Chessboard(
            vision_calibration_detect::ChessboardConfig::new(9, 6, 0.025),
        )
    }

    fn laser_features(n: usize) -> Vec<Feature> {
        (0..n)
            .map(|i| Feature::new([i as f64, 50.0], [0.0, 0.0, 0.0]))
            .collect()
    }

    /// Write a dummy "image" file and pre-seed the target-detection
    /// cache for it, so the runner takes the cache-hit path and never
    /// decodes the bytes.
    fn seed_target(
        dir: &Path,
        cache: &FsDetectionCache,
        rel_path: &str,
        features: Vec<Feature>,
    ) -> PathBuf {
        let path = dir.join(rel_path);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let bytes = rel_path.as_bytes();
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(bytes).unwrap();
        let key = CacheKey::for_detection(bytes, &detector_config(), None);
        cache.put(&key, &CachedFeatures::new(features)).unwrap();
        path
    }

    /// Write a dummy laser "image" and pre-seed the laser cache under
    /// the `laser:fake` namespace with default extraction params.
    fn seed_laser(
        dir: &Path,
        cache: &FsDetectionCache,
        rel_path: &str,
        features: Vec<Feature>,
    ) -> PathBuf {
        let path = dir.join(rel_path);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let bytes = rel_path.as_bytes();
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(bytes).unwrap();
        let key_config = laser_key_config(&LaserExtractionSpec::default(), None);
        let key = CacheKey::from_inputs(bytes, "laser:fake", &key_config);
        cache.put(&key, &CachedFeatures::new(features)).unwrap();
        path
    }

    fn laserline_spec(image_paths: &[&str], laser_paths: &[&str]) -> DatasetSpec {
        let mut camera = CameraSource::new(
            "cam0",
            ImagePattern::List {
                paths: image_paths.iter().map(PathBuf::from).collect(),
            },
        );
        camera.laser_images = Some(ImagePattern::List {
            paths: laser_paths.iter().map(PathBuf::from).collect(),
        });
        let mut spec = DatasetSpec::new(
            vec![camera],
            TargetSpec::chessboard(9, 6, 0.025),
            ProblemKind::LaserlineDevice,
        );
        spec.pose_pairing = Some(PosePairing::ByIndex);
        spec
    }

    #[test]
    fn by_index_builds_laserline_input() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        for view in 0..3 {
            seed_target(
                tmp.path(),
                &cache,
                &format!("target_{view}.png"),
                board_features(),
            );
            seed_laser(
                tmp.path(),
                &cache,
                &format!("laser_{view}.png"),
                laser_features(30),
            );
        }
        let spec = laserline_spec(
            &["target_0.png", "target_1.png", "target_2.png"],
            &["laser_0.png", "laser_1.png", "laser_2.png"],
        );
        let fake = FakeLaser::new(30);
        let result = build_laserline_device_input(
            &spec,
            tmp.path(),
            Detection::Cached(&cache),
            &fake,
            &RunControl::default(),
        )
        .unwrap();
        assert_eq!((result.usable_views, result.total_views), (3, 3));
        assert_eq!(result.input.num_views(), 3);
        assert_eq!(result.input.views[0].meta.laser_pixels.len(), 30);
        assert!(result.input.views[0].meta.laser_weights.is_empty());
        assert!(result.view_paths[2].ends_with("target_2.png"));
        assert!(result.laser_paths[2].ends_with("laser_2.png"));
        // Everything was served from the cache.
        assert_eq!(fake.calls.load(Ordering::SeqCst), 0);
    }

    /// An excluded view leaves its laser frame out with it: neither image
    /// is read, and the laser progress still reaches its total.
    #[test]
    fn an_excluded_laser_view_leaves_its_laser_frame_out() {
        use std::sync::{Arc, Mutex};
        use vision_calibration_core::Progress;
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        for view in 0..4 {
            seed_target(
                tmp.path(),
                &cache,
                &format!("target_{view}.png"),
                board_features(),
            );
            seed_laser(
                tmp.path(),
                &cache,
                &format!("laser_{view}.png"),
                laser_features(30),
            );
        }
        for name in ["target_1.png", "laser_1.png"] {
            std::fs::write(tmp.path().join(name), b"not an image").unwrap();
        }
        let spec = laserline_spec(
            &[
                "target_0.png",
                "target_1.png",
                "target_2.png",
                "target_3.png",
            ],
            &["laser_0.png", "laser_1.png", "laser_2.png", "laser_3.png"],
        );
        let mut options = ConvertOptions::default();
        options.exclude_slot(0, &tmp.path().join("target_1.png"));
        let reports = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&reports);
        let control = RunControl::default().with_observer(move |p| {
            if let Progress::LaserExtraction { done, total, .. } = p {
                seen.lock().unwrap().push((done, total));
            }
        });
        let fake = FakeLaser::new(30);
        let result = build_laserline_device_input_with(
            &spec,
            tmp.path(),
            Detection::Cached(&cache),
            &fake,
            &control,
            &options,
        )
        .unwrap();
        assert_eq!((result.usable_views, result.total_views), (3, 4));
        assert_eq!(result.rejected[0].reason, RejectReason::Excluded);
        let lasers: Vec<&str> = result
            .laser_paths
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap())
            .collect();
        assert_eq!(lasers, ["laser_0.png", "laser_2.png", "laser_3.png"]);
        assert_eq!(fake.calls.load(Ordering::SeqCst), 0);
        assert_eq!(reports.lock().unwrap().last(), Some(&(4, 4)));
    }

    #[test]
    fn weak_laser_view_dropped() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        for view in 0..4 {
            seed_target(
                tmp.path(),
                &cache,
                &format!("target_{view}.png"),
                board_features(),
            );
            // View 1 has too few laser pixels (< default min_points 20).
            let n = if view == 1 { 5 } else { 30 };
            seed_laser(
                tmp.path(),
                &cache,
                &format!("laser_{view}.png"),
                laser_features(n),
            );
        }
        let spec = laserline_spec(
            &[
                "target_0.png",
                "target_1.png",
                "target_2.png",
                "target_3.png",
            ],
            &["laser_0.png", "laser_1.png", "laser_2.png", "laser_3.png"],
        );
        let fake = FakeLaser::new(30);
        let result = build_laserline_device_input(
            &spec,
            tmp.path(),
            Detection::Cached(&cache),
            &fake,
            &RunControl::default(),
        )
        .unwrap();
        assert_eq!((result.usable_views, result.total_views), (3, 4));
        assert!(
            result
                .view_paths
                .iter()
                .all(|p| !p.ends_with("target_1.png"))
        );
    }

    #[test]
    fn by_index_laser_count_mismatch_is_error() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        for view in 0..3 {
            seed_target(
                tmp.path(),
                &cache,
                &format!("target_{view}.png"),
                board_features(),
            );
        }
        seed_laser(tmp.path(), &cache, "laser_0.png", laser_features(30));
        let spec = laserline_spec(
            &["target_0.png", "target_1.png", "target_2.png"],
            &["laser_0.png"],
        );
        let fake = FakeLaser::new(30);
        let err = build_laserline_device_input(
            &spec,
            tmp.path(),
            Detection::Cached(&cache),
            &fake,
            &RunControl::default(),
        )
        .unwrap_err();
        match err {
            RunError::LaserPairing { message } => {
                assert!(
                    message.contains("1 laser images for 3 views"),
                    "got: {message}"
                );
            }
            other => panic!("expected LaserPairing, got {other:?}"),
        }
    }

    #[test]
    fn token_pairing_gaps_drop_views_and_orphans_error() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        for view in 0..4 {
            seed_target(
                tmp.path(),
                &cache,
                &format!("target_{view}.png"),
                board_features(),
            );
        }
        // Laser frames only for views 0, 2, 3 — view 1 is a gap.
        for view in [0usize, 2, 3] {
            seed_laser(
                tmp.path(),
                &cache,
                &format!("laser_{view}.png"),
                laser_features(30),
            );
        }
        let mut spec = laserline_spec(
            &[
                "target_0.png",
                "target_1.png",
                "target_2.png",
                "target_3.png",
            ],
            &["laser_0.png", "laser_2.png", "laser_3.png"],
        );
        spec.pose_pairing = Some(PosePairing::SharedFilenameToken {
            regex: r"^(?:target|laser)_(?<view>\d+)\.png$".into(),
            group: "view".into(),
        });
        let fake = FakeLaser::new(30);
        let result = build_laserline_device_input(
            &spec,
            tmp.path(),
            Detection::Cached(&cache),
            &fake,
            &RunControl::default(),
        )
        .unwrap();
        assert_eq!((result.usable_views, result.total_views), (3, 4));

        // An orphan laser token (no matching target view) must error.
        seed_laser(tmp.path(), &cache, "laser_9.png", laser_features(30));
        if let Some(CameraSource { laser_images, .. }) = spec.cameras.first_mut() {
            *laser_images = Some(ImagePattern::List {
                paths: ["laser_0.png", "laser_2.png", "laser_3.png", "laser_9.png"]
                    .iter()
                    .map(PathBuf::from)
                    .collect(),
            });
        }
        let err = build_laserline_device_input(
            &spec,
            tmp.path(),
            Detection::Cached(&cache),
            &fake,
            &RunControl::default(),
        )
        .unwrap_err();
        match err {
            RunError::LaserPairing { message } => {
                assert!(message.contains("\"9\""), "got: {message}");
            }
            other => panic!("expected LaserPairing, got {other:?}"),
        }
    }

    #[test]
    fn real_extraction_path_keeps_roi_local_pixels_and_caches() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        // A real decodable PNG so the miss path reaches the extractor.
        let img_path = tmp.path().join("laser_real.png");
        image::DynamicImage::new_luma8(64, 64)
            .save(&img_path)
            .unwrap();

        let laser_spec = LaserExtractionSpec::default();
        let roi = Some([10u32, 20, 16, 16]);
        let key_config = laser_key_config(&laser_spec, roi);
        let fake = FakeLaser::new(25);

        let pixels = extract_laser_pixels(
            &fake,
            &laser_spec,
            &key_config,
            roi,
            &img_path,
            Detection::Cached(&cache),
        )
        .unwrap();
        assert_eq!(pixels.len(), 25);
        // FakeLaser emits x = i, y = 100 in the cropped frame; the
        // runner keeps that ROI-local camera coordinate frame.
        assert_eq!((pixels[0].x, pixels[0].y), (0.0, 100.0));
        assert_eq!(fake.calls.load(Ordering::SeqCst), 1);

        // Second call must be served from the cache, ROI-local frame preserved.
        let again = extract_laser_pixels(
            &fake,
            &laser_spec,
            &key_config,
            roi,
            &img_path,
            Detection::Cached(&cache),
        )
        .unwrap();
        assert_eq!(fake.calls.load(Ordering::SeqCst), 1);
        assert_eq!((again[0].x, again[0].y), (0.0, 100.0));
    }

    // ── Rig converter ───────────────────────────────────────────────────

    fn identity_pose_line(tx: f64) -> String {
        format!("1 0 0 {tx}  0 1 0 0  0 0 1 0  0 0 0 1\n")
    }

    /// Minimal eye-in-hand rig hand-eye export with identity hand-eye
    /// transforms and a target 1 m up the base Z axis.
    fn upstream_export(num_cameras: usize) -> RigHandeyeExport {
        let k = FxFyCxCy {
            fx: 800.0,
            fy: 800.0,
            cx: 320.0,
            cy: 240.0,
        };
        RigHandeyeExport {
            kind: ProblemKind::RigHandeye,
            schema_version: RigHandeyeExport::SCHEMA_VERSION,
            cameras: (0..num_cameras)
                .map(|_| pinhole_camera_params(&make_pinhole_camera(k, BrownConrady5::default())))
                .collect(),
            cam_se3_rig: vec![Iso3::identity(); num_cameras],
            rig_se3_target: vec![],
            handeye: HandeyeSolution::EyeInHand {
                gripper_se3_rig: Iso3::identity(),
                base_se3_target: Iso3::translation(0.0, 0.0, 1.0),
            },
            robot_refinement: None,
            mean_reproj_error: 0.5,
            per_cam_reproj_errors: vec![0.5; num_cameras],
            diagnostics: Default::default(),
            uncertainty: Default::default(),
        }
    }

    fn rig_laserline_spec(dir: &Path, num_views: usize) -> DatasetSpec {
        let poses: String = (0..num_views)
            .map(|i| identity_pose_line((i + 1) as f64 / 10.0))
            .collect();
        std::fs::write(dir.join("poses.txt"), poses).unwrap();
        std::fs::write(
            dir.join("rig_handeye_export.json"),
            serde_json::to_string(&upstream_export(2)).unwrap(),
        )
        .unwrap();
        let cam = |idx: usize| {
            let mut value = CameraSource::new(
                format!("cam{idx}"),
                ImagePattern::List {
                    paths: (0..num_views)
                        .map(|v| PathBuf::from(format!("cam{idx}/target_{v}.png")))
                        .collect(),
                },
            );
            value.laser_images = Some(ImagePattern::List {
                paths: (0..num_views)
                    .map(|v| PathBuf::from(format!("cam{idx}/laser_{v}.png")))
                    .collect(),
            });
            value
        };
        let mut spec = DatasetSpec::new(
            vec![cam(0), cam(1)],
            TargetSpec::chessboard(9, 6, 0.025),
            ProblemKind::RigLaserlineDevice,
        );
        spec.robot_poses = Some(RobotPoseSource::new(
            PathBuf::from("poses.txt"),
            RobotPoseFormat::Rowmajor4x4,
        ));
        spec.upstream_calibration = Some(PathBuf::from("rig_handeye_export.json"));
        spec.pose_pairing = Some(PosePairing::ByIndex);
        spec.pose_convention = Some(PoseConvention::new(
            TransformConvention::TBaseTcp,
            RotationFormat::Matrix4x4RowMajor,
            TranslationUnits::M,
        ));
        spec
    }

    fn seed_rig_views(dir: &Path, cache: &FsDetectionCache, num_views: usize) {
        for cam in 0..2 {
            for view in 0..num_views {
                seed_target(
                    dir,
                    cache,
                    &format!("cam{cam}/target_{view}.png"),
                    board_features(),
                );
                seed_laser(
                    dir,
                    cache,
                    &format!("cam{cam}/laser_{view}.png"),
                    laser_features(30),
                );
            }
        }
    }

    #[test]
    fn an_upstream_export_of_another_format_version_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let mut export = serde_json::to_value(upstream_export(2)).unwrap();
        let path = dir.path().join("upstream.json");
        std::fs::write(&path, serde_json::to_string(&export).unwrap()).unwrap();
        load_upstream_export(&path).unwrap();

        export["schema_version"] = 2.into();
        std::fs::write(&path, serde_json::to_string(&export).unwrap()).unwrap();
        let err = load_upstream_export(&path).unwrap_err();
        let RunError::UpstreamCalibration { message, .. } = &err else {
            panic!("expected UpstreamCalibration, got {err:?}");
        };
        assert!(message.contains("schema_version 2"), "{message}");
        assert!(message.contains("reads 1"), "{message}");
    }

    #[test]
    fn rig_laserline_builds_input_with_upstream_chain() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        seed_rig_views(tmp.path(), &cache, 3);
        let spec = rig_laserline_spec(tmp.path(), 3);
        let fake = FakeLaser::new(30);
        let result = build_rig_laserline_device_input(
            &spec,
            tmp.path(),
            Detection::Cached(&cache),
            &fake,
            &RunControl::default(),
        )
        .unwrap();
        assert_eq!((result.usable_views, result.total_views), (3, 3));
        // Laser image paths are captured per (view, camera), aligned
        // with view_paths, for the export's image_manifest.
        assert_eq!(result.laser_paths.len(), 3);
        assert!(
            result.laser_paths[2][1]
                .as_ref()
                .unwrap()
                .ends_with("cam1/laser_2.png")
        );
        let input = &result.input;
        assert_eq!(input.dataset.num_cameras, 2);
        assert_eq!(input.dataset.num_views(), 3);
        assert_eq!(
            input.dataset.views[0].laser_pixels[1]
                .as_ref()
                .unwrap()
                .len(),
            30
        );
        assert_eq!(input.upstream.cameras.len(), 2);
        assert!(
            matches!(
                input.upstream.cameras[0].sensor,
                vision_calibration_core::SensorParams::Identity
            ),
            "an untilted upstream camera keeps its identity sensor"
        );
        // EyeInHand chain with identity T_G_R:
        //   T_R_T_i = T_B_G_i^-1 * T_B_T = (-tx_i, 0, 1).
        let t = input.upstream.rig_se3_target[1].translation.vector;
        assert!(
            (t - nalgebra::Vector3::new(-0.2, 0.0, 1.0)).norm() < 1e-12,
            "got {t:?}"
        );
    }

    /// A cache that passes every call through and keeps each image file a
    /// read names ([`DetectionCache::note_image`]), as a project's does.
    struct Noting<'a> {
        inner: &'a FsDetectionCache,
        noted: std::sync::Mutex<std::collections::BTreeSet<PathBuf>>,
    }

    impl DetectionCache for Noting<'_> {
        fn get(
            &self,
            key: &CacheKey,
        ) -> Result<Option<CachedFeatures>, vision_calibration_detect::CacheError> {
            self.inner.get(key)
        }

        fn put(
            &self,
            key: &CacheKey,
            value: &CachedFeatures,
        ) -> Result<(), vision_calibration_detect::CacheError> {
            self.inner.put(key, value)
        }

        fn note_image(&self, image: &Path, _key: &CacheKey) {
            self.noted
                .lock()
                .unwrap()
                .insert(vision_calibration_core::lexical_absolute(image));
        }
    }

    /// Every frame of a rig laserline run's image manifest, target and
    /// laser, is an image the run's detection read through the cache, so a
    /// project records the content of each (and compares such runs by it).
    #[test]
    fn a_rig_laserline_run_notes_every_frame_of_its_manifest() {
        use super::super::calibrate::RunFrames;

        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        seed_rig_views(tmp.path(), &cache, 3);
        let spec = rig_laserline_spec(tmp.path(), 3);
        let noting = Noting {
            inner: &cache,
            noted: Default::default(),
        };
        let result = build_rig_laserline_device_input(
            &spec,
            tmp.path(),
            Detection::Cached(&noting),
            &FakeLaser::new(30),
            &RunControl::default(),
        )
        .unwrap();
        let manifest = result.image_manifest(&spec, tmp.path());
        assert_eq!(manifest.target_frames().count(), 6);
        assert_eq!(manifest.laser_frames().count(), 6);
        let noted = noting.noted.lock().unwrap();
        let root = tmp.path().join(&manifest.root);
        for frame in &manifest.frames {
            let path = vision_calibration_core::lexical_absolute(&root.join(&frame.path));
            assert!(noted.contains(&path), "{} was not noted", path.display());
        }
    }

    #[test]
    fn rig_handeye_laserline_builds_joint_input_without_upstream() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        seed_rig_views(tmp.path(), &cache, 3);
        let mut spec = rig_laserline_spec(tmp.path(), 3);
        spec.topology = ProblemKind::RigHandeyeLaserline;
        spec.upstream_calibration = None;
        let fake = FakeLaser::new(30);

        let result = build_rig_handeye_laserline_input(
            &spec,
            tmp.path(),
            Detection::Cached(&cache),
            &fake,
            &RunControl::default(),
        )
        .unwrap();
        assert_eq!((result.usable_views, result.total_views), (3, 3));
        assert_eq!(result.input.num_cameras, 2);
        assert_eq!(result.input.num_views(), 3);
        assert_eq!(result.view_paths.len(), 3);
        assert_eq!(result.laser_paths.len(), 3);
        assert_eq!(
            result.input.views[2].obs.laser_pixels[1]
                .as_ref()
                .unwrap()
                .len(),
            30
        );
        assert!(
            (result.input.views[1]
                .meta
                .base_se3_gripper
                .translation
                .vector
                .x
                - 0.2)
                .abs()
                < 1e-12
        );
    }

    /// A rig laser dataset of four views whose view 1 no camera detects, so
    /// the run keeps views 0, 2 and 3; and an upstream export with one board
    /// pose per upstream view, `x = 10 + its image index`.
    fn rig_laserline_dropping_view_1(dir: &Path) -> (FsDetectionCache, DatasetSpec) {
        let cache = FsDetectionCache::new(dir.join("cache"));
        seed_rig_views(dir, &cache, 4);
        for cam in 0..2 {
            seed_target(dir, &cache, &format!("cam{cam}/target_1.png"), vec![]);
        }
        (cache, rig_laserline_spec(dir, 4))
    }

    /// An upstream export that kept the views of `images` (indices into the
    /// dataset's images), recorded in its image manifest at `root` relative
    /// to the export's folder.
    fn upstream_with_views(root: &str, images: &[usize]) -> RigHandeyeExport {
        let mut upstream = upstream_export(2);
        upstream.rig_se3_target = images
            .iter()
            .map(|&i| Iso3::translation(10.0 + i as f64, 0.0, 1.0))
            .collect();
        let frames = images
            .iter()
            .enumerate()
            .flat_map(|(view, &i)| {
                (0..2).map(move |cam| {
                    vision_calibration_core::FrameRef::new(
                        view,
                        cam,
                        format!("cam{cam}/target_{i}.png"),
                    )
                })
            })
            .collect();
        upstream.diagnostics.image_manifest =
            Some(vision_calibration_core::ImageManifest::new(root, frames));
        upstream
    }

    fn write_upstream(dir: &Path, upstream: &RigHandeyeExport) {
        std::fs::write(
            dir.join("rig_handeye_export.json"),
            serde_json::to_string(upstream).unwrap(),
        )
        .unwrap();
    }

    fn rig_laserline_run(
        dir: &Path,
        cache: &FsDetectionCache,
        spec: &DatasetSpec,
    ) -> Result<RigLaserlineRunResult, RunError> {
        build_rig_laserline_device_input(
            spec,
            dir,
            Detection::Cached(cache),
            &FakeLaser::new(30),
            &RunControl::default(),
        )
    }

    fn board_x(result: &RigLaserlineRunResult) -> Vec<f64> {
        result
            .input
            .upstream
            .rig_se3_target
            .iter()
            .map(|pose| pose.translation.vector.x)
            .collect()
    }

    /// A rig view whose every image is excluded is dropped with its robot
    /// pose and its laser frames.
    #[test]
    fn an_excluded_rig_view_leaves_its_robot_pose_and_laser_frames_out() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        seed_rig_views(tmp.path(), &cache, 3);
        let mut spec = rig_laserline_spec(tmp.path(), 3);
        spec.topology = ProblemKind::RigHandeyeLaserline;
        spec.upstream_calibration = None;
        let mut options = ConvertOptions::default();
        for cam in 0..2 {
            options.exclude_slot(cam, &tmp.path().join(format!("cam{cam}/target_1.png")));
            std::fs::write(tmp.path().join(format!("cam{cam}/laser_1.png")), b"none").unwrap();
        }
        let fake = FakeLaser::new(30);
        let result = build_rig_handeye_laserline_input_with(
            &spec,
            tmp.path(),
            Detection::Cached(&cache),
            &fake,
            &RunControl::default(),
            &options,
        )
        .unwrap();
        assert_eq!((result.usable_views, result.total_views), (2, 3));
        assert_eq!(fake.calls.load(Ordering::SeqCst), 0);
        let xs: Vec<f64> = result
            .input
            .views
            .iter()
            .map(|v| v.meta.base_se3_gripper.translation.vector.x)
            .collect();
        assert_eq!(xs, [0.1, 0.3]);
        assert!(
            result.laser_paths[1][0]
                .as_ref()
                .unwrap()
                .ends_with("cam0/laser_2.png")
        );
    }

    /// Both runs kept the same views: the image manifest picks the views
    /// the positional rule picked, bit for bit.
    #[test]
    fn rig_laserline_matches_upstream_views_by_their_images() {
        let tmp = tempfile::tempdir().unwrap();
        let (cache, spec) = rig_laserline_dropping_view_1(tmp.path());

        let recorded = upstream_with_views(".", &[0, 2, 3]);
        let mut unrecorded = recorded.clone();
        unrecorded.diagnostics.image_manifest = None;

        write_upstream(tmp.path(), &unrecorded);
        let by_position = rig_laserline_run(tmp.path(), &cache, &spec).unwrap();
        write_upstream(tmp.path(), &recorded);
        let by_image = rig_laserline_run(tmp.path(), &cache, &spec).unwrap();

        assert_eq!(board_x(&by_image), [10.0, 12.0, 13.0]);
        let bits = |r: &RigLaserlineRunResult| {
            serde_json::to_string(&r.input.upstream.rig_se3_target).unwrap()
        };
        assert_eq!(bits(&by_image), bits(&by_position));
    }

    /// An export written to another folder with its root left at the
    /// dataset folder still matches, by the images' paths within it.
    #[test]
    fn rig_laserline_matches_upstream_views_by_their_path_in_the_dataset() {
        let tmp = tempfile::tempdir().unwrap();
        let (cache, mut spec) = rig_laserline_dropping_view_1(tmp.path());
        std::fs::create_dir_all(tmp.path().join("upstream")).unwrap();
        std::fs::write(
            tmp.path().join("upstream/export.json"),
            serde_json::to_string(&upstream_with_views(".", &[3, 0, 2])).unwrap(),
        )
        .unwrap();
        spec.upstream_calibration = Some(PathBuf::from("upstream/export.json"));
        let result = rig_laserline_run(tmp.path(), &cache, &spec).unwrap();
        assert_eq!(board_x(&result), [10.0, 12.0, 13.0]);
    }

    /// The upstream run dropped view 2 where this one dropped view 1: both
    /// kept three views, so the positional rule would pair view 2 with the
    /// upstream's view 1 silently. The image manifest refuses it.
    #[test]
    fn rig_laserline_refuses_a_view_the_upstream_dropped() {
        let tmp = tempfile::tempdir().unwrap();
        let (cache, spec) = rig_laserline_dropping_view_1(tmp.path());
        write_upstream(tmp.path(), &upstream_with_views(".", &[0, 1, 3]));
        let err = rig_laserline_run(tmp.path(), &cache, &spec).unwrap_err();
        assert!(
            matches!(
                &err,
                RunError::UpstreamViewMatch { token, matches: 0, .. } if token == "2"
            ),
            "{err:?}"
        );
    }

    /// A view whose images two upstream views show is ambiguous.
    #[test]
    fn rig_laserline_refuses_an_ambiguous_upstream_view() {
        let tmp = tempfile::tempdir().unwrap();
        let (cache, spec) = rig_laserline_dropping_view_1(tmp.path());
        let mut upstream = upstream_with_views(".", &[0, 2, 3]);
        let manifest = upstream.diagnostics.image_manifest.as_mut().unwrap();
        // Upstream view 1 also claims camera 1's image of view 0.
        manifest.frames.push(vision_calibration_core::FrameRef::new(
            1,
            1,
            "cam1/target_0.png",
        ));
        write_upstream(tmp.path(), &upstream);
        let err = rig_laserline_run(tmp.path(), &cache, &spec).unwrap_err();
        assert!(
            matches!(
                &err,
                RunError::UpstreamViewMatch { token, matches: 2, .. } if token == "0"
            ),
            "{err:?}"
        );
    }

    #[test]
    fn rig_laserline_preserves_upstream_rig_target_poses() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        seed_rig_views(tmp.path(), &cache, 3);
        let spec = rig_laserline_spec(tmp.path(), 3);
        let mut upstream = upstream_export(2);
        upstream.rig_se3_target = vec![
            Iso3::translation(1.0, 0.0, 1.0),
            Iso3::translation(2.0, 0.0, 1.0),
            Iso3::translation(3.0, 0.0, 1.0),
        ];
        upstream.handeye = HandeyeSolution::EyeInHand {
            gripper_se3_rig: Iso3::identity(),
            base_se3_target: Iso3::translation(0.0, 0.0, 9.0),
        };
        std::fs::write(
            tmp.path().join("rig_handeye_export.json"),
            serde_json::to_string(&upstream).unwrap(),
        )
        .unwrap();

        let fake = FakeLaser::new(30);
        let result = build_rig_laserline_device_input(
            &spec,
            tmp.path(),
            Detection::Cached(&cache),
            &fake,
            &RunControl::default(),
        )
        .unwrap();
        let t = result.input.upstream.rig_se3_target[1].translation.vector;
        assert!(
            (t - nalgebra::Vector3::new(2.0, 0.0, 1.0)).norm() < 1e-12,
            "got {t:?}"
        );
    }

    /// The upstream export's covariance reaches the rig laser input, in the
    /// upstream's terms: its cameras' labels as they are, its hand-eye and
    /// target poses as each kept view's `rig_se3_target`.
    #[test]
    fn rig_laserline_carries_the_upstream_covariance() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        seed_rig_views(tmp.path(), &cache, 3);
        let spec = rig_laserline_spec(tmp.path(), 3);
        let mut upstream = upstream_export(2);
        let mut labels = vec!["cam0.fx".to_string()];
        for pose in ["gripper_se3_rig", "base_se3_target"] {
            labels.extend(["rx", "ry", "rz", "tx", "ty", "tz"].map(|c| format!("{pose}.{c}")));
        }
        let n = labels.len();
        upstream.uncertainty = vision_calibration_core::Uncertainty::Available(
            vision_calibration_core::ParameterCovariance::new(
                labels,
                (0..n)
                    .map(|i| (0..n).map(|j| if i == j { 1e-6 } else { 0.0 }).collect())
                    .collect(),
                Vec::new(),
                100,
                87,
            ),
        );
        std::fs::write(
            tmp.path().join("rig_handeye_export.json"),
            serde_json::to_string(&upstream).unwrap(),
        )
        .unwrap();

        let result = build_rig_laserline_device_input(
            &spec,
            tmp.path(),
            Detection::Cached(&cache),
            &FakeLaser::new(30),
            &RunControl::default(),
        )
        .unwrap();
        let cov = result
            .input
            .upstream
            .uncertainty
            .covariance()
            .expect("the upstream's covariance");
        assert_eq!(cov.labels[0], "cam0.fx");
        assert_eq!(cov.labels.len(), 1 + 6 * result.input.dataset.num_views());
        assert!(cov.std_dev("view2.rig_se3_target.tz").unwrap() > 0.0);
        assert!(cov.index("gripper_se3_rig.rx").is_none());
    }

    #[test]
    fn rig_laserline_fallback_chain_applies_upstream_robot_deltas() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        seed_rig_views(tmp.path(), &cache, 3);
        let spec = rig_laserline_spec(tmp.path(), 3);
        let mut upstream = upstream_export(2);
        upstream.robot_refinement = Some(vision_calibration_optim::RobotRefinement::new(
            vec![[0.0; 6], [0.0, 0.0, 0.0, 0.05, 0.0, 0.0], [0.0; 6]],
            vision_calibration_optim::RobotPoseSigma::new(1.0e-3, 1.0e-3),
        ));
        std::fs::write(
            tmp.path().join("rig_handeye_export.json"),
            serde_json::to_string(&upstream).unwrap(),
        )
        .unwrap();

        let fake = FakeLaser::new(30);
        let result = build_rig_laserline_device_input(
            &spec,
            tmp.path(),
            Detection::Cached(&cache),
            &fake,
            &RunControl::default(),
        )
        .unwrap();
        let t = result.input.upstream.rig_se3_target[1].translation.vector;
        assert!(
            (t - nalgebra::Vector3::new(-0.25, 0.0, 1.0)).norm() < 1e-12,
            "got {t:?}"
        );
    }

    #[test]
    fn rig_laserline_camera_without_laser_is_error() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        seed_rig_views(tmp.path(), &cache, 3);
        // Re-seed cam1's laser views below the min_points bar.
        for view in 0..3 {
            seed_laser(
                tmp.path(),
                &cache,
                &format!("cam1/laser_{view}.png"),
                laser_features(3),
            );
        }
        let spec = rig_laserline_spec(tmp.path(), 3);
        let fake = FakeLaser::new(30);
        let err = build_rig_laserline_device_input(
            &spec,
            tmp.path(),
            Detection::Cached(&cache),
            &fake,
            &RunControl::default(),
        )
        .unwrap_err();
        match err {
            RunError::InsufficientLaserViews { camera_id, .. } => assert_eq!(camera_id, "cam1"),
            other => panic!("expected InsufficientLaserViews, got {other:?}"),
        }
    }

    #[test]
    fn rig_laserline_upstream_camera_count_mismatch_is_error() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        seed_rig_views(tmp.path(), &cache, 3);
        let spec = rig_laserline_spec(tmp.path(), 3);
        // Overwrite the upstream export with a 3-camera one.
        std::fs::write(
            tmp.path().join("rig_handeye_export.json"),
            serde_json::to_string(&upstream_export(3)).unwrap(),
        )
        .unwrap();
        let fake = FakeLaser::new(30);
        let err = build_rig_laserline_device_input(
            &spec,
            tmp.path(),
            Detection::Cached(&cache),
            &fake,
            &RunControl::default(),
        )
        .unwrap_err();
        match err {
            RunError::UpstreamCalibration { message, .. } => {
                assert!(message.contains("3 cameras"), "got: {message}");
            }
            other => panic!("expected UpstreamCalibration, got {other:?}"),
        }
    }

    #[test]
    fn rig_laserline_missing_upstream_file_is_actionable() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        seed_rig_views(tmp.path(), &cache, 3);
        let mut spec = rig_laserline_spec(tmp.path(), 3);
        spec.upstream_calibration = Some(PathBuf::from("does_not_exist.json"));
        let fake = FakeLaser::new(30);
        let err = build_rig_laserline_device_input(
            &spec,
            tmp.path(),
            Detection::Cached(&cache),
            &fake,
            &RunControl::default(),
        )
        .unwrap_err();
        match err {
            RunError::UpstreamCalibration { message, .. } => {
                assert!(message.contains("read failed"), "got: {message}");
            }
            other => panic!("expected UpstreamCalibration, got {other:?}"),
        }
    }

    #[test]
    fn wrong_topology_rejected_by_both_converters() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        let fake = FakeLaser::new(30);
        let mut spec = laserline_spec(&["a.png"], &["b.png"]);
        spec.topology = ProblemKind::PlanarIntrinsics;
        assert!(matches!(
            build_laserline_device_input(
                &spec,
                tmp.path(),
                Detection::Cached(&cache),
                &fake,
                &RunControl::default()
            )
            .unwrap_err(),
            RunError::UnsupportedTopology { .. }
        ));
        assert!(matches!(
            build_rig_laserline_device_input(
                &spec,
                tmp.path(),
                Detection::Cached(&cache),
                &fake,
                &RunControl::default()
            )
            .unwrap_err(),
            RunError::UnsupportedTopology { .. }
        ));
        assert!(matches!(
            build_rig_handeye_laserline_input(
                &spec,
                tmp.path(),
                Detection::Cached(&cache),
                &fake,
                &RunControl::default()
            )
            .unwrap_err(),
            RunError::UnsupportedTopology { .. }
        ));
    }
}
