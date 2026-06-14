//! `DatasetSpec → LaserlineDeviceInput / RigLaserlineDeviceInput`
//! converters for the laser topologies (ADR 0021).
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

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use nalgebra::{Translation3, UnitQuaternion, Vector3};
use serde_json::Value;

use vision_calibration_core::{Iso3, Pt2, View};
use vision_calibration_dataset::{
    DatasetSpec, ImagePattern, LaserExtractionSpec, PosePairing, Topology, validate,
};
use vision_calibration_detect::{CacheKey, CachedFeatures, DetectionCache, Feature};
use vision_calibration_optim::{
    HandEyeMode, LaserlineMeta, RigHandeyeLaserlineView, RigLaserlineDataset, RigLaserlineView,
    RobotPoseMeta,
};

use crate::laserline_device::LaserlineDeviceInput;
use crate::rig_handeye::RigHandeyeExport;
use crate::rig_handeye_laserline::RigHandeyeLaserlineInput;
use crate::rig_laserline_device::RigLaserlineDeviceInput;

use super::pairing::pair_views;
use super::poses::{load_robot_poses, match_poses_to_tokens};
use super::{
    RunError, augment_config_with_roi, detect_features, expand_camera_images, expand_image_pattern,
    features_to_obs, pattern_repr, pick_detector, rig::build_rig_core, target_to_detector_config,
};

/// Injected laser-line extractor (ADR 0021).
///
/// Unlike the sealed [`Detector`](vision_calibration_detect::Detector)
/// trait this one is **open**: the reference implementation lives in
/// the (unpublished) Tauri app and wraps `vision-metrology`, which is
/// not on crates.io. Implementations only extract; caching, ROI
/// cropping, and coordinate lifting are handled by the runner.
pub trait LaserPixelExtractor: Send + Sync {
    /// Stable implementation id. Part of the cache key
    /// (`laser:<name>`), so switching implementations re-extracts
    /// instead of serving another extractor's pixels.
    fn name(&self) -> &str;

    /// Extract subpixel laser-line points `[x, y]` from `image`
    /// (already ROI-cropped by the runner; coordinates are in the
    /// cropped frame).
    fn extract(
        &self,
        image: &image::DynamicImage,
        spec: &LaserExtractionSpec,
    ) -> anyhow::Result<Vec<[f64; 2]>>;
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
    /// `view_paths`. Not yet rendered anywhere (B-laser).
    pub laser_paths: Vec<PathBuf>,
    /// Number of views with ≥4 target features and ≥`min_points`
    /// laser pixels.
    pub usable_views: usize,
    /// Total number of paired views attempted.
    pub total_views: usize,
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
    /// Number of kept views (≥1 camera with ≥4 target features).
    pub usable_views: usize,
    /// Total number of paired views attempted.
    pub total_views: usize,
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
    /// Number of kept views (≥1 camera with ≥4 target features).
    pub usable_views: usize,
    /// Total number of paired views attempted.
    pub total_views: usize,
}

/// Convert a single-camera laserline manifest + cache into a
/// [`LaserlineDeviceInput`] for `LaserlineDevice`.
pub fn build_laserline_device_input(
    spec: &DatasetSpec,
    base_dir: &Path,
    cache: &dyn DetectionCache,
    laser_extractor: &dyn LaserPixelExtractor,
    force_redetect: bool,
) -> Result<LaserlineRunResult, RunError> {
    // Same gate ordering as the other converters: topology + target +
    // validation + pairing config before any filesystem access.
    if spec.topology != Topology::LaserlineDevice {
        return Err(RunError::UnsupportedTopology {
            topology: spec.topology,
        });
    }
    let (detector_name, detector_config) = target_to_detector_config(spec)?;
    let detector = pick_detector(detector_name)?;
    validate(spec)?;

    let Some(pairing) = &spec.pose_pairing else {
        return Err(RunError::AskUser {
            field: "pose_pairing".into(),
            prompt: "Laser topologies need to know how laser frames align with target \
                     frames. Pick by_index (the i-th laser image pairs with the i-th \
                     target image in natural-sort order) or shared_filename_token (a \
                     regex extracts a shared view token from both filename sets)."
                .into(),
            suggestions: vec!["by_index".into(), "shared_filename_token".into()],
        });
    };

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
    let key_config = augment_config_with_roi(&detector_config, roi);
    let laser_key_config = laser_key_config(&laser_spec, roi);

    let mut views: Vec<View<LaserlineMeta>> = Vec::new();
    let mut view_paths: Vec<PathBuf> = Vec::new();
    let mut laser_paths: Vec<PathBuf> = Vec::new();
    for (paths, token) in paired.paths.iter().zip(&paired.tokens) {
        let path = paths[0]
            .as_ref()
            .expect("single-camera pairing never produces gap slots");
        let (features, _cache_hit) = detect_features(
            detector.as_ref(),
            detector_name,
            &detector_config,
            &key_config,
            roi,
            path,
            cache,
            force_redetect,
        )?;
        if features.len() < 4 {
            continue; // too few corners for a homography — drop the view
        }
        // The single-camera problem needs laser pixels in *every* view
        // (LaserlineMeta validates non-empty), so a view without a
        // usable laser frame is dropped rather than gapped.
        let Some(laser_path) = laser_by_token.get(token) else {
            continue;
        };
        let pixels = extract_laser_pixels(
            laser_extractor,
            &laser_spec,
            &laser_key_config,
            roi,
            laser_path,
            cache,
            force_redetect,
        )?;
        if pixels.len() < laser_spec.min_points as usize {
            continue;
        }
        views.push(View {
            obs: features_to_obs(&features),
            meta: LaserlineMeta {
                laser_pixels: pixels,
                laser_weights: vec![],
            },
        });
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

    Ok(LaserlineRunResult {
        input: views,
        view_paths,
        laser_paths,
        usable_views,
        total_views,
    })
}

/// Convert a rig laserline manifest + cache + frozen upstream rig
/// hand-eye export into a [`RigLaserlineDeviceInput`] for
/// `RigLaserlineDevice`.
pub fn build_rig_laserline_device_input(
    spec: &DatasetSpec,
    base_dir: &Path,
    cache: &dyn DetectionCache,
    laser_extractor: &dyn LaserPixelExtractor,
    force_redetect: bool,
) -> Result<RigLaserlineRunResult, RunError> {
    if spec.topology != Topology::RigLaserlineDevice {
        return Err(RunError::UnsupportedTopology {
            topology: spec.topology,
        });
    }
    let core = build_rig_core(spec, base_dir, cache, force_redetect)?;
    let num_cameras = spec.cameras.len();

    // `validate` (inside build_rig_core) guarantees pose_pairing,
    // robot_poses, pose_convention, upstream_calibration, and
    // laser_images on every camera for this topology.
    let pairing = spec
        .pose_pairing
        .as_ref()
        .expect("build_rig_core checked pose_pairing");

    // Laser frames pair against *all* view tokens (pre-drop) — a laser
    // image for a view that was later dropped is not an error.
    let laser_spec = spec.laser.unwrap_or_default();
    let mut laser_by_token: Vec<HashMap<String, PathBuf>> = Vec::with_capacity(num_cameras);
    for camera in &spec.cameras {
        let laser_images = camera
            .laser_images
            .as_ref()
            .expect("validate guarantees laser_images for RigLaserlineDevice");
        laser_by_token.push(laser_paths_by_token(
            laser_images,
            &camera.id,
            pairing,
            &core.all_tokens,
            base_dir,
        )?);
    }

    // Per-camera laser extraction over the kept views.
    let laser_key_configs: Vec<Value> = spec
        .cameras
        .iter()
        .map(|c| laser_key_config(&laser_spec, c.roi_xywh))
        .collect();
    let mut laser_per_view: Vec<Vec<Option<Vec<Pt2>>>> = Vec::with_capacity(core.views.len());
    let mut laser_paths: Vec<Vec<Option<PathBuf>>> = Vec::with_capacity(core.views.len());
    let mut per_camera_usable = vec![0usize; num_cameras];
    for (_obs, token) in &core.views {
        let mut slots: Vec<Option<Vec<Pt2>>> = Vec::with_capacity(num_cameras);
        let mut path_slots: Vec<Option<PathBuf>> = Vec::with_capacity(num_cameras);
        for cam_idx in 0..num_cameras {
            let Some(laser_path) = laser_by_token[cam_idx].get(token) else {
                slots.push(None);
                path_slots.push(None);
                continue;
            };
            let pixels = extract_laser_pixels(
                laser_extractor,
                &laser_spec,
                &laser_key_configs[cam_idx],
                spec.cameras[cam_idx].roi_xywh,
                laser_path,
                cache,
                force_redetect,
            )?;
            if pixels.len() < laser_spec.min_points as usize {
                slots.push(None);
                path_slots.push(None);
                continue;
            }
            per_camera_usable[cam_idx] += 1;
            slots.push(Some(pixels));
            path_slots.push(Some(laser_path.clone()));
        }
        laser_per_view.push(slots);
        laser_paths.push(path_slots);
    }
    for (cam_idx, usable) in per_camera_usable.iter().enumerate() {
        if *usable == 0 {
            // A camera with zero laser observations leaves its plane
            // unconstrained — fail fast instead of solving garbage.
            return Err(RunError::InsufficientLaserViews {
                camera_id: spec.cameras[cam_idx].id.clone(),
                usable: 0,
                total: core.views.len(),
            });
        }
    }

    // Robot poses → per-kept-view T_B_G, matched by pairing token.
    let source = spec
        .robot_poses
        .as_ref()
        .expect("validate guarantees robot_poses for RigLaserlineDevice");
    let convention = spec
        .pose_convention
        .as_ref()
        .expect("validate guarantees pose_convention for RigLaserlineDevice");
    let poses = load_robot_poses(source, convention, base_dir)?;
    let kept_tokens: Vec<String> = core.views.iter().map(|(_, t)| t.clone()).collect();
    let matched = match_poses_to_tokens(&kept_tokens, &poses, spec, core.total_views)?;

    // Frozen upstream rig hand-eye export → per-view rig_se3_target.
    // Prefer the upstream export's optimized target poses; older exports
    // fall back to the hand-eye chain.
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
    if export.cameras.len() != num_cameras {
        return Err(RunError::UpstreamCalibration {
            path: upstream_path,
            message: format!(
                "export has {} cameras but the manifest declares {num_cameras}",
                export.cameras.len()
            ),
        });
    }
    let rig_se3_target: Vec<Iso3> = matched
        .iter()
        .zip(kept_tokens.iter())
        .enumerate()
        .map(|(view_idx, (base_se3_gripper, token))| {
            rig_se3_target_from_upstream(
                &export,
                base_se3_gripper,
                token,
                view_idx,
                core.views.len(),
                core.total_views,
                &upstream_path,
            )
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
        .zip(laser_per_view)
        .map(|((obs, _token), laser_pixels)| RigLaserlineView {
            cameras: obs.cameras,
            laser_pixels,
        })
        .collect();
    let usable_views = views.len();
    let dataset = RigLaserlineDataset::new(views, num_cameras).expect(
        "RigLaserlineDataset::new failed after the runner enforced non-empty views \
         and per-view camera counts; this indicates an optim-crate regression",
    );

    Ok(RigLaserlineRunResult {
        input: RigLaserlineDeviceInput {
            dataset,
            upstream,
            initial_planes_cam: None,
        },
        view_paths: core.view_paths,
        laser_paths,
        usable_views,
        total_views: core.total_views,
    })
}

/// Convert a manifest into a joint rig hand-eye laserline input.
pub fn build_rig_handeye_laserline_input(
    spec: &DatasetSpec,
    base_dir: &Path,
    cache: &dyn DetectionCache,
    laser_extractor: &dyn LaserPixelExtractor,
    force_redetect: bool,
) -> Result<RigHandeyeLaserlineRunResult, RunError> {
    if spec.topology != Topology::RigHandeyeLaserline {
        return Err(RunError::UnsupportedTopology {
            topology: spec.topology,
        });
    }
    let core = build_rig_core(spec, base_dir, cache, force_redetect)?;
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
            .expect("validate guarantees laser_images for RigHandeyeLaserline");
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
    let mut laser_per_view: Vec<Vec<Option<Vec<Pt2>>>> = Vec::with_capacity(core.views.len());
    let mut laser_paths: Vec<Vec<Option<PathBuf>>> = Vec::with_capacity(core.views.len());
    let mut per_camera_usable = vec![0usize; num_cameras];
    for (_obs, token) in &core.views {
        let mut slots: Vec<Option<Vec<Pt2>>> = Vec::with_capacity(num_cameras);
        let mut path_slots: Vec<Option<PathBuf>> = Vec::with_capacity(num_cameras);
        for cam_idx in 0..num_cameras {
            let Some(laser_path) = laser_by_token[cam_idx].get(token) else {
                slots.push(None);
                path_slots.push(None);
                continue;
            };
            let pixels = extract_laser_pixels(
                laser_extractor,
                &laser_spec,
                &laser_key_configs[cam_idx],
                spec.cameras[cam_idx].roi_xywh,
                laser_path,
                cache,
                force_redetect,
            )?;
            if pixels.len() < laser_spec.min_points as usize {
                slots.push(None);
                path_slots.push(None);
                continue;
            }
            per_camera_usable[cam_idx] += 1;
            slots.push(Some(pixels));
            path_slots.push(Some(laser_path.clone()));
        }
        laser_per_view.push(slots);
        laser_paths.push(path_slots);
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

    let source = spec
        .robot_poses
        .as_ref()
        .expect("validate guarantees robot_poses for RigHandeyeLaserline");
    let convention = spec
        .pose_convention
        .as_ref()
        .expect("validate guarantees pose_convention for RigHandeyeLaserline");
    let poses = load_robot_poses(source, convention, base_dir)?;
    let kept_tokens: Vec<String> = core.views.iter().map(|(_, t)| t.clone()).collect();
    let matched = match_poses_to_tokens(&kept_tokens, &poses, spec, core.total_views)?;

    let views: Vec<RigHandeyeLaserlineView> = core
        .views
        .into_iter()
        .zip(laser_per_view)
        .zip(matched)
        .map(
            |(((obs, _token), laser_pixels), base_se3_gripper)| RigHandeyeLaserlineView {
                obs: RigLaserlineView {
                    cameras: obs.cameras,
                    laser_pixels,
                },
                meta: RobotPoseMeta { base_se3_gripper },
            },
        )
        .collect();
    let usable_views = views.len();
    Ok(RigHandeyeLaserlineRunResult {
        input: RigHandeyeLaserlineInput { views, num_cameras },
        view_paths: core.view_paths,
        laser_paths,
        usable_views,
        total_views: core.total_views,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Cache-key config for laser extraction: the canonical
/// `LaserExtractionSpec` JSON plus the `_roi` splice (same scheme as
/// target detection).
fn laser_key_config(laser_spec: &LaserExtractionSpec, roi: Option<[u32; 4]>) -> Value {
    let config =
        serde_json::to_value(laser_spec).expect("LaserExtractionSpec serializes to a JSON object");
    augment_config_with_roi(&config, roi)
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
    cache: &dyn DetectionCache,
    force_redetect: bool,
) -> Result<Vec<Pt2>, RunError> {
    let bytes = std::fs::read(image_path)?;
    let detector_name = format!("laser:{}", extractor.name());
    let key = CacheKey::from_inputs(&bytes, &detector_name, key_config);

    let cached: Option<CachedFeatures> = if force_redetect {
        None
    } else {
        cache.get(&key)?
    };
    if let Some(entry) = cached {
        return Ok(entry
            .features
            .iter()
            .map(|f| Pt2::new(f.image_xy[0], f.image_xy[1]))
            .collect());
    }

    let img = image::load_from_memory(&bytes).map_err(|e| RunError::Decode {
        path: image_path.to_path_buf(),
        source: e,
    })?;
    let img_for_extract = if let Some([x, y, w, h]) = roi {
        img.crop_imm(x, y, w, h)
    } else {
        img
    };
    let points = extractor
        .extract(&img_for_extract, laser_spec)
        .map_err(|e| RunError::LaserExtraction {
            path: image_path.to_path_buf(),
            source: e,
        })?;
    cache.put(
        &key,
        &CachedFeatures {
            features: points
                .iter()
                .map(|p| Feature {
                    image_xy: *p,
                    world_xyz: [0.0, 0.0, 0.0],
                })
                .collect(),
        },
    )?;
    Ok(points.into_iter().map(|p| Pt2::new(p[0], p[1])).collect())
}

fn load_upstream_export(path: &Path) -> Result<RigHandeyeExport, RunError> {
    let text = std::fs::read_to_string(path).map_err(|e| RunError::UpstreamCalibration {
        path: path.to_path_buf(),
        message: format!("read failed: {e}"),
    })?;
    serde_json::from_str(&text).map_err(|e| RunError::UpstreamCalibration {
        path: path.to_path_buf(),
        message: format!("not a rig hand-eye export: {e}"),
    })
}

/// Per-view rig→target pose through the frozen hand-eye chain
/// (ADR 0021 §4, same math as the rtv3d example):
///
/// - `EyeInHand`:  `T_R_T = T_G_R⁻¹ · T_B_G⁻¹ · T_B_T`
/// - `EyeToHand`:  `T_R_T = T_R_B · T_B_G · T_G_T`
fn rig_se3_target_from_upstream(
    export: &RigHandeyeExport,
    base_se3_gripper: &Iso3,
    token: &str,
    kept_idx: usize,
    kept_views: usize,
    total_views: usize,
    path: &Path,
) -> Result<Iso3, RunError> {
    if !export.rig_se3_target.is_empty() {
        if export.rig_se3_target.len() == total_views {
            let idx = token.parse::<usize>().map_err(|_| RunError::UpstreamCalibration {
                path: path.to_path_buf(),
                message: format!(
                    "upstream rig_se3_target has {total_views} entries but view token {token:?} \
                     is not an index"
                ),
            })?;
            return export.rig_se3_target.get(idx).copied().ok_or_else(|| {
                RunError::UpstreamCalibration {
                    path: path.to_path_buf(),
                    message: format!(
                        "view token {token:?} resolved to index {idx}, outside upstream \
                         rig_se3_target length {}",
                        export.rig_se3_target.len()
                    ),
                }
            });
        }
        if export.rig_se3_target.len() == kept_views {
            return Ok(export.rig_se3_target[kept_idx]);
        }
        return Err(RunError::UpstreamCalibration {
            path: path.to_path_buf(),
            message: format!(
                "upstream rig_se3_target has {} entries, but this run has {kept_views} kept \
                 views and {total_views} total paired views",
                export.rig_se3_target.len()
            ),
        });
    }

    let base_se3_gripper = corrected_robot_pose(
        *base_se3_gripper,
        robot_delta_for_view(export, token, kept_idx, kept_views, total_views, path)?,
    );
    rig_se3_target_from_chain(export, &base_se3_gripper, path)
}

fn robot_delta_for_view(
    export: &RigHandeyeExport,
    token: &str,
    kept_idx: usize,
    kept_views: usize,
    total_views: usize,
    path: &Path,
) -> Result<Option<[f64; 6]>, RunError> {
    let Some(deltas) = export.robot_deltas.as_ref() else {
        return Ok(None);
    };
    if deltas.len() == total_views {
        let idx = token
            .parse::<usize>()
            .map_err(|_| RunError::UpstreamCalibration {
                path: path.to_path_buf(),
                message: format!(
                    "upstream robot_deltas has {total_views} entries but view token {token:?} \
                 is not an index"
                ),
            })?;
        return deltas
            .get(idx)
            .copied()
            .map(Some)
            .ok_or_else(|| RunError::UpstreamCalibration {
                path: path.to_path_buf(),
                message: format!(
                    "view token {token:?} resolved to index {idx}, outside upstream \
                     robot_deltas length {}",
                    deltas.len()
                ),
            });
    }
    if deltas.len() == kept_views {
        return Ok(Some(deltas[kept_idx]));
    }
    Err(RunError::UpstreamCalibration {
        path: path.to_path_buf(),
        message: format!(
            "upstream robot_deltas has {} entries, but this run has {kept_views} kept views \
             and {total_views} total paired views",
            deltas.len()
        ),
    })
}

fn corrected_robot_pose(robot_pose: Iso3, delta: Option<[f64; 6]>) -> Iso3 {
    let Some(delta) = delta else {
        return robot_pose;
    };
    let rot_vec = Vector3::new(delta[0], delta[1], delta[2]);
    let trans_vec = Vector3::new(delta[3], delta[4], delta[5]);
    let angle = rot_vec.norm();
    let delta_rot = if angle > 1e-12 {
        UnitQuaternion::from_axis_angle(&nalgebra::Unit::new_normalize(rot_vec), angle)
    } else {
        UnitQuaternion::identity()
    };
    let delta_iso = Iso3::from_parts(Translation3::from(trans_vec), delta_rot);
    delta_iso * robot_pose
}

fn rig_se3_target_from_chain(
    export: &RigHandeyeExport,
    base_se3_gripper: &Iso3,
    path: &Path,
) -> Result<Iso3, RunError> {
    let missing = |field: &str| RunError::UpstreamCalibration {
        path: path.to_path_buf(),
        message: format!("{:?} export is missing {field}", export.handeye_mode),
    };
    match export.handeye_mode {
        HandEyeMode::EyeInHand => {
            let gripper_se3_rig = export
                .gripper_se3_rig
                .ok_or_else(|| missing("gripper_se3_rig"))?;
            let base_se3_target = export
                .base_se3_target
                .ok_or_else(|| missing("base_se3_target"))?;
            Ok(gripper_se3_rig.inverse() * base_se3_gripper.inverse() * base_se3_target)
        }
        HandEyeMode::EyeToHand => {
            let rig_se3_base = export.rig_se3_base.ok_or_else(|| missing("rig_se3_base"))?;
            let gripper_se3_target = export
                .gripper_se3_target
                .ok_or_else(|| missing("gripper_se3_target"))?;
            Ok(rig_se3_base * base_se3_gripper * gripper_se3_target)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Write;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use vision_calibration_core::{
        BrownConrady5, FxFyCxCySkew, PerFeatureResiduals, make_pinhole_camera,
    };
    use vision_calibration_dataset::{
        CameraSource, PoseConvention, RobotPoseFormat, RobotPoseSource, RotationFormat, TargetSpec,
        TransformConvention, TranslationUnits,
    };
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
            _image: &image::DynamicImage,
            _spec: &LaserExtractionSpec,
        ) -> anyhow::Result<Vec<[f64; 2]>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok((0..self.n).map(|i| [i as f64, 100.0]).collect())
        }
    }

    fn grid_features(n: usize) -> Vec<Feature> {
        (0..n)
            .map(|i| Feature {
                image_xy: [100.0 + 10.0 * i as f64, 200.0 + 5.0 * i as f64],
                world_xyz: [0.025 * i as f64, 0.0, 0.0],
            })
            .collect()
    }

    fn detector_config() -> Value {
        json!({"rows": 9, "cols": 6, "square_size_m": 0.025})
    }

    fn laser_features(n: usize) -> Vec<Feature> {
        (0..n)
            .map(|i| Feature {
                image_xy: [i as f64, 50.0],
                world_xyz: [0.0, 0.0, 0.0],
            })
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
        let key = CacheKey::from_inputs(bytes, "chessboard", &detector_config());
        cache.put(&key, &CachedFeatures { features }).unwrap();
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
        cache.put(&key, &CachedFeatures { features }).unwrap();
        path
    }

    fn laserline_spec(image_paths: &[&str], laser_paths: &[&str]) -> DatasetSpec {
        DatasetSpec {
            version: 1,
            cameras: vec![CameraSource {
                id: "cam0".into(),
                images: ImagePattern::List {
                    paths: image_paths.iter().map(PathBuf::from).collect(),
                },
                roi_xywh: None,
                laser_images: Some(ImagePattern::List {
                    paths: laser_paths.iter().map(PathBuf::from).collect(),
                }),
            }],
            target: TargetSpec::Chessboard {
                rows: 9,
                cols: 6,
                square_size_m: 0.025,
            },
            detector: None,
            robot_poses: None,
            laser: None,
            upstream_calibration: None,
            topology: Topology::LaserlineDevice,
            pose_pairing: Some(PosePairing::ByIndex),
            pose_convention: None,
            unresolved: vec![],
            description: None,
        }
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
                grid_features(6),
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
        let result = build_laserline_device_input(&spec, tmp.path(), &cache, &fake, false).unwrap();
        assert_eq!((result.usable_views, result.total_views), (3, 3));
        assert_eq!(result.input.len(), 3);
        assert_eq!(result.input[0].meta.laser_pixels.len(), 30);
        assert!(result.input[0].meta.laser_weights.is_empty());
        assert!(result.view_paths[2].ends_with("target_2.png"));
        assert!(result.laser_paths[2].ends_with("laser_2.png"));
        // Everything was served from the cache.
        assert_eq!(fake.calls.load(Ordering::SeqCst), 0);
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
                grid_features(6),
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
        let result = build_laserline_device_input(&spec, tmp.path(), &cache, &fake, false).unwrap();
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
                grid_features(6),
            );
        }
        seed_laser(tmp.path(), &cache, "laser_0.png", laser_features(30));
        let spec = laserline_spec(
            &["target_0.png", "target_1.png", "target_2.png"],
            &["laser_0.png"],
        );
        let fake = FakeLaser::new(30);
        let err =
            build_laserline_device_input(&spec, tmp.path(), &cache, &fake, false).unwrap_err();
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
                grid_features(6),
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
        let result = build_laserline_device_input(&spec, tmp.path(), &cache, &fake, false).unwrap();
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
        let err =
            build_laserline_device_input(&spec, tmp.path(), &cache, &fake, false).unwrap_err();
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
            &cache,
            false,
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
            &cache,
            false,
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
        let k = FxFyCxCySkew {
            fx: 800.0,
            fy: 800.0,
            cx: 320.0,
            cy: 240.0,
            skew: 0.0,
        };
        RigHandeyeExport {
            cameras: (0..num_cameras)
                .map(|_| make_pinhole_camera(k, BrownConrady5::default()))
                .collect(),
            sensors: None,
            cam_se3_rig: vec![Iso3::identity(); num_cameras],
            rig_se3_target: vec![],
            handeye_mode: HandEyeMode::EyeInHand,
            gripper_se3_rig: Some(Iso3::identity()),
            rig_se3_base: None,
            base_se3_target: Some(Iso3::translation(0.0, 0.0, 1.0)),
            gripper_se3_target: None,
            robot_deltas: None,
            mean_reproj_error: 0.5,
            per_cam_reproj_errors: vec![0.5; num_cameras],
            per_feature_residuals: PerFeatureResiduals::default(),
            image_manifest: None,
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
        let cam = |idx: usize| CameraSource {
            id: format!("cam{idx}"),
            images: ImagePattern::List {
                paths: (0..num_views)
                    .map(|v| PathBuf::from(format!("cam{idx}/target_{v}.png")))
                    .collect(),
            },
            roi_xywh: None,
            laser_images: Some(ImagePattern::List {
                paths: (0..num_views)
                    .map(|v| PathBuf::from(format!("cam{idx}/laser_{v}.png")))
                    .collect(),
            }),
        };
        DatasetSpec {
            version: 1,
            cameras: vec![cam(0), cam(1)],
            target: TargetSpec::Chessboard {
                rows: 9,
                cols: 6,
                square_size_m: 0.025,
            },
            detector: None,
            robot_poses: Some(RobotPoseSource {
                path: PathBuf::from("poses.txt"),
                format: RobotPoseFormat::Rowmajor4x4,
                columns: None,
                matrix_field: None,
            }),
            laser: None,
            upstream_calibration: Some(PathBuf::from("rig_handeye_export.json")),
            topology: Topology::RigLaserlineDevice,
            pose_pairing: Some(PosePairing::ByIndex),
            pose_convention: Some(PoseConvention {
                transform: TransformConvention::TBaseTcp,
                rotation_format: RotationFormat::Matrix4x4RowMajor,
                translation_units: TranslationUnits::M,
            }),
            unresolved: vec![],
            description: None,
        }
    }

    fn seed_rig_views(dir: &Path, cache: &FsDetectionCache, num_views: usize) {
        for cam in 0..2 {
            for view in 0..num_views {
                seed_target(
                    dir,
                    cache,
                    &format!("cam{cam}/target_{view}.png"),
                    grid_features(6),
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
    fn rig_laserline_builds_input_with_upstream_chain() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        seed_rig_views(tmp.path(), &cache, 3);
        let spec = rig_laserline_spec(tmp.path(), 3);
        let fake = FakeLaser::new(30);
        let result =
            build_rig_laserline_device_input(&spec, tmp.path(), &cache, &fake, false).unwrap();
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
        assert_eq!(input.upstream.intrinsics.len(), 2);
        assert!(
            input.upstream.sensors[0].tilt_x.abs() < 1e-12,
            "pinhole upstream gets zero Scheimpflug tilt"
        );
        // EyeInHand chain with identity T_G_R:
        //   T_R_T_i = T_B_G_i^-1 * T_B_T = (-tx_i, 0, 1).
        let t = input.upstream.rig_se3_target[1].translation.vector;
        assert!(
            (t - nalgebra::Vector3::new(-0.2, 0.0, 1.0)).norm() < 1e-12,
            "got {t:?}"
        );
        assert!(input.initial_planes_cam.is_none());
    }

    #[test]
    fn rig_handeye_laserline_builds_joint_input_without_upstream() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        seed_rig_views(tmp.path(), &cache, 3);
        let mut spec = rig_laserline_spec(tmp.path(), 3);
        spec.topology = Topology::RigHandeyeLaserline;
        spec.upstream_calibration = None;
        let fake = FakeLaser::new(30);

        let result =
            build_rig_handeye_laserline_input(&spec, tmp.path(), &cache, &fake, false).unwrap();
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
        upstream.base_se3_target = Some(Iso3::translation(0.0, 0.0, 9.0));
        std::fs::write(
            tmp.path().join("rig_handeye_export.json"),
            serde_json::to_string(&upstream).unwrap(),
        )
        .unwrap();

        let fake = FakeLaser::new(30);
        let result =
            build_rig_laserline_device_input(&spec, tmp.path(), &cache, &fake, false).unwrap();
        let t = result.input.upstream.rig_se3_target[1].translation.vector;
        assert!(
            (t - nalgebra::Vector3::new(2.0, 0.0, 1.0)).norm() < 1e-12,
            "got {t:?}"
        );
    }

    #[test]
    fn rig_laserline_fallback_chain_applies_upstream_robot_deltas() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        seed_rig_views(tmp.path(), &cache, 3);
        let spec = rig_laserline_spec(tmp.path(), 3);
        let mut upstream = upstream_export(2);
        upstream.robot_deltas = Some(vec![[0.0; 6], [0.0, 0.0, 0.0, 0.05, 0.0, 0.0], [0.0; 6]]);
        std::fs::write(
            tmp.path().join("rig_handeye_export.json"),
            serde_json::to_string(&upstream).unwrap(),
        )
        .unwrap();

        let fake = FakeLaser::new(30);
        let result =
            build_rig_laserline_device_input(&spec, tmp.path(), &cache, &fake, false).unwrap();
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
        let err =
            build_rig_laserline_device_input(&spec, tmp.path(), &cache, &fake, false).unwrap_err();
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
        let err =
            build_rig_laserline_device_input(&spec, tmp.path(), &cache, &fake, false).unwrap_err();
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
        let err =
            build_rig_laserline_device_input(&spec, tmp.path(), &cache, &fake, false).unwrap_err();
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
        spec.topology = Topology::PlanarIntrinsics;
        assert!(matches!(
            build_laserline_device_input(&spec, tmp.path(), &cache, &fake, false).unwrap_err(),
            RunError::UnsupportedTopology { .. }
        ));
        assert!(matches!(
            build_rig_laserline_device_input(&spec, tmp.path(), &cache, &fake, false).unwrap_err(),
            RunError::UnsupportedTopology { .. }
        ));
        assert!(matches!(
            build_rig_handeye_laserline_input(&spec, tmp.path(), &cache, &fake, false).unwrap_err(),
            RunError::UnsupportedTopology { .. }
        ));
    }
}
