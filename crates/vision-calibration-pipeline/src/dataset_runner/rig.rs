//! `DatasetSpec → RigDataset` converters for the `RigExtrinsics` and
//! `RigHandeye` topologies.
//!
//! Both share the same per-view/per-camera detection loop; hand-eye
//! additionally loads robot poses from the manifest's
//! [`RobotPoseSource`](vision_calibration_dataset::RobotPoseSource)
//! and attaches one `RobotPoseMeta` per kept view.
//!
//! A camera that sees fewer than 4 features in a view contributes a
//! `None` slot (the rig IR supports per-camera gaps); a view where
//! *no* camera reaches 4 features is dropped entirely.

use std::path::{Path, PathBuf};

use vision_calibration_core::{CorrespondenceView, NoMeta, RigDataset, RigView, RigViewObs};
use vision_calibration_dataset::{DatasetSpec, PosePairing, Topology, validate};
use vision_calibration_detect::DetectionCache;
use vision_calibration_optim::RobotPoseMeta;

use super::pairing::pair_views;
use super::poses::{ParsedPose, load_robot_poses};
use super::{
    RunError, augment_config_with_roi, detect_features, expand_camera_images, features_to_obs,
    pattern_repr, pick_detector, target_to_detector_config,
};

/// Result of a rig dataset conversion.
#[derive(Debug)]
#[non_exhaustive]
pub struct RigRunResult<Meta> {
    /// The IR ready to feed into `CalibrationSession::set_input`.
    pub dataset: RigDataset<Meta>,
    /// `view_paths[view][camera]` — source image path per kept view
    /// and camera, aligned with `dataset.views`; `None` where the
    /// camera contributed no usable observation.
    pub view_paths: Vec<Vec<Option<PathBuf>>>,
    /// Number of kept views (≥1 camera with ≥4 features).
    pub usable_views: usize,
    /// Total number of paired views attempted.
    pub total_views: usize,
}

/// Convert a multi-camera manifest + cache into a
/// [`RigDataset<NoMeta>`] for `RigExtrinsics`.
pub fn build_rig_extrinsics_input(
    spec: &DatasetSpec,
    base_dir: &Path,
    cache: &dyn DetectionCache,
    force_redetect: bool,
) -> Result<RigRunResult<NoMeta>, RunError> {
    if spec.topology != Topology::RigExtrinsics {
        return Err(RunError::UnsupportedTopology {
            topology: spec.topology,
        });
    }
    let core = build_rig_core(spec, base_dir, cache, force_redetect)?;
    let views = core
        .views
        .into_iter()
        .map(|(obs, _token)| RigView { obs, meta: NoMeta })
        .collect();
    finish(views, spec.cameras.len(), core.view_paths, core.total_views)
}

/// Convert a multi-camera + robot-poses manifest + cache into a
/// [`RigDataset<RobotPoseMeta>`] for `RigHandeye`.
pub fn build_rig_handeye_input(
    spec: &DatasetSpec,
    base_dir: &Path,
    cache: &dyn DetectionCache,
    force_redetect: bool,
) -> Result<RigRunResult<RobotPoseMeta>, RunError> {
    if spec.topology != Topology::RigHandeye {
        return Err(RunError::UnsupportedTopology {
            topology: spec.topology,
        });
    }
    let core = build_rig_core(spec, base_dir, cache, force_redetect)?;

    // `validate` (called inside build_rig_core) guarantees both fields
    // are present for the RigHandeye topology.
    let source = spec
        .robot_poses
        .as_ref()
        .expect("validate guarantees robot_poses for RigHandeye");
    let convention = spec
        .pose_convention
        .as_ref()
        .expect("validate guarantees pose_convention for RigHandeye");
    let poses = load_robot_poses(source, convention, base_dir)?;

    let views = pair_poses_to_views(core.views, &poses, spec, core.total_views)?;
    finish(views, spec.cameras.len(), core.view_paths, core.total_views)
}

/// Kept views (observations + pairing token), pre-meta.
struct RigCore {
    views: Vec<(RigViewObs, String)>,
    view_paths: Vec<Vec<Option<PathBuf>>>,
    total_views: usize,
}

fn build_rig_core(
    spec: &DatasetSpec,
    base_dir: &Path,
    cache: &dyn DetectionCache,
    force_redetect: bool,
) -> Result<RigCore, RunError> {
    // Same gate ordering as the planar converter: target + validation
    // + pairing config before any filesystem access.
    let (detector_name, detector_config) = target_to_detector_config(&spec.target)?;
    let detector = pick_detector(detector_name)?;
    validate(spec)?;

    let Some(pairing) = &spec.pose_pairing else {
        return Err(RunError::AskUser {
            field: "pose_pairing".into(),
            prompt: "Rig topologies need to know how per-camera images align into views. \
                     Pick by_index (same count and order everywhere) or \
                     shared_filename_token (a regex extracts a view token from filenames)."
                .into(),
            suggestions: vec!["by_index".into(), "shared_filename_token".into()],
        });
    };

    let mut images: Vec<Vec<PathBuf>> = Vec::with_capacity(spec.cameras.len());
    for camera in &spec.cameras {
        let expanded = expand_camera_images(camera, base_dir)?;
        if expanded.is_empty() {
            return Err(RunError::EmptyImageMatch {
                camera_id: camera.id.clone(),
                pattern: pattern_repr(&camera.images),
                base: base_dir.to_path_buf(),
            });
        }
        images.push(expanded);
    }
    let camera_ids: Vec<String> = spec.cameras.iter().map(|c| c.id.clone()).collect();
    let paired = pair_views(&images, &camera_ids, pairing)?;
    let total_views = paired.paths.len();

    // Per-camera key configs (ROI differs per camera).
    let key_configs: Vec<_> = spec
        .cameras
        .iter()
        .map(|c| augment_config_with_roi(&detector_config, c.roi_xywh))
        .collect();

    let mut views: Vec<(RigViewObs, String)> = Vec::new();
    let mut view_paths: Vec<Vec<Option<PathBuf>>> = Vec::new();
    let mut per_camera_usable = vec![0usize; spec.cameras.len()];

    for (paths, token) in paired.paths.iter().zip(&paired.tokens) {
        let mut cameras: Vec<Option<CorrespondenceView>> = Vec::with_capacity(paths.len());
        let mut kept_paths: Vec<Option<PathBuf>> = Vec::with_capacity(paths.len());
        for (cam_idx, maybe_path) in paths.iter().enumerate() {
            let Some(path) = maybe_path else {
                cameras.push(None);
                kept_paths.push(None);
                continue;
            };
            let (features, _cache_hit) = detect_features(
                detector.as_ref(),
                detector_name,
                &detector_config,
                &key_configs[cam_idx],
                spec.cameras[cam_idx].roi_xywh,
                path,
                cache,
                force_redetect,
            )?;
            if features.len() < 4 {
                cameras.push(None);
                kept_paths.push(None);
                continue;
            }
            per_camera_usable[cam_idx] += 1;
            cameras.push(Some(features_to_obs(&features)));
            kept_paths.push(Some(path.clone()));
        }
        if cameras.iter().all(Option::is_none) {
            // No camera saw the target in this view — drop it (still
            // counted in total_views).
            continue;
        }
        views.push((RigViewObs { cameras }, token.clone()));
        view_paths.push(kept_paths);
    }

    // Every camera must contribute at least one usable observation —
    // a camera with zero observations cannot be placed in the rig.
    for (cam_idx, usable) in per_camera_usable.iter().enumerate() {
        if *usable == 0 {
            return Err(RunError::InsufficientUsableViews {
                camera_id: spec.cameras[cam_idx].id.clone(),
                usable: 0,
                total: total_views,
            });
        }
    }

    Ok(RigCore {
        views,
        view_paths,
        total_views,
    })
}

/// Attach one robot pose to every kept view.
fn pair_poses_to_views(
    views: Vec<(RigViewObs, String)>,
    poses: &[ParsedPose],
    spec: &DatasetSpec,
    total_views: usize,
) -> Result<Vec<RigView<RobotPoseMeta>>, RunError> {
    let pairing = spec
        .pose_pairing
        .as_ref()
        .expect("build_rig_core already required pose_pairing");

    match pairing {
        PosePairing::ByIndex => {
            // Poses pair with *paired* views (pre-drop): pose[i] belongs
            // to view token i even when that view was later dropped.
            if poses.len() != total_views {
                return Err(RunError::PoseCountMismatch {
                    poses: poses.len(),
                    views: total_views,
                });
            }
            views
                .into_iter()
                .map(|(obs, token)| {
                    let index: usize = token
                        .parse()
                        .expect("by_index pairing tokens are stringified indices");
                    Ok(RigView {
                        obs,
                        meta: RobotPoseMeta {
                            base_se3_gripper: poses[index].base_se3_gripper,
                        },
                    })
                })
                .collect()
        }
        PosePairing::SharedFilenameToken { .. } => {
            // Poses pair by their pose_id column equalling the view token.
            if spec
                .robot_poses
                .as_ref()
                .is_none_or(|s| s.columns.pose_id.is_none())
            {
                return Err(RunError::AskUser {
                    field: "robot_poses.columns.pose_id".into(),
                    prompt: "shared_filename_token pairing needs a pose_id column mapping \
                             so each pose row can be matched to its view token."
                        .into(),
                    suggestions: vec![],
                });
            }
            let by_id: std::collections::HashMap<&str, &ParsedPose> = poses
                .iter()
                .filter_map(|p| p.id.as_deref().map(|id| (id, p)))
                .collect();
            views
                .into_iter()
                .map(|(obs, token)| {
                    let pose = by_id.get(token.as_str()).ok_or_else(|| {
                        RunError::PairingTokenMismatch {
                            message: format!(
                                "no pose row with pose_id {token:?} (the cameras produced a \
                                 view with this token)"
                            ),
                        }
                    })?;
                    Ok(RigView {
                        obs,
                        meta: RobotPoseMeta {
                            base_se3_gripper: pose.base_se3_gripper,
                        },
                    })
                })
                .collect()
        }
    }
}

fn finish<Meta>(
    views: Vec<RigView<Meta>>,
    num_cameras: usize,
    view_paths: Vec<Vec<Option<PathBuf>>>,
    total_views: usize,
) -> Result<RigRunResult<Meta>, RunError> {
    let usable_views = views.len();
    let dataset = RigDataset::new(views, num_cameras).expect(
        "RigDataset::new failed after the runner enforced non-empty views \
         and per-view camera counts; this indicates a core-crate regression",
    );
    Ok(RigRunResult {
        dataset,
        view_paths,
        usable_views,
        total_views,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Write;
    use vision_calibration_dataset::{
        CameraSource, ImagePattern, PoseColumnMap, PoseConvention, RobotPoseFormat,
        RobotPoseSource, RotationFormat, TargetSpec, TransformConvention, TranslationUnits,
    };
    use vision_calibration_detect::{CacheKey, CachedFeatures, Feature, FsDetectionCache};

    /// Synthetic features on the chessboard lattice — enough (≥4) for
    /// the runner to accept the view.
    fn grid_features(n: usize) -> Vec<Feature> {
        (0..n)
            .map(|i| Feature {
                image_xy: [100.0 + 10.0 * i as f64, 200.0 + 5.0 * i as f64],
                world_xyz: [0.025 * i as f64, 0.0, 0.0],
            })
            .collect()
    }

    fn detector_config() -> serde_json::Value {
        json!({"rows": 9, "cols": 6, "square_size_m": 0.025})
    }

    /// Write a dummy "image" file and pre-seed the cache for it, so the
    /// runner takes the cache-hit path and never decodes the bytes.
    fn seed_image(
        dir: &Path,
        cache: &FsDetectionCache,
        rel_path: &str,
        features: Vec<Feature>,
    ) -> PathBuf {
        let path = dir.join(rel_path);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let bytes = rel_path.as_bytes(); // content only needs to be unique
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(bytes).unwrap();
        let key = CacheKey::from_inputs(bytes, "chessboard", &detector_config());
        cache.put(&key, &CachedFeatures { features }).unwrap();
        path
    }

    fn rig_spec(camera_lists: &[Vec<&str>]) -> DatasetSpec {
        DatasetSpec {
            version: 1,
            cameras: camera_lists
                .iter()
                .enumerate()
                .map(|(i, paths)| CameraSource {
                    id: format!("cam{i}"),
                    images: ImagePattern::List {
                        paths: paths.iter().map(PathBuf::from).collect(),
                    },
                    roi_xywh: None,
                })
                .collect(),
            target: TargetSpec::Chessboard {
                rows: 9,
                cols: 6,
                square_size_m: 0.025,
            },
            robot_poses: None,
            topology: Topology::RigExtrinsics,
            pose_pairing: Some(PosePairing::ByIndex),
            pose_convention: None,
            unresolved: vec![],
            description: None,
        }
    }

    #[test]
    fn rig_extrinsics_by_index_builds_dataset() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        for cam in 0..2 {
            for view in 0..3 {
                seed_image(
                    tmp.path(),
                    &cache,
                    &format!("cam{cam}/img_{view}.png"),
                    grid_features(6),
                );
            }
        }
        let spec = rig_spec(&[
            vec!["cam0/img_0.png", "cam0/img_1.png", "cam0/img_2.png"],
            vec!["cam1/img_0.png", "cam1/img_1.png", "cam1/img_2.png"],
        ]);
        let result = build_rig_extrinsics_input(&spec, tmp.path(), &cache, false).unwrap();
        assert_eq!(result.dataset.num_cameras, 2);
        assert_eq!(result.dataset.num_views(), 3);
        assert_eq!(result.usable_views, 3);
        assert_eq!(result.total_views, 3);
        assert!(
            result.view_paths[0][0]
                .as_ref()
                .unwrap()
                .ends_with("cam0/img_0.png")
        );
        assert!(result.dataset.views[2].obs.cameras[1].is_some());
    }

    #[test]
    fn weak_view_becomes_none_slot() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        seed_image(tmp.path(), &cache, "cam0/a.png", grid_features(6));
        seed_image(tmp.path(), &cache, "cam0/b.png", grid_features(6));
        seed_image(tmp.path(), &cache, "cam1/a.png", grid_features(6));
        // cam1's second view has too few features → None slot.
        seed_image(tmp.path(), &cache, "cam1/b.png", grid_features(3));
        let spec = rig_spec(&[
            vec!["cam0/a.png", "cam0/b.png"],
            vec!["cam1/a.png", "cam1/b.png"],
        ]);
        let result = build_rig_extrinsics_input(&spec, tmp.path(), &cache, false).unwrap();
        assert_eq!(result.dataset.num_views(), 2);
        assert!(result.dataset.views[1].obs.cameras[0].is_some());
        assert!(result.dataset.views[1].obs.cameras[1].is_none());
        assert!(result.view_paths[1][1].is_none());
    }

    #[test]
    fn camera_with_zero_usable_views_is_an_error() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        seed_image(tmp.path(), &cache, "cam0/a.png", grid_features(6));
        seed_image(tmp.path(), &cache, "cam1/a.png", grid_features(2));
        let spec = rig_spec(&[vec!["cam0/a.png"], vec!["cam1/a.png"]]);
        let err = build_rig_extrinsics_input(&spec, tmp.path(), &cache, false).unwrap_err();
        match err {
            RunError::InsufficientUsableViews { camera_id, .. } => assert_eq!(camera_id, "cam1"),
            other => panic!("expected InsufficientUsableViews, got {other:?}"),
        }
    }

    #[test]
    fn missing_pose_pairing_asks_user() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        let mut spec = rig_spec(&[vec!["cam0/a.png"], vec!["cam1/a.png"]]);
        spec.pose_pairing = None;
        let err = build_rig_extrinsics_input(&spec, tmp.path(), &cache, false).unwrap_err();
        match err {
            RunError::AskUser { field, .. } => assert_eq!(field, "pose_pairing"),
            other => panic!("expected AskUser, got {other:?}"),
        }
    }

    #[test]
    fn wrong_topology_rejected_by_both_converters() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        let spec = rig_spec(&[vec!["a.png"], vec!["b.png"]]); // RigExtrinsics
        let err = build_rig_handeye_input(&spec, tmp.path(), &cache, false).unwrap_err();
        assert!(matches!(err, RunError::UnsupportedTopology { .. }));

        let mut spec2 = rig_spec(&[vec!["a.png"], vec!["b.png"]]);
        spec2.topology = Topology::RigHandeye;
        let err2 = build_rig_extrinsics_input(&spec2, tmp.path(), &cache, false).unwrap_err();
        assert!(matches!(err2, RunError::UnsupportedTopology { .. }));
    }

    fn handeye_spec_with_poses(dir: &Path, csv: &str) -> DatasetSpec {
        let pose_path = dir.join("poses.csv");
        std::fs::write(&pose_path, csv).unwrap();
        let mut spec = rig_spec(&[
            vec!["cam0/img_0.png", "cam0/img_1.png", "cam0/img_2.png"],
            vec!["cam1/img_0.png", "cam1/img_1.png", "cam1/img_2.png"],
        ]);
        spec.topology = Topology::RigHandeye;
        spec.robot_poses = Some(RobotPoseSource {
            path: PathBuf::from("poses.csv"),
            format: RobotPoseFormat::Csv,
            columns: PoseColumnMap {
                pose_id: None,
                tx: "tx".into(),
                ty: "ty".into(),
                tz: "tz".into(),
                rotation: vec!["qx".into(), "qy".into(), "qz".into(), "qw".into()],
            },
        });
        spec.pose_convention = Some(PoseConvention {
            transform: TransformConvention::TBaseTcp,
            rotation_format: RotationFormat::QuatXyzw,
            translation_units: TranslationUnits::M,
        });
        spec
    }

    #[test]
    fn rig_handeye_by_index_attaches_poses() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        for cam in 0..2 {
            for view in 0..3 {
                seed_image(
                    tmp.path(),
                    &cache,
                    &format!("cam{cam}/img_{view}.png"),
                    grid_features(6),
                );
            }
        }
        let csv = "tx,ty,tz,qx,qy,qz,qw\n\
                   0.1,0,0,0,0,0,1\n\
                   0.2,0,0,0,0,0,1\n\
                   0.3,0,0,0,0,0,1";
        let spec = handeye_spec_with_poses(tmp.path(), csv);
        let result = build_rig_handeye_input(&spec, tmp.path(), &cache, false).unwrap();
        assert_eq!(result.dataset.num_views(), 3);
        let xs: Vec<f64> = result
            .dataset
            .views
            .iter()
            .map(|v| v.meta.base_se3_gripper.translation.x)
            .collect();
        assert_eq!(xs, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn rig_handeye_pose_count_mismatch_is_actionable() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        for cam in 0..2 {
            for view in 0..3 {
                seed_image(
                    tmp.path(),
                    &cache,
                    &format!("cam{cam}/img_{view}.png"),
                    grid_features(6),
                );
            }
        }
        let csv = "tx,ty,tz,qx,qy,qz,qw\n0.1,0,0,0,0,0,1"; // 1 pose, 3 views
        let spec = handeye_spec_with_poses(tmp.path(), csv);
        let err = build_rig_handeye_input(&spec, tmp.path(), &cache, false).unwrap_err();
        match err {
            RunError::PoseCountMismatch { poses, views } => {
                assert_eq!((poses, views), (1, 3));
            }
            other => panic!("expected PoseCountMismatch, got {other:?}"),
        }
    }

    #[test]
    fn token_pairing_without_pose_id_asks_user() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        for cam in 0..2 {
            for view in 0..3 {
                seed_image(
                    tmp.path(),
                    &cache,
                    &format!("cam{cam}/img_{view}.png"),
                    grid_features(6),
                );
            }
        }
        let csv = "tx,ty,tz,qx,qy,qz,qw\n0.1,0,0,0,0,0,1";
        let mut spec = handeye_spec_with_poses(tmp.path(), csv);
        spec.pose_pairing = Some(PosePairing::SharedFilenameToken {
            regex: r"^img_(?<view>\d+)\.png$".into(),
            group: "view".into(),
        });
        let err = build_rig_handeye_input(&spec, tmp.path(), &cache, false).unwrap_err();
        match err {
            RunError::AskUser { field, .. } => {
                assert_eq!(field, "robot_poses.columns.pose_id")
            }
            other => panic!("expected AskUser, got {other:?}"),
        }
    }

    #[test]
    fn token_pairing_matches_poses_by_id() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        for cam in 0..2 {
            for view in 0..2 {
                seed_image(
                    tmp.path(),
                    &cache,
                    &format!("cam{cam}/img_{view}.png"),
                    grid_features(6),
                );
            }
        }
        let csv = "view,tx,ty,tz,qx,qy,qz,qw\n\
                   1,0.2,0,0,0,0,0,1\n\
                   0,0.1,0,0,0,0,0,1"; // deliberately out of order
        let mut spec = handeye_spec_with_poses(tmp.path(), csv);
        spec.cameras[0].images = ImagePattern::List {
            paths: vec!["cam0/img_0.png".into(), "cam0/img_1.png".into()],
        };
        spec.cameras[1].images = ImagePattern::List {
            paths: vec!["cam1/img_0.png".into(), "cam1/img_1.png".into()],
        };
        spec.pose_pairing = Some(PosePairing::SharedFilenameToken {
            regex: r"^img_(?<view>\d+)\.png$".into(),
            group: "view".into(),
        });
        if let Some(rp) = &mut spec.robot_poses {
            rp.columns.pose_id = Some("view".into());
        }
        let result = build_rig_handeye_input(&spec, tmp.path(), &cache, false).unwrap();
        assert_eq!(result.dataset.num_views(), 2);
        // Views are sorted by token; pose rows were reversed in the file.
        let xs: Vec<f64> = result
            .dataset
            .views
            .iter()
            .map(|v| v.meta.base_se3_gripper.translation.x)
            .collect();
        assert_eq!(xs, vec![0.1, 0.2]);
    }
}
