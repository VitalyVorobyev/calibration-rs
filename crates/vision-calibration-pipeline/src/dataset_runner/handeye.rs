//! `DatasetSpec → SingleCamHandeyeInput` converter for the
//! `SingleCamHandeye` topology.
//!
//! Detection mirrors the planar converter (one camera, ROI-aware,
//! cached); pose loading and pose-to-view matching are shared with the
//! rig hand-eye converter (`poses.rs`). Views with fewer than 4
//! features are dropped — `by_index` pose alignment is preserved
//! because poses pair with the *pre-drop* image index.

use std::path::{Path, PathBuf};

use vision_calibration_core::View;
use vision_calibration_dataset::{DatasetSpec, Topology, validate};
use vision_calibration_detect::DetectionCache;

use crate::single_cam_handeye::{HandeyeMeta, SingleCamHandeyeInput};

use super::pairing::pair_views;
use super::poses::{load_robot_poses, match_poses_to_tokens};
use super::{
    RunError, augment_config_with_roi, detect_features, expand_camera_images, features_to_obs,
    pattern_repr, pick_detector, target_to_detector_config,
};

/// Result of a single-camera hand-eye dataset conversion.
#[derive(Debug)]
#[non_exhaustive]
pub struct HandeyeRunResult {
    /// The IR ready to feed into `CalibrationSession::set_input`.
    pub input: SingleCamHandeyeInput,
    /// One source image path per accepted view, in the same order.
    /// Used to populate the export's `image_manifest` after the solve.
    pub view_paths: Vec<PathBuf>,
    /// Number of views where detection produced ≥4 features.
    pub usable_views: usize,
    /// Total number of images attempted.
    pub total_views: usize,
}

/// Convert a single-camera + robot-poses manifest + cache into a
/// [`SingleCamHandeyeInput`] for `SingleCamHandeye`.
pub fn build_single_cam_handeye_input(
    spec: &DatasetSpec,
    base_dir: &Path,
    cache: &dyn DetectionCache,
    force_redetect: bool,
) -> Result<HandeyeRunResult, RunError> {
    // Same gate ordering as the other converters: topology + target +
    // validation + pairing config before any filesystem access.
    if spec.topology != Topology::SingleCamHandeye {
        return Err(RunError::UnsupportedTopology {
            topology: spec.topology,
        });
    }
    let (detector_name, detector_config) = target_to_detector_config(spec)?;
    let detector = pick_detector(detector_name)?;
    validate(spec)?;

    if spec.pose_pairing.is_none() {
        return Err(RunError::AskUser {
            field: "pose_pairing".into(),
            prompt: "Hand-eye topologies need to know how images align with robot poses. \
                     Pick by_index (pose row i pairs with the i-th image in natural-sort \
                     order) or shared_filename_token (a regex extracts the pose id from \
                     filenames)."
                .into(),
            suggestions: vec!["by_index".into(), "shared_filename_token".into()],
        });
    }

    // The validator enforces exactly one camera and the presence of
    // robot_poses + pose_convention for this topology.
    let camera = spec
        .cameras
        .first()
        .expect("validate guarantees exactly one camera for SingleCamHandeye");
    let images = expand_camera_images(camera, base_dir)?;
    if images.is_empty() {
        return Err(RunError::EmptyImageMatch {
            camera_id: camera.id.clone(),
            pattern: pattern_repr(&camera.images),
            base: base_dir.to_path_buf(),
        });
    }

    // Reuse the multi-camera pairing machinery with a single camera:
    // it produces one token per image (index or filename token), which
    // is exactly what pose matching needs.
    let pairing = spec
        .pose_pairing
        .as_ref()
        .expect("checked above; pose_pairing is present");
    let paired = pair_views(
        std::slice::from_ref(&images),
        std::slice::from_ref(&camera.id),
        pairing,
    )?;
    let total_views = paired.paths.len();

    let source = spec
        .robot_poses
        .as_ref()
        .expect("validate guarantees robot_poses for SingleCamHandeye");
    let convention = spec
        .pose_convention
        .as_ref()
        .expect("validate guarantees pose_convention for SingleCamHandeye");
    let poses = load_robot_poses(source, convention, base_dir)?;

    let roi = camera.roi_xywh;
    let key_config = augment_config_with_roi(&detector_config, roi);

    let mut kept_obs = Vec::new();
    let mut kept_tokens: Vec<String> = Vec::new();
    let mut view_paths: Vec<PathBuf> = Vec::new();
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
            // Too few features for a homography — drop the view (still
            // counted in total_views, keeping by_index pose alignment).
            continue;
        }
        kept_obs.push(features_to_obs(&features));
        kept_tokens.push(token.clone());
        view_paths.push(path.clone());
    }

    if kept_obs.is_empty() {
        return Err(RunError::InsufficientUsableViews {
            camera_id: camera.id.clone(),
            usable: 0,
            total: total_views,
        });
    }

    let matched = match_poses_to_tokens(&kept_tokens, &poses, spec, total_views)?;
    let views: Vec<View<HandeyeMeta>> = kept_obs
        .into_iter()
        .zip(matched)
        .map(|(obs, base_se3_gripper)| View {
            obs,
            meta: HandeyeMeta { base_se3_gripper },
        })
        .collect();

    let usable_views = views.len();
    let input = SingleCamHandeyeInput::new(views).map_err(|e| RunError::AskUser {
        field: "cameras[0].images".into(),
        prompt: format!("hand-eye input rejected: {e}"),
        suggestions: vec![],
    })?;

    Ok(HandeyeRunResult {
        input,
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
        CameraSource, ImagePattern, PoseColumnMap, PoseConvention, PosePairing, RobotPoseFormat,
        RobotPoseSource, RotationFormat, TargetSpec, TransformConvention, TranslationUnits,
    };
    use vision_calibration_detect::{CacheKey, CachedFeatures, Feature, FsDetectionCache};

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
        let bytes = rel_path.as_bytes();
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(bytes).unwrap();
        let key = CacheKey::from_inputs(bytes, "chessboard", &detector_config());
        cache.put(&key, &CachedFeatures { features }).unwrap();
        path
    }

    fn handeye_spec(dir: &Path, image_paths: &[&str], pose_lines: &str) -> DatasetSpec {
        let pose_path = dir.join("poses.txt");
        std::fs::write(&pose_path, pose_lines).unwrap();
        DatasetSpec {
            version: 1,
            cameras: vec![CameraSource {
                id: "cam0".into(),
                images: ImagePattern::List {
                    paths: image_paths.iter().map(PathBuf::from).collect(),
                },
                roi_xywh: None,
                laser_images: None,
            }],
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
            upstream_calibration: None,
            topology: Topology::SingleCamHandeye,
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

    fn identity_pose_line(tx: f64) -> String {
        format!("1 0 0 {tx}  0 1 0 0  0 0 1 0  0 0 0 1\n")
    }

    #[test]
    fn by_index_builds_input_with_poses() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        for view in 0..3 {
            seed_image(
                tmp.path(),
                &cache,
                &format!("img_{view}.png"),
                grid_features(6),
            );
        }
        let poses: String = (0..3)
            .map(|i| identity_pose_line((i + 1) as f64 / 10.0))
            .collect();
        let spec = handeye_spec(tmp.path(), &["img_0.png", "img_1.png", "img_2.png"], &poses);
        let result = build_single_cam_handeye_input(&spec, tmp.path(), &cache, false).unwrap();
        assert_eq!(result.input.num_views(), 3);
        assert_eq!((result.usable_views, result.total_views), (3, 3));
        let xs: Vec<f64> = result
            .input
            .views
            .iter()
            .map(|v| v.meta.base_se3_gripper.translation.x)
            .collect();
        assert_eq!(xs, vec![0.1, 0.2, 0.3]);
        assert!(result.view_paths[2].ends_with("img_2.png"));
    }

    #[test]
    fn weak_view_dropped_but_pose_alignment_kept() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        seed_image(tmp.path(), &cache, "img_0.png", grid_features(6));
        // Middle view: too few features → dropped.
        seed_image(tmp.path(), &cache, "img_1.png", grid_features(3));
        seed_image(tmp.path(), &cache, "img_2.png", grid_features(6));
        let poses: String = (0..3)
            .map(|i| identity_pose_line((i + 1) as f64 / 10.0))
            .collect();
        let spec = handeye_spec(tmp.path(), &["img_0.png", "img_1.png", "img_2.png"], &poses);
        let result = build_single_cam_handeye_input(&spec, tmp.path(), &cache, false).unwrap();
        assert_eq!((result.usable_views, result.total_views), (2, 3));
        // The surviving views must keep poses 0 and 2 — not 0 and 1.
        let xs: Vec<f64> = result
            .input
            .views
            .iter()
            .map(|v| v.meta.base_se3_gripper.translation.x)
            .collect();
        assert_eq!(xs, vec![0.1, 0.3]);
    }

    #[test]
    fn pose_count_mismatch_is_actionable() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        seed_image(tmp.path(), &cache, "img_0.png", grid_features(6));
        seed_image(tmp.path(), &cache, "img_1.png", grid_features(6));
        let spec = handeye_spec(
            tmp.path(),
            &["img_0.png", "img_1.png"],
            &identity_pose_line(0.1),
        );
        let err = build_single_cam_handeye_input(&spec, tmp.path(), &cache, false).unwrap_err();
        match err {
            RunError::PoseCountMismatch { poses, views } => assert_eq!((poses, views), (1, 2)),
            other => panic!("expected PoseCountMismatch, got {other:?}"),
        }
    }

    #[test]
    fn wrong_topology_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        let mut spec = handeye_spec(tmp.path(), &["a.png"], &identity_pose_line(0.1));
        spec.topology = Topology::PlanarIntrinsics;
        let err = build_single_cam_handeye_input(&spec, tmp.path(), &cache, false).unwrap_err();
        assert!(matches!(err, RunError::UnsupportedTopology { .. }));
    }

    #[test]
    fn missing_pose_pairing_asks_user() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        let mut spec = handeye_spec(tmp.path(), &["a.png"], &identity_pose_line(0.1));
        spec.pose_pairing = None;
        let err = build_single_cam_handeye_input(&spec, tmp.path(), &cache, false).unwrap_err();
        match err {
            RunError::AskUser { field, .. } => assert_eq!(field, "pose_pairing"),
            other => panic!("expected AskUser, got {other:?}"),
        }
    }

    #[test]
    fn token_pairing_matches_poses_by_id() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        seed_image(tmp.path(), &cache, "img_0.png", grid_features(6));
        seed_image(tmp.path(), &cache, "img_1.png", grid_features(6));
        // CSV with pose ids deliberately out of file order.
        let csv = "view,tx,ty,tz,qx,qy,qz,qw\n\
                   1,0.2,0,0,0,0,0,1\n\
                   0,0.1,0,0,0,0,0,1";
        let mut spec = handeye_spec(tmp.path(), &["img_0.png", "img_1.png"], "");
        std::fs::write(tmp.path().join("poses.csv"), csv).unwrap();
        spec.robot_poses = Some(RobotPoseSource {
            path: PathBuf::from("poses.csv"),
            format: RobotPoseFormat::Csv,
            columns: Some(PoseColumnMap {
                pose_id: Some("view".into()),
                tx: "tx".into(),
                ty: "ty".into(),
                tz: "tz".into(),
                rotation: vec!["qx".into(), "qy".into(), "qz".into(), "qw".into()],
            }),
            matrix_field: None,
        });
        spec.pose_convention = Some(PoseConvention {
            transform: TransformConvention::TBaseTcp,
            rotation_format: RotationFormat::QuatXyzw,
            translation_units: TranslationUnits::M,
        });
        spec.pose_pairing = Some(PosePairing::SharedFilenameToken {
            regex: r"^img_(?<view>\d+)\.png$".into(),
            group: "view".into(),
        });
        let result = build_single_cam_handeye_input(&spec, tmp.path(), &cache, false).unwrap();
        let xs: Vec<f64> = result
            .input
            .views
            .iter()
            .map(|v| v.meta.base_se3_gripper.translation.x)
            .collect();
        assert_eq!(xs, vec![0.1, 0.2]);
    }
}
