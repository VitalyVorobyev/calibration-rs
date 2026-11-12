//! `DatasetSpec → RigDataset` converters for the `RigExtrinsics` and
//! `RigHandeye` topologies.
//!
//! Both share the same per-view/per-camera detection loop; hand-eye
//! additionally loads robot poses from the manifest's
//! [`RobotPoseSource`](vision_calibration_dataset::RobotPoseSource)
//! and attaches one `RobotPoseMeta` per kept view.
//!
//! A camera whose detection in a view the view filter rejects
//! contributes a `None` slot (the rig IR supports per-camera gaps); a view
//! where *every* camera's detection is rejected is dropped entirely.

use std::path::{Path, PathBuf};

use vision_calibration_core::{NoMeta, RigDataset, RigView, RigViewObs, RunControl};
use vision_calibration_dataset::{DatasetSpec, ProblemKind, validate};
use vision_calibration_detect::Feature;
use vision_calibration_optim::RobotPoseMeta;

use super::board_turn::match_rig_labels;
use super::pairing::pair_views;
use super::poses::{ParsedPose, load_robot_poses, match_poses_to_tokens};
use super::require_pose_pairing;
use super::{
    Detection, ImageProgress, RejectReason, RejectedDetection, RunError, detect_features,
    expand_camera_images, features_to_obs, pattern_repr, target_to_detector_config,
    view_filter::{ViewFilter, no_usable_views},
};

/// Result of a rig dataset conversion.
#[derive(Debug)]
#[non_exhaustive]
pub struct RigRunResult<Meta> {
    /// The IR ready to feed into `CalibrationSession::set_input`.
    pub input: RigDataset<Meta>,
    /// `view_paths[view][camera]` — source image path per kept view
    /// and camera, aligned with `dataset.views`; `None` where the
    /// camera contributed no usable observation.
    pub view_paths: Vec<Vec<Option<PathBuf>>>,
    /// Number of kept views (≥1 camera with an accepted detection).
    pub usable_views: usize,
    /// Total number of paired views attempted.
    pub total_views: usize,
    /// The detections left out, in view order, each with its reason.
    pub rejected: Vec<RejectedDetection>,
}

/// Convert a multi-camera manifest + cache into a
/// [`RigDataset<NoMeta>`] for `RigExtrinsics`.
///
/// `control` as for [`build_planar_input`](super::build_planar_input).
pub fn build_rig_extrinsics_input(
    spec: &DatasetSpec,
    base_dir: &Path,
    detection: Detection<'_>,
    control: &RunControl,
) -> Result<RigRunResult<NoMeta>, RunError> {
    if spec.topology != ProblemKind::RigExtrinsics {
        return Err(RunError::UnsupportedTopology {
            topology: spec.topology,
        });
    }
    let RigCore {
        views,
        view_paths,
        total_views,
        rejected,
        ..
    } = build_rig_core(spec, base_dir, detection, control)?;
    let views = views
        .into_iter()
        .map(|(obs, _token)| RigView { obs, meta: NoMeta })
        .collect();
    finish(views, spec.cameras.len(), view_paths, total_views, rejected)
}

/// Convert a multi-camera + robot-poses manifest + cache into a
/// [`RigDataset<RobotPoseMeta>`] for `RigHandeye`.
///
/// `control` as for [`build_planar_input`](super::build_planar_input).
pub fn build_rig_handeye_input(
    spec: &DatasetSpec,
    base_dir: &Path,
    detection: Detection<'_>,
    control: &RunControl,
) -> Result<RigRunResult<RobotPoseMeta>, RunError> {
    if spec.topology != ProblemKind::RigHandeye {
        return Err(RunError::UnsupportedTopology {
            topology: spec.topology,
        });
    }
    // The robot poses load before any detection: a pose file that does not
    // parse fails the run before the expensive work. `validate` guarantees
    // both fields for the RigHandeye topology.
    vision_calibration_dataset::validate(spec)?;
    let source = spec
        .robot_poses
        .as_ref()
        .expect("validate guarantees robot_poses for RigHandeye");
    let convention = spec
        .pose_convention
        .as_ref()
        .expect("validate guarantees pose_convention for RigHandeye");
    let poses = load_robot_poses(source, convention, base_dir)?;

    let RigCore {
        views,
        view_paths,
        total_views,
        rejected,
        ..
    } = build_rig_core(spec, base_dir, detection, control)?;

    let views = pair_poses_to_views(views, &poses, spec, total_views)?;
    finish(views, spec.cameras.len(), view_paths, total_views, rejected)
}

/// Kept views (observations + pairing token), pre-meta. Shared with
/// the rig laserline converter (`laser.rs`), which pairs laser frames
/// against `all_tokens` and attaches laser pixels to the kept views.
pub(super) struct RigCore {
    pub(super) views: Vec<(RigViewObs, String)>,
    pub(super) view_paths: Vec<Vec<Option<PathBuf>>>,
    /// Tokens of *all* paired views, pre-drop, in view order.
    pub(super) all_tokens: Vec<String>,
    pub(super) total_views: usize,
    /// Detections the view filter left out, in view order.
    pub(super) rejected: Vec<RejectedDetection>,
}

pub(super) fn build_rig_core(
    spec: &DatasetSpec,
    base_dir: &Path,
    detection: Detection<'_>,
    control: &RunControl,
) -> Result<RigCore, RunError> {
    // Same gate ordering as the planar converter: target + validation
    // + pairing config before any filesystem access.
    let detector_config = target_to_detector_config(spec)?;
    validate(spec)?;

    let pairing = require_pose_pairing(spec)?
        .expect("every rig and laser topology needs a pairing; require_pose_pairing asks for it");

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

    let filter = ViewFilter::for_spec(spec);
    let mut rejected: Vec<(usize, RejectedDetection)> = Vec::new();
    // detected[view][camera]: the features of each detection the filter
    // keeps.
    let mut detected: Vec<Vec<Option<Vec<Feature>>>> = Vec::with_capacity(total_views);
    let images = paired.paths.iter().flatten().flatten().count();
    let mut progress = ImageProgress::targets(control, images);
    for (view, paths) in paired.paths.iter().enumerate() {
        let mut cameras = Vec::with_capacity(paths.len());
        for (cam_idx, maybe_path) in paths.iter().enumerate() {
            let Some(path) = maybe_path else {
                cameras.push(None);
                continue;
            };
            let (features, _cache_hit) = progress.read(|| {
                detect_features(
                    &detector_config,
                    spec.cameras[cam_idx].roi_xywh,
                    path,
                    detection,
                )
            })?;
            match filter.check(&features) {
                Ok(()) => cameras.push(Some(features)),
                Err(reason) => {
                    rejected.push((view, rejection(cam_idx, path, reason)));
                    cameras.push(None);
                }
            }
        }
        detected.push(cameras);
    }
    // A plain chessboard's labels are matched across the cameras of each
    // view, which needs every view at hand.
    if let Some((grid, scope)) = filter.rig_chessboard() {
        for (view, camera, reason) in match_rig_labels(&grid, scope, &mut detected) {
            let path = paired.paths[view][camera]
                .as_ref()
                .expect("a matched detection has an image");
            rejected.push((view, rejection(camera, path, reason)));
        }
        // In view order, and by camera within a view.
        rejected.sort_by_key(|(view, r)| (*view, r.camera));
    }
    let rejected: Vec<RejectedDetection> = rejected.into_iter().map(|(_, r)| r).collect();

    let mut views: Vec<(RigViewObs, String)> = Vec::new();
    let mut view_paths: Vec<Vec<Option<PathBuf>>> = Vec::new();
    let mut per_camera_usable = vec![0usize; spec.cameras.len()];
    for ((cameras, paths), token) in detected.iter().zip(&paired.paths).zip(&paired.tokens) {
        if cameras.iter().all(Option::is_none) {
            // No camera saw the target in this view — drop it (still
            // counted in total_views).
            continue;
        }
        let mut obs = Vec::with_capacity(cameras.len());
        let mut kept_paths = Vec::with_capacity(cameras.len());
        for (cam_idx, (features, path)) in cameras.iter().zip(paths).enumerate() {
            match features {
                Some(features) => {
                    per_camera_usable[cam_idx] += 1;
                    obs.push(Some(features_to_obs(features)));
                    kept_paths.push(path.clone());
                }
                None => {
                    obs.push(None);
                    kept_paths.push(None);
                }
            }
        }
        views.push((RigViewObs { cameras: obs }, token.clone()));
        view_paths.push(kept_paths);
    }

    // Every camera must contribute at least one usable observation —
    // a camera with zero observations cannot be placed in the rig.
    for (cam_idx, usable) in per_camera_usable.iter().enumerate() {
        if *usable == 0 {
            return Err(no_usable_views(spec, cam_idx, &rejected, total_views));
        }
    }

    Ok(RigCore {
        views,
        view_paths,
        all_tokens: paired.tokens,
        total_views,
        rejected,
    })
}

fn rejection(camera: usize, path: &Path, reason: RejectReason) -> RejectedDetection {
    RejectedDetection {
        camera,
        path: path.to_path_buf(),
        reason,
    }
}

/// Attach one robot pose to every kept view (matching logic shared
/// with the single-camera hand-eye converter in `poses.rs`).
fn pair_poses_to_views(
    views: Vec<(RigViewObs, String)>,
    poses: &[ParsedPose],
    spec: &DatasetSpec,
    total_views: usize,
) -> Result<Vec<RigView<RobotPoseMeta>>, RunError> {
    let tokens: Vec<String> = views.iter().map(|(_, token)| token.clone()).collect();
    let matched = match_poses_to_tokens(&tokens, poses, spec, total_views)?;
    Ok(views
        .into_iter()
        .zip(matched)
        .map(|((obs, _token), base_se3_gripper)| RigView {
            obs,
            meta: RobotPoseMeta { base_se3_gripper },
        })
        .collect())
}

fn finish<Meta>(
    views: Vec<RigView<Meta>>,
    num_cameras: usize,
    view_paths: Vec<Vec<Option<PathBuf>>>,
    total_views: usize,
    rejected: Vec<RejectedDetection>,
) -> Result<RigRunResult<Meta>, RunError> {
    let usable_views = views.len();
    let dataset = RigDataset::new(views, num_cameras).expect(
        "RigDataset::new failed after the runner enforced non-empty views \
         and per-view camera counts; this indicates a core-crate regression",
    );
    Ok(RigRunResult {
        input: dataset,
        view_paths,
        usable_views,
        total_views,
        rejected,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset_runner::RejectReason;
    use std::io::Write;
    use vision_calibration_dataset::{
        CameraSource, DetectorSpec, ImagePattern, PoseColumnMap, PoseConvention, PosePairing,
        RobotPoseFormat, RobotPoseSource, RotationFormat, TargetSpec, TransformConvention,
        TranslationUnits,
    };
    use vision_calibration_detect::DetectionCache;
    use vision_calibration_detect::{CacheKey, CachedFeatures, Feature, FsDetectionCache};

    /// Synthetic features on the chessboard lattice — enough (≥4) for
    /// the runner to accept the view.
    fn grid_features(n: usize) -> Vec<Feature> {
        (0..n)
            .map(|i| Feature {
                cov_px: None,
                image_xy: [100.0 + 10.0 * i as f64, 200.0 + 5.0 * i as f64],
                world_xyz: [0.025 * i as f64, 0.0, 0.0],
            })
            .collect()
    }

    /// Every corner of the 9-row × 6-column test board, `(u, v)` with
    /// `u < cols` and `v < rows` minus any in `missing` — a detection that
    /// spans the whole board as declared.
    fn board_features_without(missing: &[(u32, u32)]) -> Vec<Feature> {
        (0..9u32)
            .flat_map(|v| (0..6u32).map(move |u| (u, v)))
            .filter(|uv| !missing.contains(uv))
            .map(|(u, v)| Feature {
                cov_px: None,
                image_xy: [100.0 + 40.0 * f64::from(u), 50.0 + 40.0 * f64::from(v)],
                world_xyz: [0.025 * f64::from(u), 0.025 * f64::from(v), 0.0],
            })
            .collect()
    }

    fn board_features() -> Vec<Feature> {
        board_features_without(&[])
    }

    fn detector_config() -> vision_calibration_detect::DetectorConfig {
        vision_calibration_detect::DetectorConfig::Chessboard(
            vision_calibration_detect::ChessboardConfig {
                rows: 9,
                cols: 6,
                square_size_m: 0.025,
                chess_corners: None,
            },
        )
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
        let key = CacheKey::for_detection(bytes, &detector_config(), None);
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
                    laser_images: None,
                })
                .collect(),
            target: TargetSpec::Chessboard {
                rows: 9,
                cols: 6,
                square_size_m: 0.025,
            },
            detector: None,
            robot_poses: None,
            laser: None,
            upstream_calibration: None,
            topology: ProblemKind::RigExtrinsics,
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
                    board_features(),
                );
            }
        }
        let spec = rig_spec(&[
            vec!["cam0/img_0.png", "cam0/img_1.png", "cam0/img_2.png"],
            vec!["cam1/img_0.png", "cam1/img_1.png", "cam1/img_2.png"],
        ]);
        let result = build_rig_extrinsics_input(
            &spec,
            tmp.path(),
            Detection::Cached(&cache),
            &RunControl::default(),
        )
        .unwrap();
        assert_eq!(result.input.num_cameras, 2);
        assert_eq!(result.input.num_views(), 3);
        assert_eq!(result.usable_views, 3);
        assert_eq!(result.total_views, 3);
        assert!(
            result.view_paths[0][0]
                .as_ref()
                .unwrap()
                .ends_with("cam0/img_0.png")
        );
        assert!(result.input.views[2].obs.cameras[1].is_some());
    }

    #[test]
    fn weak_view_becomes_none_slot() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        seed_image(tmp.path(), &cache, "cam0/a.png", board_features());
        seed_image(tmp.path(), &cache, "cam0/b.png", board_features());
        seed_image(tmp.path(), &cache, "cam1/a.png", board_features());
        // cam1's second view has too few features → None slot.
        seed_image(tmp.path(), &cache, "cam1/b.png", grid_features(3));
        let spec = rig_spec(&[
            vec!["cam0/a.png", "cam0/b.png"],
            vec!["cam1/a.png", "cam1/b.png"],
        ]);
        let result = build_rig_extrinsics_input(
            &spec,
            tmp.path(),
            Detection::Cached(&cache),
            &RunControl::default(),
        )
        .unwrap();
        assert_eq!(result.input.num_views(), 2);
        assert!(result.input.views[1].obs.cameras[0].is_some());
        assert!(result.input.views[1].obs.cameras[1].is_none());
        assert!(result.view_paths[1][1].is_none());
    }

    #[test]
    fn manifest_feature_floor_drops_sparse_views() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        seed_image(tmp.path(), &cache, "cam0/a.png", board_features());
        seed_image(tmp.path(), &cache, "cam0/b.png", board_features());
        seed_image(tmp.path(), &cache, "cam1/a.png", board_features());
        // The whole board less one interior corner: 53 of 54.
        seed_image(
            tmp.path(),
            &cache,
            "cam1/b.png",
            board_features_without(&[(2, 4)]),
        );
        let mut spec = rig_spec(&[
            vec!["cam0/a.png", "cam0/b.png"],
            vec!["cam1/a.png", "cam1/b.png"],
        ]);
        spec.detector = Some(DetectorSpec {
            min_features_per_view: Some(54),
            ..DetectorSpec::default()
        });
        let result = build_rig_extrinsics_input(
            &spec,
            tmp.path(),
            Detection::Cached(&cache),
            &RunControl::default(),
        )
        .unwrap();
        assert!(result.input.views[0].obs.cameras[1].is_some());
        assert!(result.input.views[1].obs.cameras[1].is_none());
        assert_eq!(result.rejected.len(), 1);
        assert_eq!(result.rejected[0].reason, RejectReason::TooFewFeatures);
    }

    /// The test board with only its top `rows` rows in view.
    fn top_rows(rows: u32) -> Vec<Feature> {
        let missing: Vec<(u32, u32)> = (rows..9)
            .flat_map(|v| (0..6).map(move |u| (u, v)))
            .collect();
        board_features_without(&missing)
    }

    /// The test board seen turned by 90°: 9 corners across, 6 down.
    fn turned_board() -> Vec<Feature> {
        (0..6u32)
            .flat_map(|v| (0..9u32).map(move |u| (u, v)))
            .map(|(u, v)| Feature {
                cov_px: None,
                image_xy: [100.0 + 40.0 * f64::from(u), 50.0 + 40.0 * f64::from(v)],
                world_xyz: [0.025 * f64::from(u), 0.025 * f64::from(v), 0.0],
            })
            .collect()
    }

    #[test]
    fn partial_chessboards_are_dropped_and_turned_ones_matched() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        for (view, features) in [("a", board_features()), ("b", board_features())] {
            seed_image(tmp.path(), &cache, &format!("cam0/{view}.png"), features);
        }
        seed_image(tmp.path(), &cache, "cam0/c.png", turned_board());
        seed_image(tmp.path(), &cache, "cam1/a.png", board_features());
        // Bottom two rows out of view: labelled from its own top-left
        // corner, this would still be a 6 × 7 board in the frame.
        seed_image(tmp.path(), &cache, "cam1/b.png", top_rows(7));
        seed_image(tmp.path(), &cache, "cam1/c.png", turned_board());
        let spec = rig_spec(&[
            vec!["cam0/a.png", "cam0/b.png", "cam0/c.png"],
            vec!["cam1/a.png", "cam1/b.png", "cam1/c.png"],
        ]);
        let result = build_rig_extrinsics_input(
            &spec,
            tmp.path(),
            Detection::Cached(&cache),
            &RunControl::default(),
        )
        .unwrap();
        // Every view keeps cam0; cam1 loses its partial board only.
        assert_eq!(result.input.num_views(), 3);
        assert!(result.input.views[0].obs.cameras[1].is_some());
        assert!(result.input.views[1].obs.cameras[1].is_none());
        let rejected: Vec<_> = result
            .rejected
            .iter()
            .map(|r| (r.camera, r.reason, r.path.ends_with("cam1/b.png")))
            .collect();
        assert_eq!(rejected, vec![(1, RejectReason::PartialChessboard, true)]);
        // Both cameras saw the board turned alike: turned onto the declared
        // 6 × 9 extent, with the same labels.
        let turned = &result.input.views[2].obs.cameras;
        let (left, right) = (turned[0].as_ref().unwrap(), turned[1].as_ref().unwrap());
        assert_eq!(left.points_3d, right.points_3d);
        let max = left
            .points_3d
            .iter()
            .fold([0.0f64; 2], |m, p| [m[0].max(p.x), m[1].max(p.y)]);
        assert!((max[0] - 5.0 * 0.025).abs() < 1e-12 && (max[1] - 8.0 * 0.025).abs() < 1e-12);
    }

    #[test]
    fn a_turned_chessboard_is_rejected_where_the_frame_is_fixed() {
        // Rig hand-eye fixes the target through the views; camera 0 sees
        // the board turned by 90° in the second view, so neither camera's
        // detection of it places the frame.
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        for cam in ["cam0", "cam1"] {
            seed_image(
                tmp.path(),
                &cache,
                &format!("{cam}/a.png"),
                board_features(),
            );
            seed_image(tmp.path(), &cache, &format!("{cam}/b.png"), turned_board());
        }
        let mut spec = rig_spec(&[
            vec!["cam0/a.png", "cam0/b.png"],
            vec!["cam1/a.png", "cam1/b.png"],
        ]);
        spec.topology = ProblemKind::RigHandeye;
        let filter = ViewFilter::for_spec(&spec);
        let (grid, scope) = filter.rig_chessboard().unwrap();
        let mut detected: Vec<Vec<Option<Vec<Feature>>>> = vec![
            vec![Some(board_features()), Some(board_features())],
            vec![Some(turned_board()), Some(turned_board())],
        ];
        let rejected = match_rig_labels(&grid, scope, &mut detected);
        assert_eq!(
            rejected,
            vec![
                (1, 0, RejectReason::TurnedChessboard),
                (1, 1, RejectReason::TurnedChessboard),
            ]
        );
        assert!(detected[0].iter().all(Option::is_some));
    }

    #[test]
    fn a_camera_with_only_partial_chessboards_is_a_typed_error() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        seed_image(tmp.path(), &cache, "cam0/a.png", board_features());
        seed_image(tmp.path(), &cache, "cam0/b.png", board_features());
        seed_image(tmp.path(), &cache, "cam1/a.png", top_rows(8));
        seed_image(tmp.path(), &cache, "cam1/b.png", grid_features(3));
        let spec = rig_spec(&[
            vec!["cam0/a.png", "cam0/b.png"],
            vec!["cam1/a.png", "cam1/b.png"],
        ]);
        let err = build_rig_extrinsics_input(
            &spec,
            tmp.path(),
            Detection::Cached(&cache),
            &RunControl::default(),
        )
        .unwrap_err();
        match err {
            RunError::NoWholeChessboardViews {
                camera_id,
                partial,
                turned,
                ambiguous,
                total,
            } => assert_eq!(
                (camera_id.as_str(), partial, turned, ambiguous, total),
                ("cam1", 1, 0, 0, 2)
            ),
            other => panic!("expected NoWholeChessboardViews, got {other:?}"),
        }
    }

    #[test]
    fn self_identifying_targets_keep_partial_views() {
        // The same partial detection under a PuzzleBoard target: its
        // labels are absolute, so the view is kept.
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        let puzzle_config = vision_calibration_detect::DetectorConfig::Puzzleboard(
            vision_calibration_detect::PuzzleboardConfig {
                rows: 9,
                cols: 6,
                cell_size_m: 0.025,
            },
        );
        for (rel, features) in [
            ("cam0/a.png", board_features()),
            ("cam1/a.png", top_rows(5)),
        ] {
            let path = tmp.path().join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, rel.as_bytes()).unwrap();
            let key = CacheKey::for_detection(rel.as_bytes(), &puzzle_config, None);
            cache.put(&key, &CachedFeatures { features }).unwrap();
        }
        let mut spec = rig_spec(&[vec!["cam0/a.png"], vec!["cam1/a.png"]]);
        spec.target = TargetSpec::Puzzleboard {
            layout: "puzzle_9x6".into(),
            cell_size_m: 0.025,
        };
        let result = build_rig_extrinsics_input(
            &spec,
            tmp.path(),
            Detection::Cached(&cache),
            &RunControl::default(),
        )
        .unwrap();
        assert!(result.input.views[0].obs.cameras[1].is_some());
        assert!(result.rejected.is_empty());
    }

    #[test]
    fn camera_with_zero_usable_views_is_an_error() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        seed_image(tmp.path(), &cache, "cam0/a.png", board_features());
        seed_image(tmp.path(), &cache, "cam1/a.png", grid_features(2));
        let spec = rig_spec(&[vec!["cam0/a.png"], vec!["cam1/a.png"]]);
        let err = build_rig_extrinsics_input(
            &spec,
            tmp.path(),
            Detection::Cached(&cache),
            &RunControl::default(),
        )
        .unwrap_err();
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
        let err = build_rig_extrinsics_input(
            &spec,
            tmp.path(),
            Detection::Cached(&cache),
            &RunControl::default(),
        )
        .unwrap_err();
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
        let err = build_rig_handeye_input(
            &spec,
            tmp.path(),
            Detection::Cached(&cache),
            &RunControl::default(),
        )
        .unwrap_err();
        assert!(matches!(err, RunError::UnsupportedTopology { .. }));

        let mut spec2 = rig_spec(&[vec!["a.png"], vec!["b.png"]]);
        spec2.topology = ProblemKind::RigHandeye;
        let err2 = build_rig_extrinsics_input(
            &spec2,
            tmp.path(),
            Detection::Cached(&cache),
            &RunControl::default(),
        )
        .unwrap_err();
        assert!(matches!(err2, RunError::UnsupportedTopology { .. }));
    }

    fn handeye_spec_with_poses(dir: &Path, csv: &str) -> DatasetSpec {
        let pose_path = dir.join("poses.csv");
        std::fs::write(&pose_path, csv).unwrap();
        let mut spec = rig_spec(&[
            vec!["cam0/img_0.png", "cam0/img_1.png", "cam0/img_2.png"],
            vec!["cam1/img_0.png", "cam1/img_1.png", "cam1/img_2.png"],
        ]);
        spec.topology = ProblemKind::RigHandeye;
        spec.robot_poses = Some(RobotPoseSource {
            path: PathBuf::from("poses.csv"),
            format: RobotPoseFormat::Csv,
            columns: Some(PoseColumnMap {
                pose_id: None,
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
                    board_features(),
                );
            }
        }
        let csv = "tx,ty,tz,qx,qy,qz,qw\n\
                   0.1,0,0,0,0,0,1\n\
                   0.2,0,0,0,0,0,1\n\
                   0.3,0,0,0,0,0,1";
        let spec = handeye_spec_with_poses(tmp.path(), csv);
        let result = build_rig_handeye_input(
            &spec,
            tmp.path(),
            Detection::Cached(&cache),
            &RunControl::default(),
        )
        .unwrap();
        assert_eq!(result.input.num_views(), 3);
        let xs: Vec<f64> = result
            .input
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
                    board_features(),
                );
            }
        }
        let csv = "tx,ty,tz,qx,qy,qz,qw\n0.1,0,0,0,0,0,1"; // 1 pose, 3 views
        let spec = handeye_spec_with_poses(tmp.path(), csv);
        let err = build_rig_handeye_input(
            &spec,
            tmp.path(),
            Detection::Cached(&cache),
            &RunControl::default(),
        )
        .unwrap_err();
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
                    board_features(),
                );
            }
        }
        let csv = "tx,ty,tz,qx,qy,qz,qw\n0.1,0,0,0,0,0,1";
        let mut spec = handeye_spec_with_poses(tmp.path(), csv);
        spec.pose_pairing = Some(PosePairing::SharedFilenameToken {
            regex: r"^img_(?<view>\d+)\.png$".into(),
            group: "view".into(),
        });
        let err = build_rig_handeye_input(
            &spec,
            tmp.path(),
            Detection::Cached(&cache),
            &RunControl::default(),
        )
        .unwrap_err();
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
                    board_features(),
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
            rp.columns.as_mut().unwrap().pose_id = Some("view".into());
        }
        let result = build_rig_handeye_input(
            &spec,
            tmp.path(),
            Detection::Cached(&cache),
            &RunControl::default(),
        )
        .unwrap();
        assert_eq!(result.input.num_views(), 2);
        // Views are sorted by token; pose rows were reversed in the file.
        let xs: Vec<f64> = result
            .input
            .views
            .iter()
            .map(|v| v.meta.base_se3_gripper.translation.x)
            .collect();
        assert_eq!(xs, vec![0.1, 0.2]);
    }

    #[test]
    fn token_pairing_rejects_duplicate_pose_id() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        for cam in 0..2 {
            for view in 0..2 {
                seed_image(
                    tmp.path(),
                    &cache,
                    &format!("cam{cam}/img_{view}.png"),
                    board_features(),
                );
            }
        }
        // Two rows share pose_id "0" — an ambiguous table that must be
        // rejected rather than silently keeping the last row.
        let csv = "view,tx,ty,tz,qx,qy,qz,qw\n\
                   0,0.1,0,0,0,0,0,1\n\
                   0,0.9,0,0,0,0,0,1";
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
            rp.columns.as_mut().unwrap().pose_id = Some("view".into());
        }
        let err = build_rig_handeye_input(
            &spec,
            tmp.path(),
            Detection::Cached(&cache),
            &RunControl::default(),
        )
        .unwrap_err();
        match err {
            RunError::PairingTokenMismatch { message } => {
                assert!(
                    message.contains("duplicate pose_id") && message.contains('0'),
                    "got: {message}"
                );
            }
            other => panic!("expected PairingTokenMismatch, got {other:?}"),
        }
    }
}
