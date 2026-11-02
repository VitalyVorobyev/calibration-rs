//! A Kannala-Brandt fisheye camera (equidistant projection, `KannalaBrandt4`
//! distortion) calibrates from scratch in every workflow that initializes a
//! camera, with boards across a wide field, some beyond 90° off-axis. The
//! rig laserline workflows take their cameras from these rig workflows.

#![allow(missing_docs)]

use nalgebra::{Isometry3, Rotation3, Translation3};
use vision_calibration_core::{
    CameraModelSpec, CameraParams, CorrespondenceView, DistortionKind, DistortionParams, FxFyCxCy,
    Iso3, NoMeta, PlanarDataset, ProjectionKind, ProjectionParams, Pt3, Real, RigDataset, RigView,
    RigViewObs, SensorKind, SensorParams, View,
};
use vision_calibration_optim::RobotPoseMeta;
use vision_calibration_pipeline::common::config::CameraConfig;
use vision_calibration_pipeline::session::CalibrationSession;
use vision_calibration_pipeline::{
    planar_intrinsics::{IntrinsicsManualInit, PlanarIntrinsicsProblem, step_init_with_seed},
    rig_extrinsics::RigExtrinsicsProblem,
    rig_handeye::RigHandeyeProblem,
    single_cam_handeye::{SingleCamHandeyeInput, SingleCamHandeyeProblem, SingleCamHandeyeView},
};

fn model() -> CameraModelSpec {
    CameraModelSpec::new(
        ProjectionKind::Equidistant,
        DistortionKind::KannalaBrandt4,
        SensorKind::Identity,
    )
}

fn truth() -> CameraParams {
    CameraParams {
        projection: ProjectionParams::Equidistant,
        distortion: DistortionParams::from_values(
            DistortionKind::KannalaBrandt4,
            &[-0.02, 0.002, -0.0001, 0.0],
        )
        .unwrap(),
        sensor: SensorParams::Identity,
        intrinsics: FxFyCxCy {
            fx: 300.0,
            fy: 302.0,
            cx: 640.0,
            cy: 400.0,
        },
    }
}

fn assert_recovered(camera: &CameraParams, stage: &str) {
    assert!(
        matches!(camera.projection, ProjectionParams::Equidistant),
        "{stage}"
    );
    let (k, gt) = (camera.intrinsics, truth().intrinsics);
    for (est, want) in [(k.fx, gt.fx), (k.fy, gt.fy), (k.cx, gt.cx), (k.cy, gt.cy)] {
        assert!((est - want).abs() < 1e-3, "{stage}: {k:?}");
    }
    let d = camera.distortion.values();
    for (est, want) in d.iter().zip(truth().distortion.values()) {
        assert!((est - want).abs() < 1e-4, "{stage}: {d:?}");
    }
}

/// Board poses (camera frame) across a wide field: near and oblique, the
/// last few with part of the board beyond 90° off-axis.
fn board_poses() -> Vec<Iso3> {
    [
        ([0.0, 0.0, 0.0], [-0.12, -0.09, 0.3]),
        ([0.4, 0.0, 0.1], [-0.3, -0.1, 0.25]),
        ([-0.4, 0.0, -0.1], [0.05, -0.1, 0.25]),
        ([0.0, 0.5, 0.2], [-0.1, -0.3, 0.2]),
        ([0.0, -0.5, -0.2], [-0.1, 0.1, 0.2]),
        ([0.3, 0.3, 0.3], [-0.25, -0.25, 0.3]),
        ([-0.3, 0.3, 0.0], [0.1, -0.2, 0.3]),
        ([0.2, -0.3, 0.4], [-0.2, 0.05, 0.35]),
        ([0.0, 1.4, 0.0], [0.25, -0.09, 0.05]),
        ([0.0, -1.4, 0.3], [-0.25, -0.09, 0.0]),
        ([1.3, 0.0, 0.0], [-0.12, 0.2, 0.05]),
        ([-1.3, 0.2, 0.0], [-0.12, -0.25, 0.0]),
    ]
    .iter()
    .map(|(r, t)| {
        Isometry3::from_parts(
            Translation3::new(t[0], t[1], t[2]),
            Rotation3::from_euler_angles(r[0], r[1], r[2]).into(),
        )
    })
    .collect()
}

/// Noise-free view of a 9×7 board at `cam_se3_target`.
fn view(cam_se3_target: &Iso3) -> CorrespondenceView {
    let model = truth().build().unwrap();
    let (mut p3, mut p2) = (Vec::new(), Vec::new());
    for y in 0..7 {
        for x in 0..9 {
            let pw = Pt3::new(x as Real * 0.03, y as Real * 0.03, 0.0);
            if let Some(px) = model.project_point(&cam_se3_target.transform_point(&pw)) {
                p3.push(pw);
                p2.push(px);
            }
        }
    }
    CorrespondenceView::new(p3, p2).unwrap()
}

fn gripper_se3_camera() -> Iso3 {
    Isometry3::from_parts(
        Translation3::new(0.05, -0.03, 0.12),
        Rotation3::from_euler_angles(0.1, -0.05, 0.2).into(),
    )
}

/// The robot pose that puts the camera at `cam_se3_target`.
fn base_se3_gripper(cam_se3_target: &Iso3) -> Iso3 {
    let base_se3_target = Isometry3::from_parts(
        Translation3::new(0.4, 0.1, -0.2),
        Rotation3::from_euler_angles(0.0, 0.0, 0.3).into(),
    );
    base_se3_target * cam_se3_target.inverse() * gripper_se3_camera().inverse()
}

/// Two identical cameras 2 cm apart, both seeing the board in every view.
fn rig_views<M: Clone>(meta: impl Fn(&Iso3) -> M) -> RigDataset<M> {
    let cam1_se3_cam0 = Isometry3::translation(-0.02, 0.0, 0.0);
    let views = board_poses()
        .iter()
        .map(|cam0_se3_target| RigView {
            obs: RigViewObs {
                cameras: vec![
                    Some(view(cam0_se3_target)),
                    Some(view(&(cam1_se3_cam0 * cam0_se3_target))),
                ],
            },
            meta: meta(cam0_se3_target),
        })
        .collect();
    RigDataset::new(views, 2).unwrap()
}

fn planar_session() -> CalibrationSession<PlanarIntrinsicsProblem> {
    let views = board_poses()
        .iter()
        .map(|p| View::without_meta(view(p)))
        .collect();
    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    session
        .set_input(PlanarDataset::new(views).unwrap())
        .unwrap();
    let mut config = session.config.clone();
    config.camera = CameraConfig::new(model());
    session.set_config(config).unwrap();
    session
}

#[test]
fn the_scene_reaches_beyond_ninety_degrees() {
    let beyond = board_poses().iter().any(|pose| {
        (0..63).any(|i| {
            let pw = Pt3::new((i % 9) as Real * 0.03, (i / 9) as Real * 0.03, 0.0);
            pose.transform_point(&pw).z < 0.0
        })
    });
    assert!(beyond);
}

#[test]
fn planar_intrinsics() {
    let mut session = planar_session();
    let init =
        vision_calibration_pipeline::planar_intrinsics::step_init(&mut session, None).unwrap();
    let k = init.camera.intrinsics;
    assert!((k.fx / 300.0 - 1.0).abs() < 0.02, "init {k:?}");
    vision_calibration_pipeline::planar_intrinsics::step_optimize(&mut session, None).unwrap();
    assert_recovered(&session.export().unwrap().params.camera, "optimize");
}

#[test]
fn planar_intrinsics_from_seeded_intrinsics() {
    let mut session = planar_session();
    let mut seed = IntrinsicsManualInit::default();
    seed.intrinsics = Some(FxFyCxCy {
        fx: 290.0,
        fy: 290.0,
        cx: 630.0,
        cy: 410.0,
    });
    step_init_with_seed(&mut session, seed, None).unwrap();
    vision_calibration_pipeline::planar_intrinsics::step_optimize(&mut session, None).unwrap();
    assert_recovered(&session.export().unwrap().params.camera, "optimize");
}

#[test]
fn planar_intrinsics_from_seeded_intrinsics_and_distortion() {
    let mut session = planar_session();
    let mut seed = IntrinsicsManualInit::default();
    seed.intrinsics = Some(FxFyCxCy {
        fx: 295.0,
        fy: 305.0,
        cx: 635.0,
        cy: 405.0,
    });
    let distortion =
        DistortionParams::from_values(DistortionKind::KannalaBrandt4, &[-0.015, 0.001, 0.0, 0.0])
            .unwrap();
    seed.distortion = Some(distortion.clone());
    let init = step_init_with_seed(&mut session, seed, None).unwrap();
    assert_eq!(
        init.camera.distortion.values(),
        distortion.values(),
        "the seed is used as given"
    );
    vision_calibration_pipeline::planar_intrinsics::step_optimize(&mut session, None).unwrap();
    assert_recovered(&session.export().unwrap().params.camera, "optimize");
}

#[test]
fn a_distortion_seed_of_another_kind_is_rejected() {
    let brown_conrady =
        DistortionParams::from_values(DistortionKind::BrownConrady5, &[0.0; 5]).unwrap();

    let mut session = planar_session();
    let mut seed = IntrinsicsManualInit::default();
    seed.distortion = Some(brown_conrady.clone());
    let err = step_init_with_seed(&mut session, seed, None)
        .unwrap_err()
        .to_string();
    assert!(err.contains("distortion"), "{err}");
    assert!(err.contains("BrownConrady5"), "{err}");

    let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
    session.set_input(rig_views(|_| NoMeta)).unwrap();
    let mut config = session.config.clone();
    config.camera = CameraConfig::new(model());
    session.set_config(config).unwrap();
    let mut seed = vision_calibration_pipeline::rig_extrinsics::IntrinsicsManualInit::default();
    seed.per_cam_distortion = Some(vec![brown_conrady; 2]);
    let err = vision_calibration_pipeline::rig_extrinsics::step_intrinsics_init_all_with_seed(
        &mut session,
        seed,
        None,
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("per_cam_distortion[0]"), "{err}");
}

#[test]
fn planar_intrinsics_needs_a_new_init_after_a_projection_change() {
    let mut session = planar_session();
    vision_calibration_pipeline::planar_intrinsics::step_init(&mut session, None).unwrap();
    let mut config = session.config.clone();
    config.camera = CameraConfig::default();
    session.set_config(config).unwrap();
    let err = vision_calibration_pipeline::planar_intrinsics::step_optimize(&mut session, None)
        .unwrap_err()
        .to_string();
    assert!(err.contains("run step_init again"), "{err}");
}

#[test]
fn single_cam_handeye() {
    let views = board_poses()
        .iter()
        .map(|p| {
            SingleCamHandeyeView::new(
                view(p),
                RobotPoseMeta {
                    base_se3_gripper: base_se3_gripper(p),
                },
            )
        })
        .collect();
    let mut session = CalibrationSession::<SingleCamHandeyeProblem>::new();
    session
        .set_input(SingleCamHandeyeInput::new(views).unwrap())
        .unwrap();
    let mut config = session.config.clone();
    config.camera = CameraConfig::new(model());
    config.robot_poses.refine = false;
    session.set_config(config).unwrap();
    vision_calibration_pipeline::single_cam_handeye::run_calibration(&mut session).unwrap();
    let export = session.export().unwrap();
    assert_recovered(&export.camera, "export");
    let err = (export.gripper_se3_camera.unwrap().translation.vector
        - gripper_se3_camera().translation.vector)
        .norm();
    assert!(err < 1e-6, "hand-eye translation error {err}");
}

#[test]
fn rig_extrinsics() {
    let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
    session.set_input(rig_views(|_| NoMeta)).unwrap();
    let mut config = session.config.clone();
    config.camera = CameraConfig::new(model());
    session.set_config(config).unwrap();
    vision_calibration_pipeline::rig_extrinsics::run_calibration(&mut session).unwrap();
    let export = session.export().unwrap();
    for camera in &export.cameras {
        assert_recovered(camera, "export");
    }
    let baseline =
        export.cam_se3_rig[1].translation.vector - export.cam_se3_rig[0].translation.vector;
    assert!((baseline.norm() - 0.02).abs() < 1e-6, "{baseline:?}");
}

#[test]
fn rig_handeye() {
    let mut session = CalibrationSession::<RigHandeyeProblem>::new();
    session
        .set_input(rig_views(|p| RobotPoseMeta {
            base_se3_gripper: base_se3_gripper(p),
        }))
        .unwrap();
    let mut config = session.config.clone();
    config.camera = CameraConfig::new(model());
    config.handeye_optimize.robot_poses.refine = false;
    session.set_config(config).unwrap();
    vision_calibration_pipeline::rig_handeye::run_calibration(&mut session).unwrap();
    for camera in &session.export().unwrap().cameras {
        assert_recovered(camera, "export");
    }
}

/// With noisy corners the calibration has a covariance: its projection
/// uncertainty reaches rays beyond 90°, and next-view planning proposes and
/// ranks candidates across the whole fisheye image.
#[test]
fn wide_angle_analysis() {
    use vision_calibration_pipeline::analysis::{CandidateGrid, candidate_poses, rank_candidates};
    use vision_calibration_pipeline::common::CalibratedCameras;

    let views = board_poses()
        .iter()
        .enumerate()
        .map(|(v, p)| {
            let mut obs = view(p);
            for (i, px) in obs.points_2d.iter_mut().enumerate() {
                // Deterministic ±0.1 px pattern.
                px.x += 0.1 * ((7 * i + 3 * v) % 5) as Real / 2.0 - 0.1;
                px.y += 0.1 * ((5 * i + 11 * v) % 3) as Real - 0.1;
            }
            View::without_meta(obs)
        })
        .collect();
    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    session
        .set_input(PlanarDataset::new(views).unwrap())
        .unwrap();
    let mut config = session.config.clone();
    config.camera = CameraConfig::new(model());
    session.set_config(config).unwrap();
    vision_calibration_pipeline::planar_intrinsics::run_calibration(&mut session).unwrap();
    let export = session.export().unwrap();

    let image_size = [1280, 800];
    let map = export.projection_uncertainty(0, image_size).unwrap();
    // A node on the left edge, 640 px from the centre: about 130° off-axis.
    let row = map.rows / 2;
    let edge = map.sigma[row * map.columns].expect("a ray reaches the image edge");
    assert!(
        edge.is_finite() && edge > map.center,
        "edge {edge} centre {}",
        map.center
    );

    let board: Vec<Pt3> = (0..63)
        .map(|i| Pt3::new((i % 9) as Real * 0.03, (i / 9) as Real * 0.03, 0.0))
        .collect();
    let grid = CandidateGrid::new(0.3, 0.3, [4, 3]);
    let candidates = candidate_poses(&export.params.camera, &board, image_size, &grid).unwrap();
    assert_eq!(candidates.len(), 4 * 3 * 5);
    let ranked = rank_candidates(&export, &candidates, &board, image_size).unwrap();
    assert!(!ranked.is_empty());
}
