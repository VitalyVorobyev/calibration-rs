//! A camera whose fix list holds `p1`, `p2` keeps them at exactly zero
//! through every stage of every workflow that initializes a camera: the
//! linear stage does not fit a term the non-linear solves would then hold at
//! that fitted value. The rig laserline workflows take their cameras from
//! these rig workflows.

#![allow(missing_docs)]

use nalgebra::{Isometry3, Rotation3, Translation3};
use vision_calibration_core::{
    BrownConrady5, CameraParams, CorrespondenceView, DistortionParams, FxFyCxCy, Iso3, NoMeta,
    PlanarDataset, ProjectionParams, Pt3, Real, RigDataset, RigView, RigViewObs, SensorParams,
    View,
};
use vision_calibration_optim::RobotPoseMeta;
use vision_calibration_pipeline::common::config::CameraConfig;
use vision_calibration_pipeline::session::CalibrationSession;
use vision_calibration_pipeline::{
    planar_intrinsics::{PlanarIntrinsicsProblem, step_init, step_optimize},
    rig_extrinsics::RigExtrinsicsProblem,
    rig_handeye::RigHandeyeProblem,
    single_cam_handeye::{
        HandeyeMeta, SingleCamHandeyeInput, SingleCamHandeyeProblem, SingleCamHandeyeView,
    },
};

/// A frontal camera with real tangential distortion, so a linear fit of
/// `p1`, `p2` is far from zero.
fn truth() -> CameraParams {
    CameraParams {
        projection: ProjectionParams::Pinhole,
        distortion: DistortionParams::BrownConrady5 {
            params: BrownConrady5 {
                k1: -0.2,
                k2: 0.05,
                p1: 1.5e-3,
                p2: -1.0e-3,
                ..BrownConrady5::default()
            },
        },
        sensor: SensorParams::Identity,
        intrinsics: FxFyCxCy {
            fx: 800.0,
            fy: 790.0,
            cx: 640.0,
            cy: 360.0,
        },
    }
}

/// The default camera with `p1`, `p2` (and the default `k3`) held.
fn held() -> CameraConfig {
    let mut camera = CameraConfig::default();
    camera.fix.distortion = Some(vec!["k3".into(), "p1".into(), "p2".into()]);
    camera
}

fn assert_held(camera: &CameraParams, stage: &str) {
    let DistortionParams::BrownConrady5 { params } = camera.distortion else {
        panic!("{stage}: Brown-Conrady camera expected");
    };
    assert_eq!((params.p1, params.p2), (0.0, 0.0), "{stage}");
}

/// Twelve oblique board poses (camera frame), the board centred ~0.6 m ahead.
fn board_poses() -> Vec<Iso3> {
    [
        (0.0, 0.0),
        (0.25, 0.0),
        (-0.25, 0.0),
        (0.0, 0.25),
        (0.0, -0.25),
        (0.18, 0.18),
        (-0.18, 0.18),
        (0.18, -0.18),
        (-0.18, -0.18),
        (0.1, -0.05),
        (-0.12, 0.08),
        (0.05, 0.15),
    ]
    .iter()
    .enumerate()
    .map(|(i, &(rx, ry))| {
        Isometry3::from_parts(
            Translation3::new(-0.12, -0.09, 0.6 + 0.03 * (i % 3) as Real),
            Rotation3::from_euler_angles(rx, ry, 0.05 * i as Real).into(),
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

#[test]
fn planar_intrinsics() {
    let views = board_poses()
        .iter()
        .map(|p| View::without_meta(view(p)))
        .collect();
    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    session
        .set_input(PlanarDataset::new(views).unwrap())
        .unwrap();
    let mut config = session.config.clone();
    config.camera = held();
    session.set_config(config).unwrap();
    let init = step_init(&mut session, None).unwrap();
    assert_eq!((init.distortion.p1, init.distortion.p2), (0.0, 0.0), "init");
    step_optimize(&mut session, None).unwrap();
    assert_held(&session.export().unwrap().params.camera, "optimize");
}

#[test]
fn single_cam_handeye() {
    let views = board_poses()
        .iter()
        .map(|p| {
            SingleCamHandeyeView::new(
                view(p),
                HandeyeMeta {
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
    config.camera = held();
    config.robot_poses.refine = false;
    session.set_config(config).unwrap();
    let init =
        vision_calibration_pipeline::single_cam_handeye::step_intrinsics_init(&mut session, None)
            .unwrap();
    assert_held(&init.camera, "init");
    vision_calibration_pipeline::single_cam_handeye::run_calibration(&mut session).unwrap();
    assert_held(&session.export().unwrap().camera, "export");
}

#[test]
fn rig_extrinsics() {
    let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
    session.set_input(rig_views(|_| NoMeta)).unwrap();
    let mut config = session.config.clone();
    config.camera = held();
    session.set_config(config).unwrap();
    let init =
        vision_calibration_pipeline::rig_extrinsics::step_intrinsics_init_all(&mut session, None)
            .unwrap();
    for camera in &init.cameras {
        assert_held(camera, "init");
    }
    vision_calibration_pipeline::rig_extrinsics::run_calibration(&mut session).unwrap();
    for camera in &session.export().unwrap().cameras {
        assert_held(camera, "export");
    }
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
    config.camera = held();
    config.handeye_optimize.robot_poses.refine = false;
    session.set_config(config).unwrap();
    let init =
        vision_calibration_pipeline::rig_handeye::step_intrinsics_init_all(&mut session, None)
            .unwrap();
    for camera in &init.cameras {
        assert_held(camera, "init");
    }
    vision_calibration_pipeline::rig_handeye::run_calibration(&mut session).unwrap();
    for camera in &session.export().unwrap().cameras {
        assert_held(camera, "export");
    }
}
