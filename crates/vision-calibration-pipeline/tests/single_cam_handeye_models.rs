//! Single-camera hand-eye with the camera model from the config: every
//! distortion kind, and a tilted (Scheimpflug) sensor from a cold start and
//! from a tilt seed.

#![allow(missing_docs)]

use nalgebra::{Isometry3, Rotation3, Translation3};
use vision_calibration_core::{
    BrownConrady5, CameraModelSpec, CameraParams, CorrespondenceView, DistortionKind,
    DistortionParams, FxFyCxCy, Iso3, ProjectionParams, Pt3, Real, ScheimpflugParams, SensorKind,
    SensorParams,
};
use vision_calibration_pipeline::common::config::CameraConfig;
use vision_calibration_pipeline::session::CalibrationSession;
use vision_calibration_pipeline::single_cam_handeye::{
    HandeyeMeta, SingleCamHandeyeConfig, SingleCamHandeyeInput, SingleCamHandeyeProblem,
    SingleCamHandeyeView, SingleCamIntrinsicsManualInit, run_calibration, step_handeye_init,
    step_handeye_optimize, step_intrinsics_init_with_seed, step_intrinsics_optimize,
};

fn gripper_se3_camera() -> Iso3 {
    Isometry3::from_parts(
        Translation3::new(0.05, -0.03, 0.12),
        Rotation3::from_euler_angles(0.1, -0.05, 0.2).into(),
    )
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

/// Noise-free eye-in-hand views of a 9×7 board seen by `camera`.
fn input(camera: &CameraParams) -> SingleCamHandeyeInput {
    let model = camera.build().unwrap();
    let base_se3_target = Isometry3::from_parts(
        Translation3::new(0.4, 0.1, -0.2),
        Rotation3::from_euler_angles(0.0, 0.0, 0.3).into(),
    );
    let views = board_poses()
        .iter()
        .map(|cam_se3_target| {
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
            SingleCamHandeyeView::new(
                CorrespondenceView::new(p3, p2).unwrap(),
                HandeyeMeta {
                    base_se3_gripper: base_se3_target
                        * cam_se3_target.inverse()
                        * gripper_se3_camera().inverse(),
                },
            )
        })
        .collect();
    SingleCamHandeyeInput::new(views).unwrap()
}

fn truth(distortion: DistortionParams, sensor: SensorParams) -> CameraParams {
    CameraParams {
        projection: ProjectionParams::Pinhole,
        distortion,
        sensor,
        intrinsics: FxFyCxCy {
            fx: 800.0,
            fy: 790.0,
            cx: 640.0,
            cy: 360.0,
        },
    }
}

fn brown_conrady() -> DistortionParams {
    DistortionParams::BrownConrady5 {
        params: BrownConrady5 {
            k1: -0.25,
            k2: 0.08,
            ..BrownConrady5::default()
        },
    }
}

fn config(model: CameraModelSpec) -> SingleCamHandeyeConfig {
    let mut config = SingleCamHandeyeConfig::default();
    config.camera = CameraConfig::new(model);
    config.robot_poses.refine = false;
    config
}

fn handeye_error(export_x: Option<Iso3>) -> Real {
    (export_x.unwrap().translation.vector - gripper_se3_camera().translation.vector).norm()
}

#[test]
fn every_distortion_kind_calibrates() {
    let camera = truth(brown_conrady(), SensorParams::Identity);
    let data = input(&camera);
    for kind in [
        DistortionKind::None,
        DistortionKind::BrownConrady5,
        DistortionKind::Rational8,
        DistortionKind::ThinPrism9,
        DistortionKind::Division1,
    ] {
        let mut session = CalibrationSession::<SingleCamHandeyeProblem>::new();
        session.set_input(data.clone()).unwrap();
        session
            .set_config(config(CameraModelSpec::pinhole(kind, SensorKind::Identity)))
            .unwrap();
        run_calibration(&mut session).unwrap_or_else(|e| panic!("{kind:?}: {e}"));
        let export = session.export().unwrap();
        assert_eq!(export.camera.spec().map(|s| s.distortion), Some(kind));
        if kind != DistortionKind::None {
            // Every model with radial terms fits the Brown-Conrady truth.
            assert!(
                export.mean_reproj_error < 0.05,
                "{kind:?}: {} px",
                export.mean_reproj_error
            );
        }
    }
}

#[test]
fn tilted_camera_calibrates_from_a_cold_start() {
    let sensor_gt = ScheimpflugParams {
        tilt_x: 0.04,
        tilt_y: -0.03,
    };
    let camera = truth(
        brown_conrady(),
        SensorParams::Scheimpflug { params: sensor_gt },
    );
    let mut session = CalibrationSession::<SingleCamHandeyeProblem>::new();
    session.set_input(input(&camera)).unwrap();
    session
        .set_config(config(CameraModelSpec::pinhole(
            DistortionKind::BrownConrady5,
            SensorKind::Scheimpflug,
        )))
        .unwrap();
    run_calibration(&mut session).unwrap();
    let export = session.export().unwrap();
    assert!(
        export.mean_reproj_error < 1e-4,
        "{}",
        export.mean_reproj_error
    );
    let SensorParams::Scheimpflug { params } = export.camera.sensor else {
        panic!("tilted camera expected");
    };
    assert!(
        (params.tilt_x - sensor_gt.tilt_x).abs() < 1e-4,
        "{params:?}"
    );
    assert!(
        (params.tilt_y - sensor_gt.tilt_y).abs() < 1e-4,
        "{params:?}"
    );
    assert!(handeye_error(export.gripper_se3_camera) < 1e-4);
}

#[test]
fn tilt_seed_follows_the_camera_model() {
    let sensor_gt = ScheimpflugParams {
        tilt_x: 0.04,
        tilt_y: -0.03,
    };
    let camera = truth(
        brown_conrady(),
        SensorParams::Scheimpflug { params: sensor_gt },
    );
    let data = input(&camera);

    // A tilted camera takes its seed and skips the tilt sweep.
    let mut session = CalibrationSession::<SingleCamHandeyeProblem>::new();
    session.set_input(data.clone()).unwrap();
    session
        .set_config(config(CameraModelSpec::pinhole(
            DistortionKind::BrownConrady5,
            SensorKind::Scheimpflug,
        )))
        .unwrap();
    let mut seed = SingleCamIntrinsicsManualInit::default();
    seed.sensor = Some(sensor_gt);
    step_intrinsics_init_with_seed(&mut session, seed.clone(), None).unwrap();
    step_intrinsics_optimize(&mut session, None).unwrap();
    step_handeye_init(&mut session, None).unwrap();
    step_handeye_optimize(&mut session, None).unwrap();
    assert!(handeye_error(session.export().unwrap().gripper_se3_camera) < 1e-4);

    // An untilted camera takes only a zero tilt.
    let mut session = CalibrationSession::<SingleCamHandeyeProblem>::new();
    session.set_input(data).unwrap();
    let err = step_intrinsics_init_with_seed(&mut session, seed, None)
        .unwrap_err()
        .to_string();
    assert!(err.contains("Scheimpflug"), "{err}");

    // A model changed after the intrinsics stage is caught before a solve.
    let mut seed = SingleCamIntrinsicsManualInit::default();
    seed.sensor = Some(ScheimpflugParams::default());
    step_intrinsics_init_with_seed(&mut session, seed, None).unwrap();
    session
        .set_config(config(CameraModelSpec::pinhole(
            DistortionKind::Division1,
            SensorKind::Identity,
        )))
        .unwrap();
    let err = step_intrinsics_optimize(&mut session, None)
        .unwrap_err()
        .to_string();
    assert!(err.contains("model changed"), "{err}");
}
