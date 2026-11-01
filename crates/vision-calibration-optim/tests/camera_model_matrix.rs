//! Every builder with every camera model: each distortion kind with an
//! untilted and a tilted (Scheimpflug) sensor, on noise-free synthetic data
//! from a perturbed start. Each solve must reach the noise floor, keep the
//! camera's model, and recover the quantity it calibrates.
//!
//! The rig extrinsics builder has its own matrix in `rig_extrinsics.rs`.

use nalgebra::{Isometry3, Rotation3, Translation3, Vector3};
use vision_calibration_core::{
    CameraModel, CameraParams, CorrespondenceView, DistortionKind, DistortionParams, FxFyCxCy,
    Iso3, Pt2, Pt3, Real, RigView, RigViewObs, SensorKind, SensorParams, View,
};
use vision_calibration_optim::{
    BackendSolveOptions, HandeyeDataset, HandeyeMode, HandeyeParams, HandeyeSolveOptions,
    LaserPlane, LaserlineMeta, LaserlineParams, LaserlineResidualType, LaserlineSolveOptions,
    RigHandeyeLaserlineDataset, RigHandeyeLaserlineParams, RigHandeyeLaserlineSolveOptions,
    RigLaserlineDataset, RigLaserlineSolveOptions, RigLaserlineUpstream, RigLaserlineView,
    RobotPoseMeta, optimize_handeye, optimize_laserline, optimize_rig_handeye_laserline,
    optimize_rig_laserline,
};

/// Distortion of each kind with small, physically plausible coefficients.
fn distortion_of(kind: DistortionKind) -> DistortionParams {
    let values: Vec<f64> = kind
        .param_names()
        .iter()
        .map(|name| match *name {
            "k1" => -0.12,
            "k2" => 0.03,
            "p1" => 0.0004,
            "p2" => -0.0003,
            "lambda" => -0.1,
            _ => 0.0,
        })
        .collect();
    DistortionParams::from_values(kind, &values).expect("values match the kind's dimension")
}

/// A sensor of every kind, tilted by a few degrees where it can be.
fn sensors() -> impl Iterator<Item = SensorParams> {
    SensorKind::ALL.into_iter().map(|kind| {
        let values: Vec<f64> = kind
            .param_names()
            .iter()
            .map(|name| match *name {
                "tilt_x" => 0.05,
                "tilt_y" => -0.03,
                _ => 0.0,
            })
            .collect();
        SensorParams::from_values(kind, &values).expect("values match the kind's dimension")
    })
}

fn truth_k() -> FxFyCxCy<Real> {
    FxFyCxCy {
        fx: 800.0,
        fy: 790.0,
        cx: 640.0,
        cy: 360.0,
    }
}

/// A camera of `kind` with `sensor` and intrinsics `k`.
fn camera(kind: DistortionKind, sensor: &SensorParams, k: FxFyCxCy<Real>) -> CameraParams {
    CameraParams {
        projection: vision_calibration_core::ProjectionParams::Pinhole,
        distortion: distortion_of(kind),
        sensor: sensor.clone(),
        intrinsics: k,
    }
}

/// The start of every solve: intrinsics off by about 1 %.
fn perturbed_k() -> FxFyCxCy<Real> {
    FxFyCxCy {
        fx: 808.0,
        fy: 784.0,
        cx: 644.0,
        cy: 357.0,
    }
}

/// Eight oblique board poses in the camera frame, the board centred ~0.6 m
/// ahead.
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

/// The board's corners, in the target frame (z = 0).
fn board_points() -> Vec<Pt3> {
    (0..7)
        .flat_map(|y| (0..9).map(move |x| Pt3::new(x as Real * 0.03, y as Real * 0.03, 0.0)))
        .collect()
}

/// The board observed by `model` at `cam_se3_target`.
fn observe(model: &CameraModel, cam_se3_target: &Iso3) -> CorrespondenceView {
    let (mut p3, mut p2) = (Vec::new(), Vec::new());
    for pw in board_points() {
        if let Some(px) = model.project_point(&cam_se3_target.transform_point(&pw)) {
            p3.push(pw);
            p2.push(px);
        }
    }
    CorrespondenceView::new(p3, p2).unwrap()
}

/// The laser plane (camera frame) of the matrix: crossing the board centre of
/// the first view.
fn laser_plane() -> LaserPlane {
    let normal = Vector3::new(0.1, 0.9, 0.3).normalize();
    let centre = board_poses()[0].transform_point(&Pt3::new(0.12, 0.09, 0.0));
    LaserPlane::new(normal, -normal.dot(&centre.coords))
}

/// Laser pixels of `plane` (camera frame) on the board at `cam_se3_target`,
/// projected by `model`.
fn laser_pixels(model: &CameraModel, cam_se3_target: &Iso3, plane: &LaserPlane) -> Vec<Pt2> {
    // The plane in target coordinates, `n_t · p + d_t = 0`.
    let n_t = cam_se3_target.rotation.inverse() * plane.normal.into_inner();
    let d_t = plane.distance + plane.normal.dot(&cam_se3_target.translation.vector);
    (0..41)
        .filter_map(|i| {
            let x = i as Real * 0.006;
            let y = -(n_t.x * x + d_t) / n_t.y;
            if !(0.0..=0.18).contains(&y) {
                return None;
            }
            model.project_point(&cam_se3_target.transform_point(&Pt3::new(x, y, 0.0)))
        })
        .collect()
}

fn tight() -> BackendSolveOptions {
    BackendSolveOptions {
        max_iters: 200,
        min_abs_decrease: Some(1e-16),
        min_rel_decrease: Some(1e-16),
        min_error: Some(1e-20),
        ..Default::default()
    }
}

fn plane_errors(a: &LaserPlane, b: &LaserPlane) -> (Real, Real) {
    let angle = a.normal.dot(&b.normal).abs().min(1.0).acos();
    (angle, (a.distance - b.distance).abs())
}

/// A tilt with zero distortion is a principal-point shift and a pose
/// rotation, so only the reprojection floor is checked there.
fn identifiable(kind: DistortionKind, sensor: &SensorParams) -> bool {
    kind != DistortionKind::None || matches!(sensor, SensorParams::Identity)
}

#[test]
fn handeye_solves_every_camera_model() {
    let gripper_se3_camera = Isometry3::from_parts(
        Translation3::new(0.05, -0.03, 0.12),
        Rotation3::from_euler_angles(0.1, -0.05, 0.2).into(),
    );
    let base_se3_target = Isometry3::from_parts(
        Translation3::new(0.4, 0.1, -0.2),
        Rotation3::from_euler_angles(0.0, 0.0, 0.3).into(),
    );
    for kind in DistortionKind::ALL {
        for sensor in sensors() {
            let truth = camera(kind, &sensor, truth_k());
            let model = truth.build().unwrap();
            let views: Vec<RigView<RobotPoseMeta>> = board_poses()
                .iter()
                .map(|cam_se3_target| RigView {
                    obs: RigViewObs {
                        cameras: vec![Some(observe(&model, cam_se3_target))],
                    },
                    meta: RobotPoseMeta {
                        base_se3_gripper: base_se3_target
                            * cam_se3_target.inverse()
                            * gripper_se3_camera.inverse(),
                    },
                })
                .collect();
            let dataset = HandeyeDataset::new(views, 1, HandeyeMode::EyeInHand).unwrap();
            let initial = HandeyeParams {
                cameras: vec![camera(kind, &sensor, perturbed_k())],
                cam_to_rig: vec![Iso3::identity()],
                handeye: Isometry3::from_parts(
                    Translation3::new(0.052, -0.028, 0.118),
                    Rotation3::from_euler_angles(0.104, -0.048, 0.197).into(),
                ),
                target_poses: vec![base_se3_target],
            };
            let opts = HandeyeSolveOptions {
                fix_extrinsics: vec![true],
                ..Default::default()
            };
            let result = optimize_handeye(dataset, initial, opts, tight())
                .unwrap_or_else(|e| panic!("{kind:?}/{sensor:?}: {e}"));
            assert!(
                result.mean_reproj_error < 1e-6,
                "{kind:?}/{sensor:?}: reprojection {} px",
                result.mean_reproj_error
            );
            assert_eq!(result.params.cameras[0].spec(), truth.spec());
            if identifiable(kind, &sensor) {
                let dt = (result.params.handeye.translation.vector
                    - gripper_se3_camera.translation.vector)
                    .norm();
                assert!(dt < 1e-4, "{kind:?}/{sensor:?}: hand-eye translation {dt}");
            }
        }
    }
}

#[test]
fn laserline_solves_every_camera_model() {
    let plane = laser_plane();
    for kind in DistortionKind::ALL {
        for sensor in sensors() {
            let truth = camera(kind, &sensor, truth_k());
            let model = truth.build().unwrap();
            let poses = board_poses();
            let dataset: Vec<_> = poses
                .iter()
                .map(|pose| {
                    View::new(
                        observe(&model, pose),
                        LaserlineMeta {
                            laser_pixels: laser_pixels(&model, pose, &plane),
                            laser_weights: Vec::new(),
                        },
                    )
                })
                .collect();
            for residual in [
                LaserlineResidualType::PointToPlane,
                LaserlineResidualType::LineDistNormalized,
            ] {
                let initial = LaserlineParams::new(
                    camera(kind, &sensor, perturbed_k()),
                    poses.clone(),
                    LaserPlane::new(
                        plane.normal.into_inner() + Vector3::new(0.02, -0.01, 0.01),
                        plane.distance + 0.005,
                    ),
                )
                .unwrap();
                let opts = LaserlineSolveOptions {
                    laser_residual_type: residual,
                    ..Default::default()
                };
                let label = format!("{kind:?}/{sensor:?}/{residual:?}");
                let result = optimize_laserline(&dataset, &initial, &opts, &tight())
                    .unwrap_or_else(|e| panic!("{label}: {e}"));
                assert_eq!(result.params.camera.spec(), truth.spec(), "{label}");
                assert!(
                    result.report.final_cost < 1e-10,
                    "{label}: final cost {}",
                    result.report.final_cost
                );
                if identifiable(kind, &sensor) {
                    let (angle, dd) = plane_errors(&result.params.plane, &plane);
                    assert!(
                        angle < 1e-5 && dd < 1e-6,
                        "{label}: plane {angle} rad, {dd} m"
                    );
                }
            }
        }
    }
}

/// Two cameras: camera 0 untilted with Brown-Conrady distortion, camera 1 of
/// the model under test.
struct RigScene {
    cameras: [CameraParams; 2],
    cam_to_rig: [Iso3; 2],
    rig_se3_target: Vec<Iso3>,
    planes_cam: [LaserPlane; 2],
    views: Vec<RigLaserlineView>,
}

fn rig_scene(kind: DistortionKind, sensor: &SensorParams) -> RigScene {
    let cameras = [
        camera(
            DistortionKind::BrownConrady5,
            &SensorParams::Identity,
            truth_k(),
        ),
        camera(kind, sensor, truth_k()),
    ];
    let models = [cameras[0].build().unwrap(), cameras[1].build().unwrap()];
    let cam_to_rig = [
        Iso3::identity(),
        Isometry3::from_parts(
            Translation3::new(0.08, 0.0, 0.0),
            Rotation3::from_euler_angles(0.0, -0.04, 0.0).into(),
        ),
    ];
    let rig_se3_target = board_poses();
    let plane_rig = laser_plane();
    let planes_cam = [
        plane_rig.transform_by(&cam_to_rig[0].inverse()),
        plane_rig.transform_by(&cam_to_rig[1].inverse()),
    ];
    let views = rig_se3_target
        .iter()
        .map(|rig_pose| {
            let per_cam: Vec<_> = (0..2)
                .map(|c| {
                    let cam_se3_target = cam_to_rig[c].inverse() * rig_pose;
                    (
                        Some(observe(&models[c], &cam_se3_target)),
                        Some(laser_pixels(&models[c], &cam_se3_target, &planes_cam[c])),
                    )
                })
                .collect();
            RigLaserlineView {
                cameras: per_cam.iter().map(|(obs, _)| obs.clone()).collect(),
                laser_pixels: per_cam.into_iter().map(|(_, px)| px).collect(),
            }
        })
        .collect();
    RigScene {
        cameras,
        cam_to_rig,
        rig_se3_target,
        planes_cam,
        views,
    }
}

fn perturbed_plane(plane: &LaserPlane) -> LaserPlane {
    LaserPlane::new(
        plane.normal.into_inner() + Vector3::new(0.02, -0.01, 0.01),
        plane.distance + 0.005,
    )
}

#[test]
fn rig_laserline_solves_every_camera_model() {
    for kind in DistortionKind::ALL {
        for sensor in sensors() {
            let scene = rig_scene(kind, &sensor);
            let dataset = RigLaserlineDataset::new(scene.views.clone(), 2).unwrap();
            let upstream = RigLaserlineUpstream {
                cameras: scene.cameras.to_vec(),
                cam_to_rig: scene.cam_to_rig.to_vec(),
                rig_se3_target: scene.rig_se3_target.clone(),
            };
            let initial: Vec<_> = scene.planes_cam.iter().map(perturbed_plane).collect();
            let result = optimize_rig_laserline(
                &dataset,
                &upstream,
                &initial,
                &RigLaserlineSolveOptions::default(),
                &tight(),
            )
            .unwrap_or_else(|e| panic!("{kind:?}/{sensor:?}: {e}"));
            for (c, (got, gt)) in result
                .laser_planes_cam
                .iter()
                .zip(&scene.planes_cam)
                .enumerate()
            {
                let (angle, dd) = plane_errors(got, gt);
                assert!(
                    angle < 1e-5 && dd < 1e-5,
                    "{kind:?}/{sensor:?}: camera {c} plane {angle} rad, {dd} m"
                );
            }
        }
    }
}

#[test]
fn joint_rig_laserline_solves_every_camera_model() {
    let gripper_se3_rig = Isometry3::from_parts(
        Translation3::new(0.05, -0.03, 0.12),
        Rotation3::from_euler_angles(0.1, -0.05, 0.2).into(),
    );
    let base_se3_target = Isometry3::from_parts(
        Translation3::new(0.4, 0.1, -0.2),
        Rotation3::from_euler_angles(0.0, 0.0, 0.3).into(),
    );
    for kind in DistortionKind::ALL {
        for sensor in sensors() {
            let scene = rig_scene(kind, &sensor);
            let robot_poses: Vec<Iso3> = scene
                .rig_se3_target
                .iter()
                .map(|rig_se3_target| {
                    base_se3_target * rig_se3_target.inverse() * gripper_se3_rig.inverse()
                })
                .collect();
            let dataset = RigHandeyeLaserlineDataset::from_rig_dataset(
                RigLaserlineDataset::new(scene.views.clone(), 2).unwrap(),
                robot_poses,
                HandeyeMode::EyeInHand,
            )
            .unwrap();
            let initial = RigHandeyeLaserlineParams {
                cameras: vec![
                    scene.cameras[0].clone(),
                    camera(kind, &sensor, perturbed_k()),
                ],
                cam_to_rig: scene.cam_to_rig.to_vec(),
                handeye: Isometry3::from_parts(
                    Translation3::new(0.052, -0.028, 0.118),
                    Rotation3::from_euler_angles(0.104, -0.048, 0.197).into(),
                ),
                target_ref: base_se3_target,
                planes_cam: scene.planes_cam.iter().map(perturbed_plane).collect(),
            };
            let opts = RigHandeyeLaserlineSolveOptions {
                fix_extrinsics: vec![true, true],
                fix_planes: vec![false, false],
                ..Default::default()
            };
            let result = optimize_rig_handeye_laserline(dataset, initial, opts, tight())
                .unwrap_or_else(|e| panic!("{kind:?}/{sensor:?}: {e}"));
            assert!(
                result.mean_reproj_error_px < 1e-6,
                "{kind:?}/{sensor:?}: reprojection {} px",
                result.mean_reproj_error_px
            );
            assert_eq!(result.params.cameras[1].spec(), scene.cameras[1].spec());
            if identifiable(kind, &sensor) {
                let (angle, dd) = plane_errors(&result.params.planes_cam[1], &scene.planes_cam[1]);
                assert!(
                    angle < 1e-5 && dd < 1e-6,
                    "{kind:?}/{sensor:?}: camera 1 plane {angle} rad, {dd} m"
                );
            }
        }
    }
}
