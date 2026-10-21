//! Integration test for hand-eye calibration of a camera with a Scheimpflug
//! (tilted) sensor.

use nalgebra::{Isometry3, Rotation3, Translation3};
use vision_calibration_core::{
    BrownConrady5, Camera, CameraFix, CameraParams, CorrespondenceView, DistortionParams, FxFyCxCy,
    Pinhole, ProjectionParams, Pt3, Real, RigView, RigViewObs, ScheimpflugParams, SensorParams,
};
use vision_calibration_optim::{
    BackendSolveOptions, HandEyeDataset, HandEyeMode, HandEyeParams, HandEyeSolveOptions,
    RobotPoseMeta, optimize_handeye,
};

#[test]
fn eye_in_hand_scheimpflug_converges() {
    let intrinsics_gt = FxFyCxCy {
        fx: 800.0,
        fy: 780.0,
        cx: 640.0,
        cy: 360.0,
    };
    let distortion_gt = BrownConrady5 {
        k1: -0.22,
        k2: 0.05,
        k3: 0.0,
        p1: 0.0005,
        p2: -0.0005,
        iters: 5,
    };
    let sensor_gt = ScheimpflugParams {
        tilt_x: 0.06,
        tilt_y: -0.04,
    };

    let camera_gt = Camera::new(Pinhole, distortion_gt, sensor_gt.compile(), intrinsics_gt);

    let cam_to_rig_gt = Isometry3::identity();
    let handeye_gt = Isometry3::from_parts(
        Translation3::new(0.05, 0.03, 0.15),
        Rotation3::from_euler_angles(0.1, -0.05, 0.15).into(),
    );

    let robot_poses_gt = [
        Isometry3::from_parts(
            Translation3::new(0.0, 0.0, -1.0),
            Rotation3::identity().into(),
        ),
        Isometry3::from_parts(
            Translation3::new(-0.05, 0.05, -1.05),
            Rotation3::from_euler_angles(0.0, 0.0, 0.1).into(),
        ),
        Isometry3::from_parts(
            Translation3::new(0.05, -0.05, -0.95),
            Rotation3::from_euler_angles(0.0, 0.0, -0.1).into(),
        ),
        Isometry3::from_parts(
            Translation3::new(-0.03, 0.03, -1.02),
            Rotation3::from_euler_angles(0.0, 0.0, 0.05).into(),
        ),
        Isometry3::from_parts(
            Translation3::new(0.02, 0.04, -1.05),
            Rotation3::from_euler_angles(0.02, 0.03, -0.05).into(),
        ),
    ];

    let target_pose_gt = Isometry3::identity();

    let mut views: Vec<RigView<RobotPoseMeta>> = Vec::new();
    for robot_pose in &robot_poses_gt {
        let mut points_3d = Vec::new();
        let mut points_2d = Vec::new();
        for y in -2..=2 {
            for x in -3..=3 {
                let pw = Pt3::new(x as Real * 0.05, y as Real * 0.05, 0.0);
                let p_base = target_pose_gt.transform_point(&pw);
                let p_gripper = robot_pose.inverse_transform_point(&p_base);
                let p_rig = handeye_gt.inverse_transform_point(&p_gripper);
                let p_cam = cam_to_rig_gt.inverse_transform_point(&p_rig);
                if let Some(pixel) = camera_gt.project_point(&p_cam) {
                    points_3d.push(pw);
                    points_2d.push(pixel);
                }
            }
        }
        views.push(RigView {
            obs: RigViewObs {
                cameras: vec![Some(CorrespondenceView::new(points_3d, points_2d).unwrap())],
            },
            meta: RobotPoseMeta {
                base_se3_gripper: *robot_pose,
            },
        });
    }

    let dataset = HandEyeDataset::new(views, 1, HandEyeMode::EyeInHand).unwrap();

    // The sensor tilt starts at zero.
    let cameras_init = vec![CameraParams {
        projection: ProjectionParams::Pinhole,
        distortion: DistortionParams::BrownConrady5 {
            params: BrownConrady5 {
                k1: -0.18,
                k2: 0.03,
                k3: 0.0,
                p1: 0.0,
                p2: 0.0,
                iters: 8,
            },
        },
        sensor: SensorParams::Scheimpflug {
            params: ScheimpflugParams::default(),
        },
        intrinsics: FxFyCxCy {
            fx: 810.0,
            fy: 790.0,
            cx: 645.0,
            cy: 365.0,
        },
    }];
    let handeye_init = Isometry3::from_parts(
        Translation3::new(0.048, 0.032, 0.148),
        Rotation3::from_euler_angles(0.098, -0.048, 0.148).into(),
    );

    let initial = HandEyeParams {
        cameras: cameras_init,
        cam_to_rig: vec![cam_to_rig_gt],
        handeye: handeye_init,
        target_poses: vec![Isometry3::identity()],
    };

    // The tangential terms are solved (a tilted sensor's default fixes them).
    let opts = HandEyeSolveOptions {
        fix: vec![CameraFix {
            distortion: Some(vec!["k3".into()]),
            ..CameraFix::default()
        }],
        fix_extrinsics: vec![true],
        ..Default::default()
    };
    let backend_opts = BackendSolveOptions {
        max_iters: 150,
        verbosity: 0,
        min_abs_decrease: Some(1e-12),
        min_rel_decrease: Some(1e-12),
        min_error: Some(1e-14),
        ..Default::default()
    };

    let result = optimize_handeye(dataset, initial, opts, backend_opts).unwrap();

    // Intrinsics convergence
    let cam = &result.params.cameras[0].intrinsics;
    assert!(
        (cam.fx - intrinsics_gt.fx).abs() < 0.5,
        "fx err={}",
        (cam.fx - intrinsics_gt.fx).abs()
    );
    assert!(
        (cam.fy - intrinsics_gt.fy).abs() < 0.5,
        "fy err={}",
        (cam.fy - intrinsics_gt.fy).abs()
    );

    // Scheimpflug recovery
    let SensorParams::Scheimpflug { params: sensor } = &result.params.cameras[0].sensor else {
        panic!("the camera lost its tilted sensor");
    };
    let tilt_x_err = (sensor.tilt_x - sensor_gt.tilt_x).abs();
    let tilt_y_err = (sensor.tilt_y - sensor_gt.tilt_y).abs();
    println!("Scheimpflug errors: tilt_x={tilt_x_err:.3e} tilt_y={tilt_y_err:.3e}");
    assert!(tilt_x_err < 5e-3, "tilt_x err={tilt_x_err}");
    assert!(tilt_y_err < 5e-3, "tilt_y err={tilt_y_err}");

    // Hand-eye recovery
    let handeye_final = &result.params.handeye;
    let dt = (handeye_final.translation.vector - handeye_gt.translation.vector).norm();
    let r_diff = handeye_final.rotation.to_rotation_matrix().transpose()
        * handeye_gt.rotation.to_rotation_matrix();
    let cos_theta = ((r_diff.matrix().trace() - 1.0) * 0.5).clamp(-1.0, 1.0);
    let ang = cos_theta.acos();
    println!("Hand-eye errors: translation={dt:.3e} rotation={ang:.3e}");
    assert!(dt < 5e-3, "translation err={dt}");
    assert!(ang < 5e-3, "rotation err={ang}");

    assert!(
        result.mean_reproj_error < 1e-2,
        "mean reproj err too large: {}",
        result.mean_reproj_error
    );

    println!(
        "✓ Eye-in-hand Scheimpflug converged: reproj={:.3e}",
        result.mean_reproj_error
    );
}
