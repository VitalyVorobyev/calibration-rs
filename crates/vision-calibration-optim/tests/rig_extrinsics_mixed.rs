//! Integration test for a rig that mixes lenses: camera 0 has an untilted
//! sensor, camera 1 a Scheimpflug-tilted one. Verifies convergence of the
//! extrinsics and of camera 1's tilt.

use nalgebra::{Isometry3, Rotation3, Translation3};
use vision_calibration_core::{
    BrownConrady5, CameraFix, CameraParams, CorrespondenceView, DistortionParams, FxFyCxCy, NoMeta,
    ProjectionParams, Pt3, Real, RigDataset, RigView, RigViewObs, ScheimpflugParams, SensorParams,
};
use vision_calibration_optim::{
    BackendSolveOptions, RigExtrinsicsParams, RigExtrinsicsSolveOptions, optimize_rig_extrinsics,
};

fn camera(k: FxFyCxCy<Real>, dist: BrownConrady5<Real>, sensor: SensorParams) -> CameraParams {
    CameraParams {
        projection: ProjectionParams::Pinhole,
        distortion: DistortionParams::BrownConrady5 { params: dist },
        sensor,
        intrinsics: k,
    }
}

fn tilt(camera: &CameraParams) -> ScheimpflugParams {
    match camera.sensor {
        SensorParams::Scheimpflug { params } => params,
        _ => ScheimpflugParams::default(),
    }
}

#[test]
fn stereo_scheimpflug_rig_extrinsics_converges() {
    let intrinsics_gt = FxFyCxCy {
        fx: 800.0,
        fy: 780.0,
        cx: 640.0,
        cy: 360.0,
    };
    let distortion_gt = BrownConrady5 {
        k1: -0.20,
        k2: 0.06,
        k3: 0.0,
        p1: 0.0005,
        p2: -0.0005,
        iters: 5,
    };

    // Camera 0 has an untilted sensor; camera 1 a Scheimpflug tilt.
    let sensors_gt = [
        ScheimpflugParams::default(),
        ScheimpflugParams {
            tilt_x: 0.08,
            tilt_y: -0.05,
        },
    ];
    let cameras_gt = [
        camera(intrinsics_gt, distortion_gt, SensorParams::Identity),
        camera(
            intrinsics_gt,
            distortion_gt,
            SensorParams::Scheimpflug {
                params: sensors_gt[1],
            },
        ),
    ];
    let cam_gt: Vec<_> = cameras_gt.iter().map(|c| c.build().unwrap()).collect();

    // Ground truth extrinsics.
    let cam0_to_rig_gt = Isometry3::identity();
    let cam1_to_rig_gt = Isometry3::from_parts(
        Translation3::new(0.12, 0.0, 0.0),
        Rotation3::identity().into(),
    );
    let cam_to_rig_gt = [cam0_to_rig_gt, cam1_to_rig_gt];

    let rig_from_target_gt = vec![
        Isometry3::from_parts(
            Translation3::new(0.1, -0.05, 1.0),
            Rotation3::from_euler_angles(0.1, -0.05, 0.2).into(),
        ),
        Isometry3::from_parts(
            Translation3::new(-0.08, 0.03, 1.1),
            Rotation3::from_euler_angles(-0.08, 0.06, -0.15).into(),
        ),
        Isometry3::from_parts(
            Translation3::new(0.05, 0.08, 0.95),
            Rotation3::from_euler_angles(0.05, -0.08, 0.12).into(),
        ),
        Isometry3::from_parts(
            Translation3::new(-0.02, 0.06, 1.05),
            Rotation3::from_euler_angles(-0.05, 0.10, 0.08).into(),
        ),
    ];

    // Generate synthetic observations using the tilted cameras.
    let mut views = Vec::new();
    for rig_pose in &rig_from_target_gt {
        let mut cameras_obs = Vec::new();
        for (cam_idx, cam_to_rig) in cam_to_rig_gt.iter().enumerate() {
            let mut points_3d = Vec::new();
            let mut points_2d = Vec::new();
            for y in -2..=2 {
                for x in -3..=3 {
                    let pw = Pt3::new(x as Real * 0.05, y as Real * 0.05, 0.0);
                    let p_rig = rig_pose.transform_point(&pw);
                    let p_cam = cam_to_rig.inverse_transform_point(&p_rig);
                    if let Some(pixel) = cam_gt[cam_idx].project_point(&p_cam) {
                        points_3d.push(pw);
                        points_2d.push(pixel);
                    }
                }
            }
            cameras_obs.push(Some(CorrespondenceView::new(points_3d, points_2d).unwrap()));
        }
        views.push(RigView {
            obs: RigViewObs {
                cameras: cameras_obs,
            },
            meta: NoMeta,
        });
    }

    let dataset = RigDataset::new(views, 2).unwrap();

    // Initial values (perturbed from ground truth).
    let intrinsics_init = FxFyCxCy {
        fx: 810.0,
        fy: 790.0,
        cx: 645.0,
        cy: 365.0,
    };
    let distortion_init = BrownConrady5 {
        k1: -0.15,
        k2: 0.04,
        k3: 0.0,
        p1: 0.0,
        p2: 0.0,
        iters: 8,
    };
    // Start camera 1's tilt at zero (the optimizer must recover it).
    let cameras_init = vec![
        camera(intrinsics_init, distortion_init, SensorParams::Identity),
        camera(
            intrinsics_init,
            distortion_init,
            SensorParams::Scheimpflug {
                params: ScheimpflugParams::default(),
            },
        ),
    ];

    let cam1_to_rig_init = Isometry3::from_parts(
        Translation3::new(0.115, 0.003, -0.002),
        Rotation3::from_euler_angles(0.008, 0.005, -0.01).into(),
    );
    let rig_from_target_init = rig_from_target_gt
        .iter()
        .map(|iso| {
            let t = iso.translation.vector + nalgebra::Vector3::new(0.008, -0.005, 0.01);
            let r = iso.rotation.to_rotation_matrix()
                * Rotation3::from_euler_angles(0.01, -0.008, 0.005);
            Isometry3::from_parts(Translation3::from(t), r.into())
        })
        .collect::<Vec<_>>();

    let initial = RigExtrinsicsParams {
        cameras: cameras_init,
        cam_to_rig: vec![cam0_to_rig_gt, cam1_to_rig_init],
        rig_from_target: rig_from_target_init,
    };

    // Break gauge: fix cam0 pose. Both cameras keep k3 fixed and solve the
    // tangential terms (the tilted camera's default would fix them).
    let k3_only = CameraFix {
        distortion: Some(vec!["k3".into()]),
        ..CameraFix::default()
    };
    let opts = RigExtrinsicsSolveOptions {
        fix: vec![k3_only.clone(), k3_only],
        fix_extrinsics: vec![true, false],
        ..Default::default()
    };
    let backend_opts = BackendSolveOptions {
        max_iters: 120,
        verbosity: 0,
        min_abs_decrease: Some(1e-12),
        min_rel_decrease: Some(1e-12),
        min_error: Some(1e-14),
        ..Default::default()
    };

    let result = optimize_rig_extrinsics(dataset, initial, opts, backend_opts).unwrap();
    assert!(matches!(
        result.params.cameras[0].sensor,
        SensorParams::Identity
    ));

    // Intrinsics/distortion convergence.
    for (cam_idx, cam) in result.params.cameras.iter().enumerate() {
        let gt = &cameras_gt[cam_idx].intrinsics;
        let err_fx = (cam.intrinsics.fx - gt.fx).abs();
        let err_fy = (cam.intrinsics.fy - gt.fy).abs();
        assert!(err_fx < 0.5, "cam{cam_idx} fx err={err_fx}");
        assert!(err_fy < 0.5, "cam{cam_idx} fy err={err_fy}");
    }

    // Scheimpflug tilt recovery.
    for (cam_idx, cam) in result.params.cameras.iter().enumerate() {
        let sensor = tilt(cam);
        let err_x = (sensor.tilt_x - sensors_gt[cam_idx].tilt_x).abs();
        let err_y = (sensor.tilt_y - sensors_gt[cam_idx].tilt_y).abs();
        println!("Camera {cam_idx} tilt errors: tilt_x={err_x:.3e} tilt_y={err_y:.3e}");
        assert!(err_x < 5e-3, "cam{cam_idx} tilt_x err={err_x}");
        assert!(err_y < 5e-3, "cam{cam_idx} tilt_y err={err_y}");
    }

    // Camera 1 extrinsics recovery.
    let cam1_final = &result.params.cam_to_rig[1];
    let dt = (cam1_final.translation.vector - cam1_to_rig_gt.translation.vector).norm();
    let r_diff = cam1_final.rotation.to_rotation_matrix().transpose()
        * cam1_to_rig_gt.rotation.to_rotation_matrix();
    let cos_theta = ((r_diff.matrix().trace() - 1.0) * 0.5).clamp(-1.0, 1.0);
    let ang = cos_theta.acos();
    assert!(dt < 5e-3, "cam1 translation err={dt}");
    assert!(ang < 5e-3, "cam1 rotation err={ang}");

    assert!(
        result.mean_reproj_error < 1e-2,
        "mean reproj error too large: {}",
        result.mean_reproj_error
    );

    println!(
        "✓ Scheimpflug rig extrinsics converged: reproj={:.3e}",
        result.mean_reproj_error
    );
}
