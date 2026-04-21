//! Integration test for Scheimpflug rig extrinsics optimization.
//!
//! Mirrors `rig_extrinsics.rs` but gives camera 1 a non-zero Scheimpflug tilt
//! and verifies convergence of both the extrinsics and the tilt parameters.

use nalgebra::{Isometry3, Rotation3, Translation3};
use vision_calibration_core::{
    BrownConrady5, Camera, CorrespondenceView, FxFyCxCySkew, NoMeta, Pinhole, Pt3, Real,
    RigDataset, RigView, RigViewObs, ScheimpflugParams, make_pinhole_camera,
};
use vision_calibration_optim::{
    BackendSolveOptions, RigExtrinsicsScheimpflugParams, RigExtrinsicsScheimpflugSolveOptions,
    ScheimpflugFixMask, optimize_rig_extrinsics_scheimpflug,
};

#[test]
fn stereo_scheimpflug_rig_extrinsics_converges() {
    let intrinsics_gt = FxFyCxCySkew {
        fx: 800.0,
        fy: 780.0,
        cx: 640.0,
        cy: 360.0,
        skew: 0.0,
    };
    let distortion_gt = BrownConrady5 {
        k1: -0.20,
        k2: 0.06,
        k3: 0.0,
        p1: 0.0005,
        p2: -0.0005,
        iters: 5,
    };

    // Camera 0 is pinhole-equivalent; camera 1 has a Scheimpflug tilt around X.
    let sensors_gt = [
        ScheimpflugParams::default(),
        ScheimpflugParams {
            tilt_x: 0.08,
            tilt_y: -0.05,
        },
    ];

    let cam_gt: Vec<Camera<Real, Pinhole, BrownConrady5<Real>, _, FxFyCxCySkew<Real>>> = sensors_gt
        .iter()
        .map(|sensor| Camera::new(Pinhole, distortion_gt, sensor.compile(), intrinsics_gt))
        .collect();
    let pinhole_cameras_gt = [
        make_pinhole_camera(intrinsics_gt, distortion_gt),
        make_pinhole_camera(intrinsics_gt, distortion_gt),
    ];

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
    let intrinsics_init = FxFyCxCySkew {
        fx: 810.0,
        fy: 790.0,
        cx: 645.0,
        cy: 365.0,
        skew: 0.0,
    };
    let distortion_init = BrownConrady5 {
        k1: -0.15,
        k2: 0.04,
        k3: 0.0,
        p1: 0.0,
        p2: 0.0,
        iters: 8,
    };
    let cameras_init = vec![
        make_pinhole_camera(intrinsics_init, distortion_init),
        make_pinhole_camera(intrinsics_init, distortion_init),
    ];
    // Start the sensor tilts at zero (the optimizer must recover them).
    let sensors_init = vec![ScheimpflugParams::default(), ScheimpflugParams::default()];

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

    let initial = RigExtrinsicsScheimpflugParams {
        cameras: cameras_init,
        sensors: sensors_init,
        cam_to_rig: vec![cam0_to_rig_gt, cam1_to_rig_init],
        rig_from_target: rig_from_target_init,
    };

    // Break gauge: fix cam0 pose (as in pinhole test) AND fix cam0 Scheimpflug
    // since its ground-truth tilt is zero and a free cam0 tilt admits a partial
    // ambiguity with per-view rig poses.
    let opts = RigExtrinsicsScheimpflugSolveOptions {
        fix_extrinsics: vec![true, false],
        scheimpflug_overrides: vec![
            Some(ScheimpflugFixMask {
                tilt_x: true,
                tilt_y: true,
            }),
            None,
        ],
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

    let result = optimize_rig_extrinsics_scheimpflug(dataset, initial, opts, backend_opts).unwrap();

    // Intrinsics/distortion convergence.
    for (cam_idx, cam) in result.params.cameras.iter().enumerate() {
        let gt = &pinhole_cameras_gt[cam_idx];
        assert!(
            (cam.k.fx - gt.k.fx).abs() < 0.5,
            "cam{cam_idx} fx err={}",
            (cam.k.fx - gt.k.fx).abs()
        );
        assert!(
            (cam.k.fy - gt.k.fy).abs() < 0.5,
            "cam{cam_idx} fy err={}",
            (cam.k.fy - gt.k.fy).abs()
        );
    }

    // Scheimpflug tilt recovery.
    for (cam_idx, sensor) in result.params.sensors.iter().enumerate() {
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
