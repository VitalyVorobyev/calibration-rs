//! The stopping rule in a joint solve whose residual families have different
//! units and noise: reprojection in pixels beside a point-to-plane laser
//! residual in metres. Started a σ off in the plane, the solve must end
//! within 0.01 of the plane's own σ of its minimum, on both backends.

#![allow(missing_docs)]

use nalgebra::Vector3;
use vision_calibration_core::synthetic::laser::{BoardExtent, laser_stripe_pixels};
use vision_calibration_core::synthetic::noise::UniformPixelNoise;
use vision_calibration_core::synthetic::{planar, poses};
use vision_calibration_core::{
    BrownConrady5, CameraParams, DistortionParams, FxFyCxCy, LaserPlane, ProjectionParams,
    SensorParams, Uncertainty, View,
};
use vision_calibration_optim::{
    BackendSolveOptions, LaserlineDataset, LaserlineMeta, LaserlineParams, LaserlineResidualType,
    LaserlineSolveOptions, SolverBackend, optimize_laserline,
};

fn scene() -> (LaserlineDataset, LaserlineParams) {
    let camera = CameraParams::new(
        ProjectionParams::Pinhole,
        DistortionParams::BrownConrady5 {
            params: BrownConrady5 {
                k1: -0.05,
                ..BrownConrady5::default()
            },
        },
        SensorParams::Identity,
        FxFyCxCy {
            fx: 850.0,
            fy: 845.0,
            cx: 640.0,
            cy: 480.0,
        },
    );
    let model = camera.build().unwrap();
    let normal = Vector3::new(0.30, 0.0, 1.0).normalize();
    let plane = LaserPlane::new(normal, -0.5 * normal.z).unwrap();
    let extent = BoardExtent::from_grid(9, 7, 0.06);
    let c = extent.center();
    let specs = poses::seeded_board_pose_specs(12, 0.35, 0x51DE);
    let board_poses = poses::centered_board_poses(&specs, (c[0], c[1]), 0.5);
    let noise = |seed| UniformPixelNoise {
        seed,
        max_abs_px: 0.1 * 3.0_f64.sqrt(),
    };
    let corners = planar::project_views_noisy(
        &model,
        &planar::grid_points(9, 7, 0.06),
        &board_poses,
        &noise(1),
    )
    .unwrap();
    let views = corners
        .into_iter()
        .zip(&board_poses)
        .enumerate()
        .map(|(v, (obs, pose))| {
            let stripe = laser_stripe_pixels(
                &model,
                pose,
                &plane.normal(),
                plane.distance(),
                &extent,
                v,
                &noise(2),
            );
            View::new(obs, LaserlineMeta::new(stripe))
        })
        .collect();
    let params = LaserlineParams::new(camera, board_poses, plane).unwrap();
    (LaserlineDataset::new(views).unwrap(), params)
}

#[test]
fn a_plane_measured_in_metres_beside_pixels_ends_at_its_minimum() {
    let (dataset, truth) = scene();
    let opts = LaserlineSolveOptions {
        laser_residual_type: LaserlineResidualType::PointToPlane,
        ..LaserlineSolveOptions::default()
    };
    for backend in [SolverBackend::TinySolver, SolverBackend::Factrs] {
        let tight = BackendSolveOptions {
            backend,
            max_iters: 300,
            sigma_tolerance: 0.0,
            min_abs_decrease: Some(0.0),
            min_rel_decrease: Some(1e-15),
            ..BackendSolveOptions::default()
        };
        let reference = optimize_laserline(&dataset, &truth, &opts, &tight).unwrap();
        let Uncertainty::Available(cov) = &reference.uncertainty else {
            panic!("{backend:?}: no covariance");
        };
        let sigma_d = cov.std_dev("plane.d").unwrap();
        let minimum = reference.params.laser_plane.distance();
        // The minimum, the plane moved a σ along its distance.
        let start_plane = LaserPlane::new(
            reference.params.laser_plane.normal().into_inner(),
            minimum + sigma_d,
        )
        .unwrap();
        let start = LaserlineParams::new(
            reference.params.camera.clone(),
            reference.params.cam_se3_target.clone(),
            start_plane,
        )
        .unwrap();
        let default = BackendSolveOptions {
            backend,
            ..BackendSolveOptions::default()
        };
        let solved = optimize_laserline(&dataset, &start, &opts, &default).unwrap();
        let off = (solved.params.laser_plane.distance() - minimum).abs() / sigma_d;
        println!(
            "{backend:?}: σ(d) {sigma_d:.3e} m, ends {off:.2e} σ off after {} iterations ({:?})",
            solved.report.num_iters, solved.report.termination
        );
        assert!(off <= 0.01, "{backend:?}: the plane ends {off:.4} σ off");
    }
}
