//! Synthetic multi-camera rig + robot hand-eye calibration session.

use anyhow::Result;
use calib::core::{Iso3, Pt3, Real, Vec2};
use calib::optim::ir::HandEyeMode;
use calib::prelude::*;
use nalgebra::{UnitQuaternion, Vector3};

fn main() -> Result<()> {
    let mode = parse_mode(
        std::env::args()
            .collect::<Vec<_>>()
            .iter()
            .map(|s| s.as_str()),
    );

    // Camera model
    let k_gt = FxFyCxCySkew {
        fx: 800.0,
        fy: 780.0,
        cx: 640.0,
        cy: 360.0,
        skew: 0.0,
    };
    let dist_gt = BrownConrady5 {
        k1: 0.0,
        k2: 0.0,
        k3: 0.0,
        p1: 0.0,
        p2: 0.0,
        iters: 8,
    };
    let cam0 = Camera::new(Pinhole, dist_gt, IdentitySensor, k_gt);
    let cam1 = Camera::new(Pinhole, dist_gt, IdentitySensor, k_gt);

    // Rig extrinsics (camera -> rig)
    let cam0_to_rig = Iso3::identity();
    let cam1_to_rig = Iso3::from_parts(
        Vector3::new(0.18, -0.02, 0.0).into(),
        UnitQuaternion::from_scaled_axis(Vector3::new(0.0, -0.04, 0.0)),
    );

    // Board points
    let nx = 6;
    let ny = 5;
    let spacing = 0.04_f64;
    let mut board_points = Vec::new();
    for j in 0..ny {
        for i in 0..nx {
            board_points.push(Pt3::new(i as f64 * spacing, j as f64 * spacing, 0.0));
        }
    }

    // Scenario constants depend on mode.
    let (handeye_gt, target_gt) = match mode {
        HandEyeMode::EyeInHand => {
            // gripper_from_rig, base_from_target
            let gripper_from_rig = Iso3::from_parts(
                Vector3::new(0.02, -0.01, 0.05).into(),
                UnitQuaternion::from_scaled_axis(Vector3::new(0.02, 0.01, -0.03)),
            );
            let base_from_target = Iso3::from_parts(
                Vector3::new(0.3, -0.1, 0.8).into(),
                UnitQuaternion::from_scaled_axis(Vector3::new(0.1, -0.05, 0.02)),
            );
            (gripper_from_rig, base_from_target)
        }
        HandEyeMode::EyeToHand => {
            // rig_from_base, gripper_from_target
            let rig_from_base = Iso3::from_parts(
                Vector3::new(0.05, 0.02, 0.01).into(),
                UnitQuaternion::from_scaled_axis(Vector3::new(-0.01, 0.02, 0.04)),
            );
            let gripper_from_target = Iso3::from_parts(
                Vector3::new(0.0, 0.0, 0.4).into(),
                UnitQuaternion::from_scaled_axis(Vector3::new(0.1, 0.0, 0.0)),
            );
            (rig_from_base, gripper_from_target)
        }
    };

    // Views
    let mut views = Vec::new();
    for view_idx in 0..6 {
        let base_from_gripper = Iso3::from_parts(
            Vector3::new(0.0, 0.0, 0.2 + 0.03 * view_idx as f64).into(),
            UnitQuaternion::from_scaled_axis(Vector3::new(
                0.01 * view_idx as f64,
                0.04 * view_idx as f64,
                0.02 * view_idx as f64,
            )),
        );

        let rig_from_target = match mode {
            HandEyeMode::EyeInHand => {
                let base_from_rig = base_from_gripper * handeye_gt;
                base_from_rig.inverse() * target_gt
            }
            HandEyeMode::EyeToHand => handeye_gt * base_from_gripper * target_gt,
        };

        let cam0_from_target = cam0_to_rig.inverse() * rig_from_target;
        let cam1_from_target = cam1_to_rig.inverse() * rig_from_target;

        let mut cam0_pixels = Vec::new();
        let mut cam1_pixels = Vec::new();
        for pw in &board_points {
            if let Some(p) = cam0.project_point(&cam0_from_target.transform_point(pw)) {
                cam0_pixels.push(Vec2::new(p.x, p.y));
            }
            if let Some(p) = cam1.project_point(&cam1_from_target.transform_point(pw)) {
                cam1_pixels.push(Vec2::new(p.x, p.y));
            }
        }

        views.push(calib::session::problem_types::RigHandEyeViewData {
            cameras: vec![
                Some(CameraViewData {
                    points_3d: board_points.clone(),
                    points_2d: cam0_pixels,
                    weights: None,
                }),
                Some(CameraViewData {
                    points_3d: board_points.clone(),
                    points_2d: cam1_pixels,
                    weights: None,
                }),
            ],
            base_from_gripper,
        });
    }

    let obs = calib::session::problem_types::RigHandEyeObservations {
        views,
        num_cameras: 2,
        mode,
    };

    let mut session = CalibrationSession::<calib::session::problem_types::RigHandEyeProblem>::new();
    session.set_observations(obs.clone());

    println!("=== Rig + Hand-Eye (synthetic) ===");
    println!("Mode: {:?}", mode);
    println!();

    session.initialize(calib::session::problem_types::RigHandEyeInitOptions::default())?;
    let init = session.initial_values().unwrap();
    println!("--- Seed ---");
    println!("ref_cam_idx: {}", init.ref_cam_idx);
    println!(
        "rig seed mean reproj error: {:.6} px",
        rig_seed_reproj(&obs, init)?
    );
    println!();

    let mut optim_opts = calib::session::problem_types::RigHandEyeOptimOptions::default();
    optim_opts.solve_opts.refine_robot_poses = true;
    optim_opts.backend_opts.max_iters = 200;
    session.optimize(optim_opts)?;

    let report = session.export()?;
    println!("--- Optimized ---");
    println!("final_cost: {:.6}", report.final_cost);
    println!(
        "joint mean reproj error: {:.6} px",
        mean_reproj_error(&obs, &report)?
    );
    println!("handeye:\n  t={:?}", report.handeye.translation.vector);
    println!();

    Ok(())
}

fn parse_mode<'a>(args: impl Iterator<Item = &'a str>) -> HandEyeMode {
    for a in args {
        if let Some(raw) = a.strip_prefix("--mode=") {
            return match raw {
                "eye-in-hand" | "eih" => HandEyeMode::EyeInHand,
                "eye-to-hand" | "eth" => HandEyeMode::EyeToHand,
                _ => HandEyeMode::EyeInHand,
            };
        }
    }
    HandEyeMode::EyeInHand
}

fn rig_seed_reproj(
    obs: &calib::session::problem_types::RigHandEyeObservations,
    init: &calib::session::problem_types::RigHandEyeInitial,
) -> Result<Real> {
    let input = calib::rig::RigExtrinsicsInput {
        views: obs
            .views
            .iter()
            .map(|v| calib::rig::RigViewData {
                cameras: v.cameras.clone(),
            })
            .collect(),
        num_cameras: obs.num_cameras,
    };
    let err = calib::rig::rig_reprojection_errors(
        &input,
        &init.cameras,
        &init.cam_to_rig,
        &init.rig_from_target,
    )?;
    Ok(err.mean_px.unwrap_or(f64::NAN))
}

fn mean_reproj_error(
    obs: &calib::session::problem_types::RigHandEyeObservations,
    report: &calib::session::problem_types::RigHandEyeOptimized,
) -> Result<Real> {
    use calib::core::{Camera, IdentitySensor, Pinhole};

    let mut total_error = 0.0;
    let mut total_n = 0usize;

    for (view_idx, view) in obs.views.iter().enumerate() {
        let base_from_gripper = view.base_from_gripper;
        for (cam_idx, cam_view_opt) in view.cameras.iter().enumerate() {
            let Some(cam_view) = cam_view_opt else {
                continue;
            };

            let k = match report.cameras[cam_idx].intrinsics {
                calib::core::IntrinsicsParams::FxFyCxCySkew { params } => params,
            };
            let dist = match report.cameras[cam_idx].distortion {
                calib::core::DistortionParams::BrownConrady5 { params } => params,
                calib::core::DistortionParams::None => dist_gt(),
            };
            let camera = Camera::new(Pinhole, dist, IdentitySensor, k);

            let target_pose = report.target_poses[view_idx];

            for (pw, uv) in cam_view.points_3d.iter().zip(cam_view.points_2d.iter()) {
                let p_cam = match obs.mode {
                    HandEyeMode::EyeInHand => {
                        let p_base = target_pose.transform_point(pw);
                        let p_gripper = base_from_gripper.inverse_transform_point(&p_base);
                        let p_rig = report.handeye.inverse_transform_point(&p_gripper);
                        report.cam_to_rig[cam_idx].inverse_transform_point(&p_rig)
                    }
                    HandEyeMode::EyeToHand => {
                        let p_gripper = target_pose.transform_point(pw);
                        let p_base = base_from_gripper.transform_point(&p_gripper);
                        let p_rig = report.handeye.transform_point(&p_base);
                        report.cam_to_rig[cam_idx].inverse_transform_point(&p_rig)
                    }
                };

                let Some(proj) = camera.project_point(&p_cam) else {
                    continue;
                };

                total_error += (proj - *uv).norm();
                total_n += 1;
            }
        }
    }

    anyhow::ensure!(total_n > 0, "no valid projections");
    Ok(total_error / total_n as Real)
}

fn dist_gt() -> calib::core::BrownConrady5<Real> {
    calib::core::BrownConrady5 {
        k1: 0.0,
        k2: 0.0,
        k3: 0.0,
        p1: 0.0,
        p2: 0.0,
        iters: 8,
    }
}
