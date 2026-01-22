//! Multi-camera rig hand-eye calibration with synthetic data.
//!
//! This example demonstrates the full 6-step rig hand-eye calibration workflow:
//! 1. Initialize per-camera intrinsics (Zhang's method)
//! 2. Optimize per-camera intrinsics (bundle adjustment)
//! 3. Initialize rig extrinsics (camera-to-rig transforms)
//! 4. Optimize rig extrinsics (bundle adjustment)
//! 5. Initialize hand-eye transform (Tsai-Lenz)
//! 6. Optimize hand-eye (bundle adjustment with robot pose priors)
//!
//! The example uses a synthetic stereo rig (2 cameras) mounted on a robot arm.
//!
//! Run with: `cargo run -p calib --example rig_handeye_synthetic`

use anyhow::Result;
use calib::core::{
    make_pinhole_camera, BrownConrady5, CorrespondenceView, FxFyCxCySkew, Iso3, Pt2, Pt3,
    RigDataset, RigView, RigViewObs,
};
use calib::handeye::RobotPoseMeta;
use calib::prelude::*;
use calib::rig_handeye::{
    step_handeye_init, step_handeye_optimize, step_intrinsics_init_all, step_intrinsics_optimize_all,
    step_rig_init, step_rig_optimize, RigHandeyeInput, RigHandeyeProblem,
};
use nalgebra::{Rotation3, Translation3, UnitQuaternion};

fn main() -> Result<()> {
    println!("=== Multi-Camera Rig Hand-Eye Calibration (Synthetic) ===\n");

    // Ground truth camera parameters (stereo rig with 2 cameras)
    let k0_gt = FxFyCxCySkew {
        fx: 800.0,
        fy: 780.0,
        cx: 640.0,
        cy: 360.0,
        skew: 0.0,
    };
    let k1_gt = FxFyCxCySkew {
        fx: 810.0,
        fy: 790.0,
        cx: 635.0,
        cy: 355.0,
        skew: 0.0,
    };
    // Zero distortion for cleaner synthetic data
    let dist_gt = BrownConrady5::default();
    let cam0_gt = make_pinhole_camera(k0_gt, dist_gt);
    let cam1_gt = make_pinhole_camera(k1_gt, dist_gt);

    // Ground truth rig extrinsics: cam_se3_rig (T_C_R)
    // Camera 0 is at rig origin (identity)
    let cam0_se3_rig_gt = Iso3::identity();
    // Camera 1 is 12cm to the right of camera 0
    let cam1_se3_rig_gt = make_iso((0.0, 0.0, 0.0), (0.12, 0.0, 0.0));

    // Ground truth hand-eye transform: gripper_se3_rig (T_G_R)
    let handeye_gt = make_iso((0.05, -0.03, 0.1), (0.0, -0.05, 0.08));

    // Ground truth target pose in base frame: base_se3_target (T_B_T)
    // Target is 1m in front of the base
    let target_in_base_gt = make_iso((0.0, 0.0, 0.0), (0.0, 0.0, 1.0));

    println!("Ground truth:");
    println!(
        "  Camera 0: fx={:.1}, fy={:.1}, cx={:.1}, cy={:.1}",
        k0_gt.fx, k0_gt.fy, k0_gt.cx, k0_gt.cy
    );
    println!(
        "  Camera 1: fx={:.1}, fy={:.1}, cx={:.1}, cy={:.1}",
        k1_gt.fx, k1_gt.fy, k1_gt.cx, k1_gt.cy
    );
    println!("  Distortion: zero (default)");
    println!(
        "  Baseline (cam1-cam0): {:.4}m",
        cam1_se3_rig_gt.translation.vector.norm()
    );
    println!(
        "  Hand-eye |t|: {:.4}m",
        handeye_gt.translation.vector.norm()
    );
    println!();

    // Generate calibration board points (6x5 grid, 50mm squares)
    let board_pts: Vec<Pt3> = (0..6)
        .flat_map(|i| (0..5).map(move |j| Pt3::new(i as f64 * 0.05, j as f64 * 0.05, 0.0)))
        .collect();

    // Generate robot poses with diverse rotations and translations
    // Important: diverse rotation axes are critical for hand-eye calibration
    let robot_poses = [
        make_iso((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        make_iso((0.3, 0.0, 0.0), (0.1, 0.0, 0.0)),       // Roll rotation
        make_iso((0.0, 0.3, 0.0), (0.0, 0.1, 0.0)),       // Pitch rotation
        make_iso((0.0, 0.0, 0.3), (0.0, 0.0, 0.1)),       // Yaw rotation
        make_iso((0.2, 0.2, 0.0), (0.05, -0.05, 0.0)),    // Combined
        make_iso((-0.2, 0.0, 0.2), (-0.05, 0.05, 0.0)),   // Different axis
        make_iso((0.15, -0.15, 0.1), (0.0, -0.05, 0.05)), // More variation
    ];

    // Project observations through the transformation chain
    println!("Generating {} views for stereo rig...", robot_poses.len());
    let views: Vec<RigView<RobotPoseMeta>> = robot_poses
        .iter()
        .enumerate()
        .map(|(idx, robot_pose)| {
            // Transformation chain for EyeInHand rig:
            // target → T_B_T → base → robot_pose^-1 → gripper → handeye^-1 → rig → cam_se3_rig → camera
            //
            // rig_se3_target = (robot_pose * handeye_gt)^-1 * target_in_base_gt
            let rig_se3_target = (robot_pose * handeye_gt).inverse() * target_in_base_gt;

            // Project for each camera
            let obs0 = project_view(&cam0_gt, &(cam0_se3_rig_gt * rig_se3_target), &board_pts);
            let obs1 = project_view(&cam1_gt, &(cam1_se3_rig_gt * rig_se3_target), &board_pts);

            println!(
                "  View {}: cam0={} pts, cam1={} pts",
                idx,
                obs0.as_ref().map_or(0, |o| o.points_2d.len()),
                obs1.as_ref().map_or(0, |o| o.points_2d.len())
            );

            RigView {
                meta: RobotPoseMeta {
                    robot_pose: *robot_pose,
                },
                obs: RigViewObs {
                    cameras: vec![obs0, obs1],
                },
            }
        })
        .collect();

    println!();

    // Create input dataset
    let input: RigHandeyeInput = RigDataset::new(views, 2)?;
    println!("Created dataset with {} views, {} cameras\n", input.num_views(), input.num_cameras);

    // Create calibration session
    let mut session = CalibrationSession::<RigHandeyeProblem>::new();
    session.set_input(input)?;

    // Step 1: Per-camera intrinsics initialization
    println!("--- Step 1: Per-Camera Intrinsics Initialization ---");
    step_intrinsics_init_all(&mut session, None)?;
    for (i, cam) in session
        .state
        .per_cam_intrinsics
        .as_ref()
        .unwrap()
        .iter()
        .enumerate()
    {
        println!(
            "  Camera {}: fx={:.1}, fy={:.1}, cx={:.1}, cy={:.1}",
            i, cam.k.fx, cam.k.fy, cam.k.cx, cam.k.cy
        );
    }
    println!();

    // Step 2: Per-camera intrinsics optimization
    println!("--- Step 2: Per-Camera Intrinsics Optimization ---");
    step_intrinsics_optimize_all(&mut session, None)?;
    for (i, cam) in session
        .state
        .per_cam_intrinsics
        .as_ref()
        .unwrap()
        .iter()
        .enumerate()
    {
        let reproj = session
            .state
            .per_cam_reproj_errors
            .as_ref()
            .map(|e| e[i])
            .unwrap_or(f64::NAN);
        println!(
            "  Camera {}: fx={:.1}, fy={:.1}, reproj_err={:.4}px",
            i, cam.k.fx, cam.k.fy, reproj
        );
    }
    println!();

    // Step 3: Rig initialization
    println!("--- Step 3: Rig Extrinsics Initialization ---");
    step_rig_init(&mut session)?;
    let init_extr = session.state.initial_cam_se3_rig.as_ref().unwrap();
    let baseline_init = (init_extr[1].translation.vector - init_extr[0].translation.vector).norm();
    println!(
        "  Initial baseline: {:.4}m (GT: {:.4}m)",
        baseline_init,
        cam1_se3_rig_gt.translation.vector.norm()
    );
    println!();

    // Step 4: Rig optimization
    println!("--- Step 4: Rig Bundle Adjustment ---");
    step_rig_optimize(&mut session, None)?;
    let rig_reproj = session.state.rig_ba_reproj_error.unwrap_or(f64::NAN);
    println!("  Rig BA reprojection error: {:.4} px", rig_reproj);
    println!();

    // Step 5: Hand-eye initialization
    println!("--- Step 5: Hand-Eye Initialization (Tsai-Lenz) ---");
    step_handeye_init(&mut session, None)?;
    let init_he = session.state.initial_handeye.as_ref().unwrap();
    println!(
        "  Hand-eye |t|: {:.4}m (GT: {:.4}m)",
        init_he.translation.vector.norm(),
        handeye_gt.translation.vector.norm()
    );
    if let Some(init_target) = session.state.initial_target_se3_base.as_ref() {
        println!(
            "  Target in base |t|: {:.4}m (GT: {:.4}m)",
            init_target.translation.vector.norm(),
            target_in_base_gt.translation.vector.norm()
        );
    }
    println!();

    // Step 6: Hand-eye optimization
    println!("--- Step 6: Hand-Eye Bundle Adjustment ---");
    step_handeye_optimize(&mut session, None)?;
    let he_reproj = session.state.handeye_reproj_error.unwrap_or(f64::NAN);
    println!("  Final reprojection error: {:.4} px", he_reproj);
    println!();

    // Export and compare with ground truth
    println!("--- Final Results ---");
    let export = session.export()?;

    println!("Camera 0:");
    let k0 = &export.cameras[0].k;
    println!(
        "  fx: {:.2} (GT: {:.2}, err: {:.2}%)",
        k0.fx,
        k0_gt.fx,
        100.0 * (k0.fx - k0_gt.fx).abs() / k0_gt.fx
    );
    println!(
        "  fy: {:.2} (GT: {:.2}, err: {:.2}%)",
        k0.fy,
        k0_gt.fy,
        100.0 * (k0.fy - k0_gt.fy).abs() / k0_gt.fy
    );

    println!("Camera 1:");
    let k1 = &export.cameras[1].k;
    println!(
        "  fx: {:.2} (GT: {:.2}, err: {:.2}%)",
        k1.fx,
        k1_gt.fx,
        100.0 * (k1.fx - k1_gt.fx).abs() / k1_gt.fx
    );
    println!(
        "  fy: {:.2} (GT: {:.2}, err: {:.2}%)",
        k1.fy,
        k1_gt.fy,
        100.0 * (k1.fy - k1_gt.fy).abs() / k1_gt.fy
    );

    println!("Rig extrinsics:");
    let baseline_final =
        (export.cam_se3_rig[1].translation.vector - export.cam_se3_rig[0].translation.vector).norm();
    let baseline_gt = cam1_se3_rig_gt.translation.vector.norm();
    println!(
        "  Baseline: {:.4}m (GT: {:.4}m, err: {:.2}%)",
        baseline_final,
        baseline_gt,
        100.0 * (baseline_final - baseline_gt).abs() / baseline_gt
    );

    println!("Hand-eye transform:");
    let he_t = export.handeye.translation.vector.norm();
    let he_gt_t = handeye_gt.translation.vector.norm();
    println!(
        "  |t|: {:.4}m (GT: {:.4}m, err: {:.2}%)",
        he_t,
        he_gt_t,
        100.0 * (he_t - he_gt_t).abs() / he_gt_t
    );

    println!("Target pose:");
    let target_t = export.target_se3_base.translation.vector.norm();
    let target_gt_t = target_in_base_gt.translation.vector.norm();
    println!(
        "  |t|: {:.4}m (GT: {:.4}m, err: {:.2}%)",
        target_t,
        target_gt_t,
        100.0 * (target_t - target_gt_t).abs() / target_gt_t
    );

    println!("Reprojection error:");
    println!("  Mean: {:.4} px", export.mean_reproj_error);
    println!("  Per-camera: {:?}", export.per_cam_reproj_errors);

    if let Some(deltas) = &export.robot_deltas {
        let max_rot: f64 = deltas
            .iter()
            .map(|d| (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);
        let max_trans: f64 = deltas
            .iter()
            .map(|d| (d[3] * d[3] + d[4] * d[4] + d[5] * d[5]).sqrt())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);
        println!("Robot pose refinements ({} views):", deltas.len());
        println!(
            "  Max rotation delta: {:.4} rad ({:.2}°)",
            max_rot,
            max_rot.to_degrees()
        );
        println!(
            "  Max translation delta: {:.4}m ({:.2}mm)",
            max_trans,
            max_trans * 1000.0
        );
    }

    Ok(())
}

/// Create an isometry from Euler angles (radians) and translation.
fn make_iso(angles: (f64, f64, f64), t: (f64, f64, f64)) -> Iso3 {
    let rot = Rotation3::from_euler_angles(angles.0, angles.1, angles.2);
    Iso3::from_parts(
        Translation3::new(t.0, t.1, t.2),
        UnitQuaternion::from_rotation_matrix(&rot),
    )
}

/// Project board points through camera and return CorrespondenceView if all points visible.
fn project_view(
    camera: &PinholeCamera,
    cam_se3_target: &Iso3,
    board_pts: &[Pt3],
) -> Option<CorrespondenceView> {
    let mut points_2d = Vec::new();
    let mut points_3d = Vec::new();

    for p in board_pts {
        let p_cam = cam_se3_target.transform_point(p);
        // Only include points in front of camera
        if p_cam.z <= 0.0 {
            continue;
        }
        if let Some(uv) = camera.project_point_c(&p_cam.coords) {
            // Check if within reasonable image bounds
            if uv.x >= 0.0 && uv.x < 1280.0 && uv.y >= 0.0 && uv.y < 720.0 {
                points_2d.push(uv);
                points_3d.push(*p);
            }
        }
    }

    if points_2d.len() >= 4 {
        CorrespondenceView::new(points_3d, points_2d).ok()
    } else {
        None
    }
}
