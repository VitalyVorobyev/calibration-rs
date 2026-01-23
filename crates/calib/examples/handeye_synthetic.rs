//! Single-camera hand-eye calibration with synthetic data.
//!
//! This example demonstrates the full hand-eye calibration workflow:
//! 1. Generate synthetic robot poses and camera observations
//! 2. Initialize intrinsics (Zhang's method + distortion)
//! 3. Optimize intrinsics (bundle adjustment)
//! 4. Initialize hand-eye transform (Tsai-Lenz)
//! 5. Optimize hand-eye (bundle adjustment with robot pose priors)
//! 6. Compare results with ground truth
//!
//! Note: Hand-eye calibration is highly sensitive to robot pose diversity.
//! The linear Tsai-Lenz initialization requires diverse rotation axes to
//! work well. This example demonstrates the API workflow; for production
//! use, careful pose planning is essential.
//!
//! Run with: `cargo run -p calib --example handeye_synthetic`

use anyhow::Result;
use calib::core::{
    make_pinhole_camera, BrownConrady5, CorrespondenceView, FxFyCxCySkew, Iso3, Pt2, Pt3,
};
use calib::prelude::*;
use calib::single_cam_handeye::{
    step_handeye_init, step_handeye_optimize, step_intrinsics_init, step_intrinsics_optimize,
    HandeyeMeta, SingleCamHandeyeInput, SingleCamHandeyeProblem, SingleCamHandeyeView,
};
use nalgebra::{Rotation3, Translation3, UnitQuaternion};

fn main() -> Result<()> {
    println!("=== Single-Camera Hand-Eye Calibration (Synthetic) ===\n");

    // Ground truth camera parameters
    let k_gt = FxFyCxCySkew {
        fx: 800.0,
        fy: 780.0,
        cx: 640.0,
        cy: 360.0,
        skew: 0.0,
    };
    // Using zero distortion for cleaner synthetic data
    let dist_gt = BrownConrady5::default();
    let camera_gt = make_pinhole_camera(k_gt, dist_gt);

    // Ground truth hand-eye transform: T_G_C (gripper-to-camera)
    let handeye_gt = make_iso((0.1, -0.05, 0.02), (0.05, -0.03, 0.1));

    // Ground truth target pose in base frame: T_B_T (target 1m in front of base)
    let target_in_base_gt = make_iso((0.0, 0.0, 0.0), (0.0, 0.0, 1.0));

    println!("Ground truth:");
    println!(
        "  Camera: fx={:.1}, fy={:.1}, cx={:.1}, cy={:.1}",
        k_gt.fx, k_gt.fy, k_gt.cx, k_gt.cy
    );
    println!("  Distortion: zero (default)");
    println!(
        "  Hand-eye translation: |t|={:.3}m",
        handeye_gt.translation.vector.norm()
    );
    println!();

    // Generate calibration board points (6x5 grid, 50mm squares)
    let board_pts: Vec<Pt3> = (0..6)
        .flat_map(|i| (0..5).map(move |j| Pt3::new(i as f64 * 0.05, j as f64 * 0.05, 0.0)))
        .collect();

    // Generate robot poses with varying rotations and translations
    // Key: diverse rotation axes are important for hand-eye calibration
    let robot_poses = [
        make_iso((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        make_iso((0.3, 0.0, 0.0), (0.1, 0.0, 0.0)), // Roll rotation
        make_iso((0.0, 0.3, 0.0), (0.0, 0.1, 0.0)), // Pitch rotation
        make_iso((0.0, 0.0, 0.3), (0.0, 0.0, 0.1)), // Yaw rotation
        make_iso((0.2, 0.2, 0.0), (0.05, -0.05, 0.0)), // Combined
        make_iso((-0.2, 0.0, 0.2), (-0.05, 0.05, 0.0)), // Different axis
        make_iso((0.15, -0.15, 0.1), (0.0, -0.05, 0.05)), // More variation
    ];

    // Project observations through hand-eye transform chain
    println!("Generating {} views...", robot_poses.len());
    let views: Vec<SingleCamHandeyeView> = robot_poses
        .iter()
        .enumerate()
        .map(|(idx, robot_pose)| {
            // For EyeInHand: T_C_T = (T_B_G * T_G_C)^-1 * T_B_T
            // T_C_T transforms points from target frame to camera frame
            let cam_pose = (robot_pose * handeye_gt).inverse() * target_in_base_gt;

            // Project all points
            let points_2d: Vec<Pt2> = board_pts
                .iter()
                .map(|p| {
                    let p_cam = cam_pose.transform_point(p);
                    camera_gt
                        .project_point_c(&p_cam.coords)
                        .expect("projection failed")
                })
                .collect();

            println!("  View {}: {} points", idx, points_2d.len());

            SingleCamHandeyeView {
                obs: CorrespondenceView::new(board_pts.clone(), points_2d).unwrap(),
                meta: HandeyeMeta {
                    base_se3_gripper: *robot_pose,
                },
            }
        })
        .collect();

    let input = SingleCamHandeyeInput::new(views)?;
    println!();

    // Create calibration session
    let mut session = CalibrationSession::<SingleCamHandeyeProblem>::new();
    session.set_input(input)?;

    // Step 1: Intrinsics initialization
    println!("--- Step 1: Intrinsics Initialization ---");
    step_intrinsics_init(&mut session, None)?;
    let init_k = session.state.initial_intrinsics.as_ref().unwrap();
    println!(
        "  Intrinsics: fx={:.1}, fy={:.1}, cx={:.1}, cy={:.1}",
        init_k.fx, init_k.fy, init_k.cx, init_k.cy
    );
    println!(
        "  fx error: {:.1}%",
        100.0 * (init_k.fx - k_gt.fx).abs() / k_gt.fx
    );
    println!();

    // Step 2: Intrinsics optimization
    println!("--- Step 2: Intrinsics Optimization ---");
    step_intrinsics_optimize(&mut session, None)?;
    let opt_cam = session.state.optimized_camera.as_ref().unwrap();
    println!(
        "  Intrinsics: fx={:.1}, fy={:.1}, cx={:.1}, cy={:.1}",
        opt_cam.k.fx, opt_cam.k.fy, opt_cam.k.cx, opt_cam.k.cy
    );
    println!(
        "  Distortion: k1={:.4}, k2={:.4}",
        opt_cam.dist.k1, opt_cam.dist.k2
    );
    let reproj_err = session.state.intrinsics_reproj_error.unwrap_or(f64::NAN);
    println!("  Reprojection error: {:.4} px", reproj_err);
    println!();

    // Step 3: Hand-eye initialization
    // Note: Linear hand-eye init (Tsai-Lenz) requires diverse rotation axes.
    // If results diverge, try adding more poses with varied rotation axes.
    println!("--- Step 3: Hand-Eye Initialization ---");
    step_handeye_init(&mut session, None)?;
    let init_he = session.state.initial_handeye.as_ref().unwrap();
    let init_he_t = init_he.translation.vector.norm();
    println!(
        "  Hand-eye |t|: {:.4}m (GT: {:.4}m)",
        init_he_t,
        handeye_gt.translation.vector.norm()
    );
    if (init_he_t - handeye_gt.translation.vector.norm()).abs() > 0.1 {
        println!("  Warning: Large initialization error - may need more diverse poses");
    }
    println!();

    // Step 4: Hand-eye optimization
    println!("--- Step 4: Hand-Eye Optimization ---");
    step_handeye_optimize(&mut session, None)?;
    let reproj_err = session.state.handeye_reproj_error.unwrap_or(f64::NAN);
    println!("  Final reprojection error: {:.4} px", reproj_err);
    println!();

    // Export and compare with ground truth
    println!("--- Final Results ---");
    let export = session.export()?;
    let final_k = &export.camera.k;
    let final_d = &export.camera.dist;

    println!("Camera intrinsics:");
    println!(
        "  fx: {:.2} (GT: {:.2}, err: {:.2}%)",
        final_k.fx,
        k_gt.fx,
        100.0 * (final_k.fx - k_gt.fx).abs() / k_gt.fx
    );
    println!(
        "  fy: {:.2} (GT: {:.2}, err: {:.2}%)",
        final_k.fy,
        k_gt.fy,
        100.0 * (final_k.fy - k_gt.fy).abs() / k_gt.fy
    );
    println!(
        "  cx: {:.2} (GT: {:.2}, err: {:.2}px)",
        final_k.cx,
        k_gt.cx,
        (final_k.cx - k_gt.cx).abs()
    );
    println!(
        "  cy: {:.2} (GT: {:.2}, err: {:.2}px)",
        final_k.cy,
        k_gt.cy,
        (final_k.cy - k_gt.cy).abs()
    );

    println!("Distortion:");
    println!("  k1: {:.5} (GT: 0)", final_d.k1);
    println!("  k2: {:.5} (GT: 0)", final_d.k2);

    println!("Hand-eye transform:");
    let he_t = export.handeye.translation.vector.norm();
    let he_gt_t = handeye_gt.translation.vector.norm();
    println!(
        "  |t|: {:.4}m (GT: {:.4}m, err: {:.2}%)",
        he_t,
        he_gt_t,
        100.0 * (he_t - he_gt_t).abs() / he_gt_t
    );

    println!("Reprojection error:");
    println!("  Mean: {:.4} px", export.mean_reproj_error);
    println!("  Per-camera: {:?}", export.per_cam_reproj_errors);

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
