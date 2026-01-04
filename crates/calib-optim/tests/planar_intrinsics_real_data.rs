//! Integration test for planar intrinsics optimization using real stereo chessboard data.
//!
//! This test validates the full pipeline:
//! 1. Load real corner detections from stereo_linear.json
//! 2. Initialize intrinsics using Zhang's method (calib-linear)
//! 3. Optimize intrinsics + distortion + poses (calib-optim)
//! 4. Verify reprojection error improves after optimization

use calib_core::{
    test_utils::{pixel_from_normalized, undistort_pixel_normalized, CalibrationView},
    BrownConrady5, DistortionModel, Mat3, Pt2, Pt3, Real, Vec2,
};
use calib_linear::{HomographySolver, PlanarIntrinsicsLinearInit};
use calib_optim::ir::RobustLoss;
use calib_optim::params::distortion::BrownConrady5Params;
use calib_optim::params::intrinsics::Intrinsics4;
use calib_optim::problems::planar_intrinsics::{
    optimize_planar_intrinsics, PlanarDataset, PlanarIntrinsicsInit, PlanarIntrinsicsSolveOptions,
    PlanarViewObservations,
};
use calib_optim::BackendSolveOptions;
use nalgebra::Isometry3;
use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct StereoData {
    board: BoardSpec,
    intrinsics: IntrinsicsPair,
    distortion: DistortionPair,
    views: Vec<CalibrationView>,
}

#[derive(Debug, Deserialize)]
struct BoardSpec {
    square_size: Real,
}

#[derive(Debug, Deserialize)]
struct IntrinsicsPair {
    left: [[Real; 3]; 3],
    right: [[Real; 3]; 3],
}

#[derive(Debug, Deserialize)]
struct DistortionPair {
    left: [Real; 5],
    right: [Real; 5],
}

fn load_data() -> StereoData {
    // Load from calib-linear test data
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("calib-linear")
        .join("tests")
        .join("data")
        .join("stereo_linear.json");
    let contents = fs::read_to_string(&path).expect("read stereo_linear.json");
    let mut data: StereoData = serde_json::from_str(&contents).expect("parse stereo_linear.json");
    data.views.sort_by_key(|v| v.view_index);
    data
}

fn mat3_from_array(a: &[[Real; 3]; 3]) -> Mat3 {
    Mat3::from_row_slice(&[
        a[0][0], a[0][1], a[0][2], a[1][0], a[1][1], a[1][2], a[2][0], a[2][1], a[2][2],
    ])
}

fn distortion_from_array(d: [Real; 5]) -> BrownConrady5<Real> {
    BrownConrady5 {
        k1: d[0],
        k2: d[1],
        p1: d[2],
        p2: d[3],
        k3: d[4],
        iters: 8,
    }
}

fn board_point_2d(i: usize, j: usize, square: Real) -> Pt2 {
    Pt2::new(i as Real * square, j as Real * square)
}

fn board_point_3d(i: usize, j: usize, square: Real) -> Pt3 {
    Pt3::new(i as Real * square, j as Real * square, 0.0)
}

/// Compute reprojection error for a set of observations.
fn compute_reprojection_error(
    views: &[PlanarViewObservations],
    intrinsics: &Intrinsics4,
    distortion: &BrownConrady5Params,
    poses: &[Isometry3<Real>],
) -> (Real, Real) {
    let k = intrinsics.to_core();
    let dist = distortion.to_core();

    let mut errors = Vec::new();

    for (view, pose) in views.iter().zip(poses.iter()) {
        for (pw, uv) in view.points_3d.iter().zip(&view.points_2d) {
            // Transform to camera frame
            let pc = pose.transform_point(&nalgebra::Point3::new(pw.x, pw.y, pw.z));

            // Project to normalized coordinates
            if pc.z <= 1e-6 {
                continue;
            }
            let x_norm = pc.x / pc.z;
            let y_norm = pc.y / pc.z;

            // Apply distortion
            let distorted = dist.distort(&Vec2::new(x_norm, y_norm));

            // Apply intrinsics
            let u_proj = k.fx * distorted.x + k.cx;
            let v_proj = k.fy * distorted.y + k.cy;

            // Compute pixel error
            let du = uv.x - u_proj;
            let dv = uv.y - v_proj;
            let err = (du * du + dv * dv).sqrt();
            errors.push(err);
        }
    }

    let mean = errors.iter().sum::<Real>() / errors.len() as Real;
    let max = errors.iter().cloned().fold(0.0, Real::max);
    (mean, max)
}

#[test]
fn planar_intrinsics_real_data_improves_reprojection() {
    let data = load_data();
    let board = &data.board;

    // Test both left and right cameras
    for (side, k_arr, d_arr) in [
        ("left", data.intrinsics.left, data.distortion.left),
        ("right", data.intrinsics.right, data.distortion.right),
    ] {
        println!("\n=== Testing {side} camera ===");

        let k_gt = mat3_from_array(&k_arr);
        let dist_gt = distortion_from_array(d_arr);

        // Step 1: Prepare undistorted observations for Zhang's method
        let mut homographies = Vec::new();
        let mut undistorted_views = Vec::new();

        for view in &data.views {
            let det = if side == "left" {
                &view.left
            } else {
                &view.right
            };

            let mut world = Vec::new();
            let mut undist_pixels = Vec::new();

            for c in &det.corners {
                let i = c[0] as usize;
                let j = c[1] as usize;
                let pixel = Pt2::new(c[2], c[3]);

                // Undistort using ground truth (simulates having good corner detection)
                let undist_norm = undistort_pixel_normalized(pixel, &k_gt, &dist_gt);
                let undist_pixel = pixel_from_normalized(undist_norm, &k_gt);

                world.push(board_point_2d(i, j, board.square_size));
                undist_pixels.push(undist_pixel);
            }

            let h = HomographySolver::dlt(&world, &undist_pixels).expect("homography");
            homographies.push(h);
            undistorted_views.push((world, undist_pixels));
        }

        // Step 2: Initialize intrinsics using Zhang's method (linear)
        let linear_init =
            PlanarIntrinsicsLinearInit::from_homographies(&homographies).expect("zhang init");

        println!(
            "Linear init: fx={:.2}, fy={:.2}, cx={:.2}, cy={:.2}",
            linear_init.fx, linear_init.fy, linear_init.cx, linear_init.cy
        );

        // Step 3: Prepare dataset for non-linear optimization
        let k_init = Mat3::new(
            linear_init.fx,
            linear_init.skew,
            linear_init.cx,
            0.0,
            linear_init.fy,
            linear_init.cy,
            0.0,
            0.0,
            1.0,
        );

        let mut nl_views = Vec::new();
        let mut init_poses = Vec::new();

        for (h, (world, undist_pixels)) in homographies.iter().zip(&undistorted_views) {
            // Estimate initial pose from homography
            let pose =
                calib_linear::PlanarPoseSolver::from_homography(&k_init, h).expect("planar pose");

            // Use original distorted pixels for optimization
            let points_3d: Vec<Pt3> = world
                .iter()
                .map(|p| {
                    board_point_3d(
                        (p.x / board.square_size) as usize,
                        (p.y / board.square_size) as usize,
                        board.square_size,
                    )
                })
                .collect();

            let points_2d: Vec<Vec2> = undist_pixels.iter().map(|p| Vec2::new(p.x, p.y)).collect();

            nl_views.push(
                PlanarViewObservations::new(points_3d, points_2d)
                    .expect("planar view observations"),
            );
            init_poses.push(pose);
        }

        let dataset = PlanarDataset::new(nl_views.clone()).expect("planar dataset");

        let init = PlanarIntrinsicsInit {
            intrinsics: Intrinsics4 {
                fx: linear_init.fx,
                fy: linear_init.fy,
                cx: linear_init.cx,
                cy: linear_init.cy,
            },
            distortion: BrownConrady5Params::zeros(), // Start with zero distortion
            poses: init_poses.clone(),
        };

        // Compute initial reprojection error
        let (init_mean, init_max) =
            compute_reprojection_error(&nl_views, &init.intrinsics, &init.distortion, &init_poses);
        println!(
            "Initial reprojection error: mean={:.3} px, max={:.3} px",
            init_mean, init_max
        );

        // Step 4: Run non-linear optimization
        let opts = PlanarIntrinsicsSolveOptions {
            robust_loss: RobustLoss::Huber { scale: 2.0 },
            ..Default::default()
        };

        let backend_opts = BackendSolveOptions {
            max_iters: 50,
            verbosity: 0,
            ..Default::default()
        };

        let result = optimize_planar_intrinsics(dataset, init, opts, backend_opts)
            .expect("optimization failed");

        println!(
            "Optimized: fx={:.2}, fy={:.2}, cx={:.2}, cy={:.2}",
            result.camera.k.fx, result.camera.k.fy, result.camera.k.cx, result.camera.k.cy
        );
        println!(
            "Distortion: k1={:.6}, k2={:.6}, k3={:.6}, p1={:.6}, p2={:.6}",
            result.camera.dist.k1,
            result.camera.dist.k2,
            result.camera.dist.k3,
            result.camera.dist.p1,
            result.camera.dist.p2
        );

        // Compute final reprojection error
        let opt_intrinsics = Intrinsics4 {
            fx: result.camera.k.fx,
            fy: result.camera.k.fy,
            cx: result.camera.k.cx,
            cy: result.camera.k.cy,
        };
        let opt_distortion = BrownConrady5Params::from_core(&result.camera.dist);

        let (final_mean, final_max) =
            compute_reprojection_error(&nl_views, &opt_intrinsics, &opt_distortion, &result.poses);
        println!(
            "Final reprojection error: mean={:.3} px, max={:.3} px",
            final_mean, final_max
        );

        // Verify improvement (allow for modest improvement since initialization is already good)
        assert!(
            final_mean < init_mean * 0.95,
            "{side}: Mean reprojection error should improve: init={:.3}, final={:.3}",
            init_mean,
            final_mean
        );
        println!(
            "✓ Reprojection error improved by {:.1}%",
            (1.0 - final_mean / init_mean) * 100.0
        );

        // Verify reasonable final error
        assert!(
            final_mean < 2.0,
            "{side}: Final mean error too large: {:.3} px",
            final_mean
        );
        assert!(
            final_max < 5.0,
            "{side}: Final max error too large: {:.3} px",
            final_max
        );

        println!("✓ All checks passed for {side} camera\n");
    }
}

#[test]
fn planar_intrinsics_parameter_fixing_works() {
    let data = load_data();
    let board = &data.board;

    // Use only left camera for this test
    let k_gt = mat3_from_array(&data.intrinsics.left);
    let dist_gt = distortion_from_array(data.distortion.left);

    println!("\n=== Testing parameter fixing ===");

    // Prepare data (same as above but condensed)
    let mut homographies = Vec::new();
    let mut nl_views = Vec::new();
    let mut init_poses = Vec::new();

    for view in &data.views {
        let det = &view.left;

        let mut world = Vec::new();
        let mut undist_pixels = Vec::new();
        let mut points_3d = Vec::new();
        let mut points_2d = Vec::new();

        for c in &det.corners {
            let i = c[0] as usize;
            let j = c[1] as usize;
            let pixel = Pt2::new(c[2], c[3]);

            let undist_norm = undistort_pixel_normalized(pixel, &k_gt, &dist_gt);
            let undist_pixel = pixel_from_normalized(undist_norm, &k_gt);

            world.push(board_point_2d(i, j, board.square_size));
            undist_pixels.push(undist_pixel);
            points_3d.push(board_point_3d(i, j, board.square_size));
            points_2d.push(Vec2::new(undist_pixel.x, undist_pixel.y));
        }

        let h = HomographySolver::dlt(&world, &undist_pixels).expect("homography");
        homographies.push(h);

        nl_views.push(
            PlanarViewObservations::new(points_3d, points_2d).expect("planar view observations"),
        );
    }

    let linear_init =
        PlanarIntrinsicsLinearInit::from_homographies(&homographies).expect("zhang init");

    let k_init = Mat3::new(
        linear_init.fx,
        linear_init.skew,
        linear_init.cx,
        0.0,
        linear_init.fy,
        linear_init.cy,
        0.0,
        0.0,
        1.0,
    );

    for (h, _) in homographies.iter().zip(&nl_views) {
        let pose =
            calib_linear::PlanarPoseSolver::from_homography(&k_init, h).expect("planar pose");
        init_poses.push(pose);
    }

    let dataset = PlanarDataset::new(nl_views).expect("planar dataset");

    // Test 1: Fix tangential distortion (p1, p2)
    println!("\nTest 1: Fixing tangential distortion (p1, p2)");

    let init = PlanarIntrinsicsInit {
        intrinsics: Intrinsics4 {
            fx: linear_init.fx,
            fy: linear_init.fy,
            cx: linear_init.cx,
            cy: linear_init.cy,
        },
        distortion: BrownConrady5Params::zeros(),
        poses: init_poses.clone(),
    };

    let opts = PlanarIntrinsicsSolveOptions {
        fix_p1: true, // Fix tangential
        fix_p2: true,
        robust_loss: RobustLoss::None,
        ..Default::default()
    };

    let result =
        optimize_planar_intrinsics(dataset.clone(), init, opts, BackendSolveOptions::default())
            .expect("optimization with fixed p1, p2");

    assert_eq!(result.camera.dist.p1, 0.0, "p1 should remain fixed at 0.0");
    assert_eq!(result.camera.dist.p2, 0.0, "p2 should remain fixed at 0.0");
    println!(
        "✓ Tangential distortion stayed fixed: p1={}, p2={}",
        result.camera.dist.p1, result.camera.dist.p2
    );

    // Radial should have changed
    assert!(
        result.camera.dist.k1.abs() > 1e-6 || result.camera.dist.k2.abs() > 1e-6,
        "Radial distortion should be optimized"
    );
    println!(
        "✓ Radial distortion optimized: k1={:.6}, k2={:.6}",
        result.camera.dist.k1, result.camera.dist.k2
    );

    // Test 2: Fix k3 (default behavior)
    println!("\nTest 2: Fixing k3 (default)");

    let init2 = PlanarIntrinsicsInit {
        intrinsics: Intrinsics4 {
            fx: linear_init.fx,
            fy: linear_init.fy,
            cx: linear_init.cx,
            cy: linear_init.cy,
        },
        distortion: BrownConrady5Params::zeros(),
        poses: init_poses.clone(),
    };

    let opts2 = PlanarIntrinsicsSolveOptions::default(); // fix_k3 is true by default

    let result2 = optimize_planar_intrinsics(
        dataset.clone(),
        init2,
        opts2,
        BackendSolveOptions::default(),
    )
    .expect("optimization with default k3 fixed");

    assert_eq!(result2.camera.dist.k3, 0.0, "k3 should be fixed by default");
    println!(
        "✓ k3 stayed fixed at default: k3={}",
        result2.camera.dist.k3
    );

    // Test 3: Fix intrinsics, optimize only distortion
    println!("\nTest 3: Fixing intrinsics (fx, fy), optimizing distortion");

    let init3 = PlanarIntrinsicsInit {
        intrinsics: Intrinsics4 {
            fx: linear_init.fx,
            fy: linear_init.fy,
            cx: linear_init.cx,
            cy: linear_init.cy,
        },
        distortion: BrownConrady5Params::zeros(),
        poses: init_poses.clone(),
    };

    let opts3 = PlanarIntrinsicsSolveOptions {
        fix_fx: true,
        fix_fy: true,
        robust_loss: RobustLoss::None,
        ..Default::default()
    };

    let result3 = optimize_planar_intrinsics(
        dataset,
        init3.clone(),
        opts3,
        BackendSolveOptions::default(),
    )
    .expect("optimization with fixed fx, fy");

    assert!(
        (result3.camera.k.fx - init3.intrinsics.fx).abs() < 1e-6,
        "fx should remain fixed"
    );
    assert!(
        (result3.camera.k.fy - init3.intrinsics.fy).abs() < 1e-6,
        "fy should remain fixed"
    );
    println!(
        "✓ Intrinsics stayed fixed: fx={:.2}, fy={:.2}",
        result3.camera.k.fx, result3.camera.k.fy
    );
    println!(
        "✓ Distortion optimized: k1={:.6}, k2={:.6}",
        result3.camera.dist.k1, result3.camera.dist.k2
    );

    println!("\n✓ All parameter fixing tests passed\n");
}

// ============================================================================
// NEW TEST: Full calibration pipeline with iterative linear initialization
// ============================================================================

use calib_linear::iterative_intrinsics::{
    IterativeCalibView, IterativeIntrinsicsOptions, IterativeIntrinsicsSolver,
};
use calib_linear::DistortionFitOptions;

/// Test the complete calibration pipeline WITHOUT using ground truth distortion.
///
/// Pipeline:
/// 1. Iterative linear init: Estimate K + distortion from raw pixels (calib-linear)
/// 2. Non-linear refinement: Optimize all parameters via bundle adjustment (calib-optim)
///
/// This demonstrates a realistic calibration workflow from corner detections to
/// final calibrated camera parameters.
#[test]
fn planar_intrinsics_with_iterative_linear_init() {
    let data = load_data();
    let board = &data.board;

    // Test both cameras
    for (side, k_arr, d_arr) in [
        ("left", data.intrinsics.left, data.distortion.left),
        ("right", data.intrinsics.right, data.distortion.right),
    ] {
        println!("\n=== Testing {side} camera with iterative init ===");

        let k_gt = mat3_from_array(&k_arr);
        let dist_gt = distortion_from_array(d_arr);

        // Step 1: Iterative linear initialization (NO ground truth used)
        println!("Step 1: Iterative linear initialization...");

        let views_iter: Vec<IterativeCalibView> = data
            .views
            .iter()
            .map(|view| {
                let det = if side == "left" {
                    &view.left
                } else {
                    &view.right
                };

                let board_points: Vec<Pt2> = det
                    .corners
                    .iter()
                    .map(|c| board_point_2d(c[0] as usize, c[1] as usize, board.square_size))
                    .collect();

                let pixel_points: Vec<Pt2> =
                    det.corners.iter().map(|c| Pt2::new(c[2], c[3])).collect();

                IterativeCalibView::new(board_points, pixel_points)
            })
            .collect();

        let iter_opts = IterativeIntrinsicsOptions {
            iterations: 2,
            distortion_opts: DistortionFitOptions {
                fix_k3: true,
                fix_tangential: false,
                iters: 8,
            },
        };

        let iter_result = IterativeIntrinsicsSolver::estimate(&views_iter, iter_opts)
            .expect("iterative linear init");

        println!(
            "  Initial K: fx={:.1}, fy={:.1}, cx={:.1}, cy={:.1}",
            iter_result.intrinsics.fx,
            iter_result.intrinsics.fy,
            iter_result.intrinsics.cx,
            iter_result.intrinsics.cy
        );
        println!(
            "  Initial distortion: k1={:.4}, k2={:.4}, p1={:.4}, p2={:.4}",
            iter_result.distortion.k1,
            iter_result.distortion.k2,
            iter_result.distortion.p1,
            iter_result.distortion.p2
        );

        // Step 2: Prepare optimization dataset
        println!("Step 2: Non-linear refinement...");

        let k_iter = iter_result.intrinsics.k_matrix();

        let mut views_optim = Vec::new();
        let mut init_poses = Vec::new();

        for view in &data.views {
            let det = if side == "left" {
                &view.left
            } else {
                &view.right
            };

            // Compute homography for pose initialization
            let board_pts: Vec<Pt2> = det
                .corners
                .iter()
                .map(|c| board_point_2d(c[0] as usize, c[1] as usize, board.square_size))
                .collect();

            let pixel_pts: Vec<Pt2> = det.corners.iter().map(|c| Pt2::new(c[2], c[3])).collect();

            let h =
                calib_linear::HomographySolver::dlt(&board_pts, &pixel_pts).expect("homography");
            let pose = calib_linear::PlanarPoseSolver::from_homography(&k_iter, &h).expect("pose");
            init_poses.push(pose);

            let points_3d: Vec<Pt3> = det
                .corners
                .iter()
                .map(|c| board_point_3d(c[0] as usize, c[1] as usize, board.square_size))
                .collect();

            let points_2d: Vec<Vec2> = det.corners.iter().map(|c| Vec2::new(c[2], c[3])).collect();

            views_optim.push(PlanarViewObservations::new(points_3d, points_2d).expect("view"));
        }

        let dataset = PlanarDataset::new(views_optim.clone()).expect("dataset");

        // Initialize from iterative result
        let init = PlanarIntrinsicsInit {
            intrinsics: Intrinsics4 {
                fx: iter_result.intrinsics.fx,
                fy: iter_result.intrinsics.fy,
                cx: iter_result.intrinsics.cx,
                cy: iter_result.intrinsics.cy,
            },
            distortion: BrownConrady5Params::from_core(&iter_result.distortion),
            poses: init_poses.clone(),
        };

        let optim_opts = PlanarIntrinsicsSolveOptions {
            robust_loss: RobustLoss::None,
            ..Default::default()
        };

        let backend_opts = BackendSolveOptions {
            max_iters: 100,
            verbosity: 0,
            ..Default::default()
        };

        let result = optimize_planar_intrinsics(dataset, init, optim_opts, backend_opts)
            .expect("optimization");

        println!(
            "  Final K: fx={:.1}, fy={:.1}, cx={:.1}, cy={:.1}",
            result.camera.k.fx, result.camera.k.fy, result.camera.k.cx, result.camera.k.cy
        );
        println!(
            "  Final distortion: k1={:.4}, k2={:.4}, p1={:.4}, p2={:.4}",
            result.camera.dist.k1,
            result.camera.dist.k2,
            result.camera.dist.p1,
            result.camera.dist.p2
        );

        // Step 3: Compute reprojection error
        let opt_intrinsics = Intrinsics4 {
            fx: result.camera.k.fx,
            fy: result.camera.k.fy,
            cx: result.camera.k.cx,
            cy: result.camera.k.cy,
        };
        let opt_distortion_params = BrownConrady5Params::from_core(&result.camera.dist);
        let (mean_err, max_err) = compute_reprojection_error(
            &views_optim,
            &opt_intrinsics,
            &opt_distortion_params,
            &result.poses,
        );

        println!(
            "  Reprojection error: mean={:.3}px, max={:.3}px",
            mean_err, max_err
        );

        // Step 4: Validate final results
        let fx_gt = k_gt[(0, 0)];
        let fy_gt = k_gt[(1, 1)];
        let cx_gt = k_gt[(0, 2)];
        let cy_gt = k_gt[(1, 2)];

        let fx_err_pct = (result.camera.k.fx - fx_gt).abs() / fx_gt * 100.0;
        let fy_err_pct = (result.camera.k.fy - fy_gt).abs() / fy_gt * 100.0;
        let cx_err = (result.camera.k.cx - cx_gt).abs();
        let cy_err = (result.camera.k.cy - cy_gt).abs();

        println!(
            "  Final errors: fx={:.2}%, fy={:.2}%, cx={:.2}px, cy={:.2}px",
            fx_err_pct, fy_err_pct, cx_err, cy_err
        );

        // Validate: After non-linear refinement, should achieve good accuracy
        assert!(
            mean_err < 1.5,
            "{side}: mean reprojection error {mean_err:.3}px too large"
        );
        assert!(
            max_err < 3.0,
            "{side}: max reprojection error {max_err:.3}px too large"
        );

        // Intrinsics should be close to ground truth after optimization
        assert!(
            fx_err_pct < 5.0,
            "{side}: fx error {fx_err_pct:.2}% too large"
        );
        assert!(
            fy_err_pct < 5.0,
            "{side}: fy error {fy_err_pct:.2}% too large"
        );
        assert!(cx_err < 10.0, "{side}: cx error {cx_err:.2}px too large");
        assert!(cy_err < 10.0, "{side}: cy error {cy_err:.2}px too large");

        // Distortion signs should match
        assert!(
            result.camera.dist.k1.signum() == dist_gt.k1.signum(),
            "{side}: k1 sign mismatch"
        );

        println!("✓ {side} camera calibration successful!");
    }

    println!("\n✓ Full pipeline test passed (iterative init → optimization)\n");
}
