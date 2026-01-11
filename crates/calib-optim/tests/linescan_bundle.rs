//! Integration tests for linescan bundle adjustment.
//!
//! These tests verify end-to-end convergence with synthetic ground truth data.

use calib_core::{BrownConrady5, Camera, FxFyCxCySkew, IdentitySensor, Iso3, Pinhole, Pt3, Vec2};
use calib_optim::backend::BackendSolveOptions;
use calib_optim::params::laser_plane::LaserPlane;
use calib_optim::problems::linescan_bundle::*;
use nalgebra::{Point3, Rotation3, Translation3, Vector3};

#[test]
fn synthetic_linescan_calibration_smoke_test() {
    // Ground truth camera (simple pinhole with moderate distortion)
    let camera = Camera::new(
        Pinhole,
        BrownConrady5 {
            k1: -0.2,
            k2: 0.05,
            k3: 0.0,
            p1: 0.001,
            p2: -0.001,
            iters: 8,
        },
        IdentitySensor,
        FxFyCxCySkew {
            fx: 800.0,
            fy: 800.0,
            cx: 640.0,
            cy: 480.0,
            skew: 0.0,
        },
    );

    // Simple calibration target: 5x5 grid in target frame (Z=0)
    let grid_spacing = 0.03; // 3cm
    let mut target_points_3d = Vec::new();
    for i in 0..5 {
        for j in 0..5 {
            target_points_3d.push(Pt3::new(
                i as f64 * grid_spacing,
                j as f64 * grid_spacing,
                0.0,
            ));
        }
    }

    // Create 3 views with different poses
    let poses = [
        Iso3::from_parts(
            Translation3::new(0.0, 0.0, 0.5),
            Rotation3::from_euler_angles(0.0, 0.0, 0.0).into(),
        ),
        Iso3::from_parts(
            Translation3::new(0.05, 0.0, 0.55),
            Rotation3::from_euler_angles(0.2, -0.1, 0.0).into(),
        ),
        Iso3::from_parts(
            Translation3::new(-0.05, 0.0, 0.6),
            Rotation3::from_euler_angles(-0.2, 0.1, 0.0).into(),
        ),
    ];

    // Ground truth laser plane: tilted in camera frame, passing near target center.
    let laser_normal = Vector3::new(0.1, 0.05, 0.99);
    let laser_normal_unit = laser_normal.normalize();
    let target_center = Pt3::new(2.0 * grid_spacing, 2.0 * grid_spacing, 0.0);
    let target_center_cam = poses[0].transform_point(&target_center);
    let laser_plane = LaserPlane::new(
        laser_normal,
        -laser_normal_unit.dot(&target_center_cam.coords),
    );

    // Generate synthetic observations
    let mut views = Vec::new();
    let mut used_poses = Vec::new();
    for (view_idx, pose) in poses.iter().enumerate() {
        // Project calibration points
        let mut calib_pixels = Vec::new();
        for (pt_idx, pt) in target_points_3d.iter().enumerate() {
            let pt_camera = pose.transform_point(pt);
            if let Some(proj) = camera.project_point(&pt_camera) {
                // Add deterministic noise
                let seed = view_idx * 1000 + pt_idx;
                let noise_u = ((seed * 1103515245 + 12345) % 1000) as f64 / 1000.0 - 0.5;
                let noise_v = ((seed * 48271 + 11) % 1000) as f64 / 1000.0 - 0.5;
                calib_pixels.push(Vec2::new(proj.x + noise_u, proj.y + noise_v));
            }
        }

        // Generate laser pixels from the laser/target plane intersection line.
        let mut laser_pixels = Vec::new();
        let n_c = laser_plane.normal.as_ref();
        let n_t = pose.rotation.inverse_transform_vector(n_c);
        let d_t = n_c.dot(&pose.translation.vector) + laser_plane.distance;

        let dir = Vector3::new(n_t.y, -n_t.x, 0.0);
        let dir_norm = dir.norm();
        if dir_norm > 1e-9 {
            let dir = dir / dir_norm;
            let (x0, y0) = if n_t.x.abs() > n_t.y.abs() {
                (-d_t / n_t.x, 0.0)
            } else {
                (0.0, -d_t / n_t.y)
            };

            for i in 0..40 {
                let s = (i as f64 / 39.0) * 0.2 - 0.1;
                let pt_target = Point3::new(x0 + s * dir.x, y0 + s * dir.y, 0.0);
                let pt_camera = pose.transform_point(&pt_target);

                if let Some(proj) = camera.project_point(&pt_camera) {
                    if proj.x > 100.0 && proj.x < 1180.0 && proj.y > 100.0 && proj.y < 860.0 {
                        let seed = view_idx * 10000 + i;
                        let noise_u = ((seed * 1103515245 + 12345) % 1000) as f64 / 1000.0 - 0.5;
                        let noise_v = ((seed * 48271 + 11) % 1000) as f64 / 1000.0 - 0.5;
                        laser_pixels.push(Vec2::new(proj.x + noise_u, proj.y + noise_v));
                    }
                }
            }
        }

        if calib_pixels.len() >= 4 && laser_pixels.len() >= 3 {
            views.push(
                LinescanViewObservations::new(target_points_3d.clone(), calib_pixels, laser_pixels)
                    .unwrap(),
            );
            used_poses.push(*pose);
        }
    }

    assert!(!views.is_empty(), "failed to generate valid views for test");

    let dataset = LinescanDataset::new_single_plane(views).unwrap();

    // Create perturbed initialization (10-20% error)
    let intrinsics_init = FxFyCxCySkew {
        fx: camera.k.fx * 1.1,
        fy: camera.k.fy * 0.9,
        cx: camera.k.cx + 20.0,
        cy: camera.k.cy - 15.0,
        skew: 0.0,
    };

    let distortion_init = BrownConrady5 {
        k1: camera.dist.k1 * 1.2,
        k2: camera.dist.k2 * 0.8,
        k3: 0.0,
        p1: camera.dist.p1 * 1.1,
        p2: camera.dist.p2 * 0.9,
        iters: 8,
    };

    // Perturb poses slightly
    let poses_init: Vec<Iso3> = used_poses
        .iter()
        .map(|pose| {
            let small_rot = Rotation3::from_euler_angles(0.05, -0.03, 0.02);
            let trans_noise = Translation3::new(0.01, -0.01, 0.02);
            Iso3::from_parts(
                Translation3::from(pose.translation.vector + trans_noise.vector),
                pose.rotation * small_rot,
            )
        })
        .collect();

    // Perturb laser plane
    let plane_init = LaserPlane::new(
        laser_plane.normal.as_ref() * 1.2 + Vector3::new(0.05, -0.03, 0.02),
        laser_plane.distance * 1.1,
    );

    let initial = LinescanInit::new(
        intrinsics_init,
        distortion_init,
        poses_init,
        vec![plane_init],
    )
    .unwrap();

    // Run optimization
    let opts = LinescanSolveOptions {
        fix_k3: true,
        fix_poses: vec![0],
        ..Default::default()
    };

    let backend_opts = BackendSolveOptions {
        max_iters: 50,
        verbosity: 1,
        ..Default::default()
    };

    let result = optimize_linescan(&dataset, &initial, &opts, &backend_opts).unwrap();

    // Basic sanity checks (relaxed tolerances for smoke test)
    let fx_error = (result.camera.k.fx - camera.k.fx).abs() / camera.k.fx;
    let fy_error = (result.camera.k.fy - camera.k.fy).abs() / camera.k.fy;

    println!(
        "Intrinsics errors: fx={:.3}%, fy={:.3}%",
        fx_error * 100.0,
        fy_error * 100.0
    );

    // Should converge reasonably well (within 6% for smoke test)
    assert!(fx_error < 0.06, "fx error {:.3}% > 6%", fx_error * 100.0);
    assert!(fy_error < 0.06, "fy error {:.3}% > 6%", fy_error * 100.0);

    // Check laser plane converged (within 5 degrees)
    let normal_dot = result.planes[0].normal.dot(&laser_plane.normal);
    let normal_angle_deg = normal_dot.abs().acos().to_degrees();

    println!("Laser plane normal angle error: {:.2}°", normal_angle_deg);

    assert!(
        normal_angle_deg < 5.0,
        "normal angle {:.2}° > 5°",
        normal_angle_deg
    );

    println!("Final cost: {:.6}", result.final_cost);
    println!("✓ Linescan bundle adjustment smoke test passed");
}

#[test]
fn synthetic_linescan_line_dist_normalized_converges() {
    // Same setup as above but use LineDistNormalized residual type
    let camera = Camera::new(
        Pinhole,
        BrownConrady5 {
            k1: -0.2,
            k2: 0.05,
            k3: 0.0,
            p1: 0.001,
            p2: -0.001,
            iters: 8,
        },
        IdentitySensor,
        FxFyCxCySkew {
            fx: 800.0,
            fy: 800.0,
            cx: 640.0,
            cy: 480.0,
            skew: 0.0,
        },
    );

    let grid_spacing = 0.03;
    let mut target_points_3d = Vec::new();
    for i in 0..5 {
        for j in 0..5 {
            target_points_3d.push(Pt3::new(
                i as f64 * grid_spacing,
                j as f64 * grid_spacing,
                0.0,
            ));
        }
    }

    let poses = [
        Iso3::from_parts(
            Translation3::new(0.0, 0.0, 0.5),
            Rotation3::from_euler_angles(0.0, 0.0, 0.0).into(),
        ),
        Iso3::from_parts(
            Translation3::new(0.05, 0.0, 0.55),
            Rotation3::from_euler_angles(0.2, -0.1, 0.0).into(),
        ),
        Iso3::from_parts(
            Translation3::new(-0.05, 0.0, 0.6),
            Rotation3::from_euler_angles(-0.2, 0.1, 0.0).into(),
        ),
    ];

    let laser_normal = Vector3::new(0.1, 0.05, 0.99);
    let laser_normal_unit = laser_normal.normalize();
    let target_center = Pt3::new(2.0 * grid_spacing, 2.0 * grid_spacing, 0.0);
    let target_center_cam = poses[0].transform_point(&target_center);
    let laser_plane = LaserPlane::new(
        laser_normal,
        -laser_normal_unit.dot(&target_center_cam.coords),
    );

    let mut views = Vec::new();
    let mut used_poses = Vec::new();
    for (view_idx, pose) in poses.iter().enumerate() {
        let mut calib_pixels = Vec::new();
        for (pt_idx, pt) in target_points_3d.iter().enumerate() {
            let pt_camera = pose.transform_point(pt);
            if let Some(proj) = camera.project_point(&pt_camera) {
                let seed = view_idx * 1000 + pt_idx;
                let noise_u = ((seed * 1103515245 + 12345) % 1000) as f64 / 1000.0 - 0.5;
                let noise_v = ((seed * 48271 + 11) % 1000) as f64 / 1000.0 - 0.5;
                calib_pixels.push(Vec2::new(proj.x + noise_u, proj.y + noise_v));
            }
        }

        let mut laser_pixels = Vec::new();
        let n_c = laser_plane.normal.as_ref();
        let n_t = pose.rotation.inverse_transform_vector(n_c);
        let d_t = n_c.dot(&pose.translation.vector) + laser_plane.distance;

        let dir = Vector3::new(n_t.y, -n_t.x, 0.0);
        let dir_norm = dir.norm();
        if dir_norm > 1e-9 {
            let dir = dir / dir_norm;
            let (x0, y0) = if n_t.x.abs() > n_t.y.abs() {
                (-d_t / n_t.x, 0.0)
            } else {
                (0.0, -d_t / n_t.y)
            };

            for i in 0..40 {
                let s = (i as f64 / 39.0) * 0.2 - 0.1;
                let pt_target = Point3::new(x0 + s * dir.x, y0 + s * dir.y, 0.0);
                let pt_camera = pose.transform_point(&pt_target);

                if let Some(proj) = camera.project_point(&pt_camera) {
                    if proj.x > 100.0 && proj.x < 1180.0 && proj.y > 100.0 && proj.y < 860.0 {
                        let seed = view_idx * 10000 + i;
                        let noise_u = ((seed * 1103515245 + 12345) % 1000) as f64 / 1000.0 - 0.5;
                        let noise_v = ((seed * 48271 + 11) % 1000) as f64 / 1000.0 - 0.5;
                        laser_pixels.push(Vec2::new(proj.x + noise_u, proj.y + noise_v));
                    }
                }
            }
        }

        if calib_pixels.len() >= 4 && laser_pixels.len() >= 3 {
            views.push(
                LinescanViewObservations::new(target_points_3d.clone(), calib_pixels, laser_pixels)
                    .unwrap(),
            );
            used_poses.push(*pose);
        }
    }

    assert!(!views.is_empty(), "failed to generate valid views for test");

    let dataset = LinescanDataset::new_single_plane(views).unwrap();

    let intrinsics_init = FxFyCxCySkew {
        fx: camera.k.fx * 1.1,
        fy: camera.k.fy * 0.9,
        cx: camera.k.cx + 20.0,
        cy: camera.k.cy - 15.0,
        skew: 0.0,
    };

    let distortion_init = BrownConrady5 {
        k1: camera.dist.k1 * 1.2,
        k2: camera.dist.k2 * 0.8,
        k3: 0.0,
        p1: camera.dist.p1 * 1.1,
        p2: camera.dist.p2 * 0.9,
        iters: 8,
    };

    let poses_init: Vec<Iso3> = used_poses
        .iter()
        .map(|pose| {
            let small_rot = Rotation3::from_euler_angles(0.05, -0.03, 0.02);
            let trans_noise = Translation3::new(0.01, -0.01, 0.02);
            Iso3::from_parts(
                Translation3::from(pose.translation.vector + trans_noise.vector),
                pose.rotation * small_rot,
            )
        })
        .collect();

    let plane_init = LaserPlane::new(
        laser_plane.normal.as_ref() * 1.2 + Vector3::new(0.05, -0.03, 0.02),
        laser_plane.distance * 1.1,
    );

    let initial = LinescanInit::new(
        intrinsics_init,
        distortion_init,
        poses_init,
        vec![plane_init],
    )
    .unwrap();

    // Use LineDistNormalized residual type
    let opts = LinescanSolveOptions {
        fix_k3: true,
        fix_poses: vec![0],
        laser_residual_type: LaserResidualType::LineDistNormalized,
        ..Default::default()
    };

    let backend_opts = BackendSolveOptions {
        max_iters: 50,
        verbosity: 1,
        ..Default::default()
    };

    let result = optimize_linescan(&dataset, &initial, &opts, &backend_opts).unwrap();

    // Basic sanity checks (should converge as well as point-to-plane)
    let fx_error = (result.camera.k.fx - camera.k.fx).abs() / camera.k.fx;
    let fy_error = (result.camera.k.fy - camera.k.fy).abs() / camera.k.fy;

    println!(
        "LineDistNormalized - Intrinsics errors: fx={:.3}%, fy={:.3}%",
        fx_error * 100.0,
        fy_error * 100.0
    );

    assert!(fx_error < 0.06, "fx error {:.3}% > 6%", fx_error * 100.0);
    assert!(fy_error < 0.06, "fy error {:.3}% > 6%", fy_error * 100.0);

    let normal_dot = result.planes[0].normal.dot(&laser_plane.normal);
    let normal_angle_deg = normal_dot.abs().acos().to_degrees();

    println!(
        "LineDistNormalized - Laser plane normal angle error: {:.2}°",
        normal_angle_deg
    );

    assert!(
        normal_angle_deg < 5.0,
        "normal angle {:.2}° > 5°",
        normal_angle_deg
    );

    println!("LineDistNormalized - Final cost: {:.6}", result.final_cost);
    println!("✓ LineDistNormalized linescan calibration converged");
}

#[test]
fn compare_point_plane_vs_line_dist() {
    // Run same scenario with both residual types and compare results
    let camera = Camera::new(
        Pinhole,
        BrownConrady5 {
            k1: -0.2,
            k2: 0.05,
            k3: 0.0,
            p1: 0.001,
            p2: -0.001,
            iters: 8,
        },
        IdentitySensor,
        FxFyCxCySkew {
            fx: 800.0,
            fy: 800.0,
            cx: 640.0,
            cy: 480.0,
            skew: 0.0,
        },
    );

    let grid_spacing = 0.03;
    let mut target_points_3d = Vec::new();
    for i in 0..5 {
        for j in 0..5 {
            target_points_3d.push(Pt3::new(
                i as f64 * grid_spacing,
                j as f64 * grid_spacing,
                0.0,
            ));
        }
    }

    let poses = [
        Iso3::from_parts(
            Translation3::new(0.0, 0.0, 0.5),
            Rotation3::from_euler_angles(0.0, 0.0, 0.0).into(),
        ),
        Iso3::from_parts(
            Translation3::new(0.05, 0.0, 0.55),
            Rotation3::from_euler_angles(0.2, -0.1, 0.0).into(),
        ),
        Iso3::from_parts(
            Translation3::new(-0.05, 0.0, 0.6),
            Rotation3::from_euler_angles(-0.2, 0.1, 0.0).into(),
        ),
    ];

    let laser_normal = Vector3::new(0.1, 0.05, 0.99);
    let laser_normal_unit = laser_normal.normalize();
    let target_center = Pt3::new(2.0 * grid_spacing, 2.0 * grid_spacing, 0.0);
    let target_center_cam = poses[0].transform_point(&target_center);
    let laser_plane = LaserPlane::new(
        laser_normal,
        -laser_normal_unit.dot(&target_center_cam.coords),
    );

    let mut views = Vec::new();
    let mut used_poses = Vec::new();
    for (view_idx, pose) in poses.iter().enumerate() {
        let mut calib_pixels = Vec::new();
        for (pt_idx, pt) in target_points_3d.iter().enumerate() {
            let pt_camera = pose.transform_point(pt);
            if let Some(proj) = camera.project_point(&pt_camera) {
                let seed = view_idx * 1000 + pt_idx;
                let noise_u = ((seed * 1103515245 + 12345) % 1000) as f64 / 1000.0 - 0.5;
                let noise_v = ((seed * 48271 + 11) % 1000) as f64 / 1000.0 - 0.5;
                calib_pixels.push(Vec2::new(proj.x + noise_u, proj.y + noise_v));
            }
        }

        let mut laser_pixels = Vec::new();
        let n_c = laser_plane.normal.as_ref();
        let n_t = pose.rotation.inverse_transform_vector(n_c);
        let d_t = n_c.dot(&pose.translation.vector) + laser_plane.distance;

        let dir = Vector3::new(n_t.y, -n_t.x, 0.0);
        let dir_norm = dir.norm();
        if dir_norm > 1e-9 {
            let dir = dir / dir_norm;
            let (x0, y0) = if n_t.x.abs() > n_t.y.abs() {
                (-d_t / n_t.x, 0.0)
            } else {
                (0.0, -d_t / n_t.y)
            };

            for i in 0..40 {
                let s = (i as f64 / 39.0) * 0.2 - 0.1;
                let pt_target = Point3::new(x0 + s * dir.x, y0 + s * dir.y, 0.0);
                let pt_camera = pose.transform_point(&pt_target);

                if let Some(proj) = camera.project_point(&pt_camera) {
                    if proj.x > 100.0 && proj.x < 1180.0 && proj.y > 100.0 && proj.y < 860.0 {
                        let seed = view_idx * 10000 + i;
                        let noise_u = ((seed * 1103515245 + 12345) % 1000) as f64 / 1000.0 - 0.5;
                        let noise_v = ((seed * 48271 + 11) % 1000) as f64 / 1000.0 - 0.5;
                        laser_pixels.push(Vec2::new(proj.x + noise_u, proj.y + noise_v));
                    }
                }
            }
        }

        if calib_pixels.len() >= 4 && laser_pixels.len() >= 3 {
            views.push(
                LinescanViewObservations::new(target_points_3d.clone(), calib_pixels, laser_pixels)
                    .unwrap(),
            );
            used_poses.push(*pose);
        }
    }

    let dataset = LinescanDataset::new_single_plane(views).unwrap();

    let intrinsics_init = FxFyCxCySkew {
        fx: camera.k.fx * 1.1,
        fy: camera.k.fy * 0.9,
        cx: camera.k.cx + 20.0,
        cy: camera.k.cy - 15.0,
        skew: 0.0,
    };

    let distortion_init = BrownConrady5 {
        k1: camera.dist.k1 * 1.2,
        k2: camera.dist.k2 * 0.8,
        k3: 0.0,
        p1: camera.dist.p1 * 1.1,
        p2: camera.dist.p2 * 0.9,
        iters: 8,
    };

    let poses_init: Vec<Iso3> = used_poses
        .iter()
        .map(|pose| {
            let small_rot = Rotation3::from_euler_angles(0.05, -0.03, 0.02);
            let trans_noise = Translation3::new(0.01, -0.01, 0.02);
            Iso3::from_parts(
                Translation3::from(pose.translation.vector + trans_noise.vector),
                pose.rotation * small_rot,
            )
        })
        .collect();

    let plane_init = LaserPlane::new(
        laser_plane.normal.as_ref() * 1.2 + Vector3::new(0.05, -0.03, 0.02),
        laser_plane.distance * 1.1,
    );

    let initial = LinescanInit::new(
        intrinsics_init,
        distortion_init,
        poses_init,
        vec![plane_init],
    )
    .unwrap();

    let backend_opts = BackendSolveOptions {
        max_iters: 50,
        verbosity: 0, // Quiet for comparison
        ..Default::default()
    };

    // Run with PointToPlane
    let opts_point = LinescanSolveOptions {
        fix_k3: true,
        fix_poses: vec![0],
        laser_residual_type: LaserResidualType::PointToPlane,
        ..Default::default()
    };

    let result_point = optimize_linescan(&dataset, &initial, &opts_point, &backend_opts).unwrap();

    // Run with LineDistNormalized
    let opts_line = LinescanSolveOptions {
        fix_k3: true,
        fix_poses: vec![0],
        laser_residual_type: LaserResidualType::LineDistNormalized,
        ..Default::default()
    };

    let result_line = optimize_linescan(&dataset, &initial, &opts_line, &backend_opts).unwrap();

    // Compare results
    println!("\n=== Comparison: PointToPlane vs LineDistNormalized ===");

    let fx_err_point = (result_point.camera.k.fx - camera.k.fx).abs() / camera.k.fx;
    let fx_err_line = (result_line.camera.k.fx - camera.k.fx).abs() / camera.k.fx;

    println!(
        "fx error: PointToPlane={:.3}%, LineDistNormalized={:.3}%",
        fx_err_point * 100.0,
        fx_err_line * 100.0
    );

    let fy_err_point = (result_point.camera.k.fy - camera.k.fy).abs() / camera.k.fy;
    let fy_err_line = (result_line.camera.k.fy - camera.k.fy).abs() / camera.k.fy;

    println!(
        "fy error: PointToPlane={:.3}%, LineDistNormalized={:.3}%",
        fy_err_point * 100.0,
        fy_err_line * 100.0
    );

    let normal_angle_point = result_point.planes[0]
        .normal
        .dot(&laser_plane.normal)
        .abs()
        .acos()
        .to_degrees();
    let normal_angle_line = result_line.planes[0]
        .normal
        .dot(&laser_plane.normal)
        .abs()
        .acos()
        .to_degrees();

    println!(
        "Plane normal error: PointToPlane={:.2}°, LineDistNormalized={:.2}°",
        normal_angle_point, normal_angle_line
    );

    println!(
        "Final cost: PointToPlane={:.6}, LineDistNormalized={:.6}",
        result_point.final_cost, result_line.final_cost
    );

    // Both should converge to acceptable accuracy
    assert!(fx_err_point < 0.06 && fx_err_line < 0.06);
    assert!(fy_err_point < 0.06 && fy_err_line < 0.06);
    assert!(normal_angle_point < 5.0 && normal_angle_line < 5.0);

    println!("✓ Both residual types converged successfully");
}
