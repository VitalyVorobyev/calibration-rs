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

    // Ground truth laser plane: tilted in camera frame
    let laser_plane = LaserPlane::new(Vector3::new(0.1, 0.05, 0.99), -0.4);

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

    // Generate synthetic observations
    let mut views = Vec::new();
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

        // Generate laser pixels on the plane
        let mut laser_pixels = Vec::new();
        let n = laser_plane.normal.as_ref();
        let tangent1 = Vector3::new(1.0, 0.0, 0.0).cross(n).normalize();
        let tangent2 = n.cross(&tangent1);

        for i in 0..20 {
            let u = (i as f64 / 20.0) * 0.4 - 0.2;
            let v = 0.0;
            // Point on plane: -d*n + u*t1 + v*t2
            let pt_on_plane = Point3::from(-laser_plane.distance * n + u * tangent1 + v * tangent2);

            if let Some(proj) = camera.project_point(&pt_on_plane) {
                if proj.x > 100.0 && proj.x < 1180.0 && proj.y > 100.0 && proj.y < 860.0 {
                    let seed = view_idx * 10000 + i;
                    let noise_u = ((seed * 1103515245 + 12345) % 1000) as f64 / 1000.0 - 0.5;
                    let noise_v = ((seed * 48271 + 11) % 1000) as f64 / 1000.0 - 0.5;
                    laser_pixels.push(Vec2::new(proj.x + noise_u, proj.y + noise_v));
                }
            }
        }

        if calib_pixels.len() >= 4 && laser_pixels.len() >= 3 {
            views.push(
                LinescanViewObservations::new(target_points_3d.clone(), calib_pixels, laser_pixels)
                    .unwrap(),
            );
        }
    }

    assert!(!views.is_empty(), "failed to generate valid views for test");

    let dataset = LinescanDataset::new_single_plane(views).unwrap();

    // Create perturbed initialization (10-20% error)
    let intrinsics_init = calib_optim::params::intrinsics::Intrinsics4 {
        fx: camera.k.fx * 1.1,
        fy: camera.k.fy * 0.9,
        cx: camera.k.cx + 20.0,
        cy: camera.k.cy - 15.0,
    };

    let distortion_init = calib_optim::params::distortion::BrownConrady5Params {
        k1: camera.dist.k1 * 1.2,
        k2: camera.dist.k2 * 0.8,
        k3: 0.0,
        p1: camera.dist.p1 * 1.1,
        p2: camera.dist.p2 * 0.9,
    };

    // Perturb poses slightly
    let poses_init: Vec<Iso3> = poses
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
