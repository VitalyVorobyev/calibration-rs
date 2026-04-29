//! Synthetic ground-truth test for `optimize_rig_laserline`.
//!
//! Mirrors `laserline_bundle.rs` but for the rig-level joint solve.
//! Synthesizes a 2-camera rig with known intrinsics, extrinsics, and per-camera
//! laser planes. Projects laser-line points into each camera, builds a
//! `RigLaserlineDataset`, and verifies that `optimize_rig_laserline` recovers
//! the planes within the specified tolerances.

use nalgebra::{Isometry3, Rotation3, Translation3, Unit, Vector3};
use vision_calibration_core::{
    BrownConrady5, Camera, CorrespondenceView, FxFyCxCySkew, Pinhole, Pt3, Real, ScheimpflugParams,
};
use vision_calibration_optim::{
    BackendSolveOptions, LaserPlane, RigLaserlineDataset, RigLaserlineSolveOptions,
    RigLaserlineUpstream, RigLaserlineView, optimize_rig_laserline,
};

fn make_target_points() -> Vec<Pt3> {
    let mut pts = Vec::new();
    for y in -2i32..=2 {
        for x in -3i32..=3 {
            pts.push(Pt3::new(x as Real * 0.02, y as Real * 0.02, 0.0));
        }
    }
    pts
}

#[test]
fn stereo_rig_laserline_recovers_planes() {
    // Ground-truth camera model (same for both cameras, zero distortion for clarity).
    let intr = FxFyCxCySkew {
        fx: 600.0,
        fy: 600.0,
        cx: 320.0,
        cy: 240.0,
        skew: 0.0,
    };
    let dist = BrownConrady5 {
        k1: 0.0,
        k2: 0.0,
        k3: 0.0,
        p1: 0.0,
        p2: 0.0,
        iters: 5,
    };
    let sensor = ScheimpflugParams::default();
    let cam = Camera::new(Pinhole, dist, sensor.compile(), intr);

    // Rig geometry: cam0 at origin, cam1 displaced along X.
    let cam0_to_rig = Isometry3::identity();
    let cam1_to_rig = Isometry3::from_parts(
        Translation3::new(0.15, 0.0, 0.0),
        Rotation3::identity().into(),
    );
    let cam_to_rig = [cam0_to_rig, cam1_to_rig];

    // Per-view rig poses (target at varying depths).
    let rig_se3_target = vec![
        Isometry3::from_parts(
            Translation3::new(0.0, 0.0, 0.30),
            Rotation3::identity().into(),
        ),
        Isometry3::from_parts(
            Translation3::new(0.0, 0.0, 0.32),
            Rotation3::identity().into(),
        ),
        Isometry3::from_parts(
            Translation3::new(0.0, 0.0, 0.34),
            Rotation3::identity().into(),
        ),
        Isometry3::from_parts(
            Translation3::new(0.0, 0.0, 0.36),
            Rotation3::identity().into(),
        ),
    ];

    // GT laser planes in rig frame (distinct per camera).
    let planes_rig_gt = [
        LaserPlane::new(Vector3::new(0.0, 1.0, 0.1), -0.0),
        LaserPlane::new(Vector3::new(0.0, 1.0, -0.1), -0.0),
    ];
    // Convert to camera frames.
    let planes_cam_gt: Vec<LaserPlane> = planes_rig_gt
        .iter()
        .zip(cam_to_rig.iter())
        .map(|(p_rig, t)| p_rig.transform_by(&t.inverse()))
        .collect();

    let target_pts = make_target_points();

    // Build synthetic views.
    let mut views: Vec<RigLaserlineView> = Vec::new();
    for rig_pose in &rig_se3_target {
        let mut cam_obs = Vec::new();
        let mut laser_obs = Vec::new();
        for cam_idx in 0..2 {
            let cam_se3_target = cam_to_rig[cam_idx].inverse() * rig_pose;

            // Target correspondences.
            let mut pts3 = Vec::new();
            let mut pts2 = Vec::new();
            for p in &target_pts {
                let p_cam = cam_se3_target.transform_point(p);
                if let Some(uv) = cam.project_point(&p_cam) {
                    pts3.push(*p);
                    pts2.push(uv);
                }
            }
            cam_obs.push(Some(CorrespondenceView::new(pts3, pts2).unwrap()));

            // Laser pixels: intersect GT plane with target plane (z=0).
            let plane = &planes_cam_gt[cam_idx];
            let n = plane.normal.into_inner();
            let d = plane.distance;
            let rot = cam_se3_target.rotation.to_rotation_matrix();
            let trans = cam_se3_target.translation.vector;
            let r_col0: Vector3<f64> = rot.matrix().column(0).into_owned();
            let r_col1: Vector3<f64> = rot.matrix().column(1).into_owned();
            let c1 = n.dot(&r_col0);
            let c2 = n.dot(&r_col1);
            let c0 = n.dot(&trans) + d;
            if c2.abs() < 1e-9 {
                laser_obs.push(Some(Vec::new()));
                continue;
            }
            let mut pixels = Vec::new();
            for k in -10..=10i32 {
                let x = k as f64 * 0.004;
                let y = -(c1 * x + c0) / c2;
                let pt_cam = cam_se3_target.transform_point(&Pt3::new(x, y, 0.0));
                if let Some(uv) = cam.project_point(&pt_cam) {
                    pixels.push(uv);
                }
            }
            laser_obs.push(Some(pixels));
        }
        views.push(RigLaserlineView {
            cameras: cam_obs,
            laser_pixels: laser_obs,
        });
    }

    let dataset = RigLaserlineDataset::new(views, 2).unwrap();

    let upstream = RigLaserlineUpstream {
        intrinsics: vec![intr, intr],
        distortion: vec![dist, dist],
        sensors: vec![sensor, sensor],
        cam_to_rig: cam_to_rig.to_vec(),
        rig_se3_target,
    };

    // Initial planes: slightly perturbed from GT (in camera frame).
    let initial_planes_cam: Vec<LaserPlane> = planes_cam_gt
        .iter()
        .map(|p| {
            let n = p.normal.into_inner();
            let perturbed = Unit::new_normalize(Vector3::new(n.x + 0.05, n.y, n.z));
            LaserPlane {
                normal: perturbed,
                distance: p.distance + 0.01,
            }
        })
        .collect();

    let opts = RigLaserlineSolveOptions::default();
    let backend_opts = BackendSolveOptions {
        max_iters: 100,
        verbosity: 0,
        min_abs_decrease: Some(1e-12),
        min_rel_decrease: Some(1e-12),
        min_error: Some(1e-14),
        ..Default::default()
    };

    let est = optimize_rig_laserline(
        &dataset,
        &upstream,
        &initial_planes_cam,
        &opts,
        &backend_opts,
    )
    .unwrap();

    // Verify per-camera rig-frame plane recovery.
    for (cam_idx, (gt, got)) in planes_rig_gt.iter().zip(&est.laser_planes_rig).enumerate() {
        let n_dot = gt.normal.dot(&got.normal);
        let ang_rad = n_dot.abs().min(1.0).acos();
        let ang_deg = ang_rad.to_degrees();
        let d_err = (gt.distance - got.distance).abs();
        let d_rel = if gt.distance.abs() > 1e-9 {
            d_err / gt.distance.abs()
        } else {
            d_err
        };
        println!("cam{cam_idx}: normal_angle={ang_deg:.3}° d_err={d_err:.4} d_rel={d_rel:.4}");
        assert!(
            ang_deg < 0.5,
            "cam{cam_idx} normal angle {ang_deg:.3}° >= 0.5°"
        );
        // Distance is ~0; use absolute tolerance.
        assert!(d_err < 0.01, "cam{cam_idx} distance error {d_err:.4}");
    }

    // Reprojection residual quality.
    for (cam_idx, stats) in est.per_camera_stats.iter().enumerate() {
        println!("cam{cam_idx} reproj={:.3e}", stats.mean_reproj_error);
        assert!(
            stats.mean_reproj_error < 0.5,
            "cam{cam_idx} RMS reproj {:.3} px >= 0.5 px",
            stats.mean_reproj_error
        );
    }
}
