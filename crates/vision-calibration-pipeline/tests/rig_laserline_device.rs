//! Pipeline integration test for `rig_laserline_device`.
//!
//! Builds a synthetic upstream rig calibration (2 cameras, 4 views) and
//! a rig laserline dataset, then runs both steps and asserts per-camera
//! laser-plane recovery within tight tolerance.

use nalgebra::{Isometry3, Rotation3, Translation3, Unit, Vector3};
use vision_calibration_core::{
    BrownConrady5, Camera, CorrespondenceView, FxFyCxCySkew, Pinhole, Pt3, Real, ScheimpflugParams,
};
use vision_calibration_optim::{LaserPlane, RigLaserlineDataset, RigLaserlineView};
use vision_calibration_pipeline::rig_laserline_device::{
    RigLaserlineDeviceExport, RigLaserlineDeviceInput, RigLaserlineDeviceProblem,
    RigUpstreamCalibration, run_calibration,
};
use vision_calibration_pipeline::session::CalibrationSession;

fn make_target_points() -> Vec<Pt3> {
    let mut pts = Vec::new();
    for y in -2i32..=2 {
        for x in -3i32..=3 {
            pts.push(Pt3::new(x as Real * 0.02, y as Real * 0.02, 0.0));
        }
    }
    pts
}

/// Build synthetic rig laserline dataset and upstream calibration.
fn make_dataset() -> (
    RigLaserlineDataset,
    RigUpstreamCalibration,
    Vec<LaserPlane>, // ground-truth planes in rig frame
) {
    let intrinsics = FxFyCxCySkew {
        fx: 600.0,
        fy: 600.0,
        cx: 320.0,
        cy: 240.0,
        skew: 0.0,
    };
    let distortion = BrownConrady5 {
        k1: 0.0,
        k2: 0.0,
        k3: 0.0,
        p1: 0.0,
        p2: 0.0,
        iters: 5,
    };
    let sensor = ScheimpflugParams::default();
    let cam = Camera::new(Pinhole, distortion, sensor.compile(), intrinsics);

    let cam0_to_rig = Isometry3::identity();
    let cam1_to_rig = Isometry3::from_parts(
        Translation3::new(0.15, 0.0, 0.0),
        Rotation3::identity().into(),
    );
    let cam_to_rig = [cam0_to_rig, cam1_to_rig];

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

    // GT laser planes in rig frame.
    let planes_rig_gt = vec![
        LaserPlane::new(Vector3::new(0.0, 1.0, 0.1), -0.0),
        LaserPlane::new(Vector3::new(0.0, 1.0, -0.1), -0.0),
    ];
    // Per-camera GT planes in camera frame.
    let planes_cam_gt: Vec<LaserPlane> = planes_rig_gt
        .iter()
        .zip(cam_to_rig.iter())
        .map(|(p_rig, t)| p_rig.transform_by(&t.inverse()))
        .collect();

    let target_pts = make_target_points();

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

            // Laser pixels by intersecting GT plane with target plane (z=0).
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
            for k in -8..=8i32 {
                let x = k as f64 * 0.004;
                let y = -(c1 * x + c0) / c2;
                let pt_target = Pt3::new(x, y, 0.0);
                let pt_cam = cam_se3_target.transform_point(&pt_target);
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
    let upstream = RigUpstreamCalibration {
        intrinsics: vec![intrinsics, intrinsics],
        distortion: vec![distortion, distortion],
        sensors: vec![sensor, sensor],
        cam_se3_rig: cam_to_rig.iter().map(|t| t.inverse()).collect(),
        rig_se3_target,
    };

    (dataset, upstream, planes_rig_gt)
}

#[test]
fn pipeline_converges_rig_laserline() {
    let (dataset, upstream, planes_rig_gt) = make_dataset();

    // Perturb initial planes slightly.
    let initial_planes_cam: Vec<LaserPlane> = planes_rig_gt
        .iter()
        .zip(
            [
                Isometry3::identity(),
                Isometry3::from_parts(
                    Translation3::new(0.15, 0.0, 0.0),
                    Rotation3::identity().into(),
                ),
            ]
            .iter(),
        )
        .map(|(p_rig, c2r)| {
            let p_cam = p_rig.transform_by(&c2r.inverse());
            let n = p_cam.normal.into_inner();
            let perturbed = Unit::new_normalize(Vector3::new(n.x + 0.05, n.y, n.z));
            LaserPlane {
                normal: perturbed,
                distance: p_cam.distance + 0.01,
            }
        })
        .collect();

    let input = RigLaserlineDeviceInput {
        dataset,
        upstream,
        initial_planes_cam: Some(initial_planes_cam),
    };

    let mut session = CalibrationSession::<RigLaserlineDeviceProblem>::new();
    session.set_input(input).unwrap();
    run_calibration(&mut session).unwrap();

    let export = session.export().unwrap();

    // JSON round-trip.
    let json = serde_json::to_string(&export).unwrap();
    let _: RigLaserlineDeviceExport = serde_json::from_str(&json).unwrap();

    // Per-camera plane recovery.
    for (cam_idx, (gt, got)) in planes_rig_gt
        .iter()
        .zip(&export.laser_planes_rig)
        .enumerate()
    {
        let n_dot = gt.normal.dot(&got.normal);
        let ang = n_dot.abs().min(1.0).acos();
        let d_err = (gt.distance - got.distance).abs();
        assert!(ang < 0.01, "cam{cam_idx} normal error {ang:.4} rad");
        assert!(d_err < 0.01, "cam{cam_idx} distance error {d_err:.4}");
    }

    // Reprojection stats present.
    assert_eq!(export.per_camera_stats.len(), 2);
    for (i, stats) in export.per_camera_stats.iter().enumerate() {
        assert!(
            stats.mean_reproj_error < 1.0,
            "cam{i} mean reproj error too large: {}",
            stats.mean_reproj_error
        );
    }
}

#[test]
fn json_roundtrip_for_config() {
    use vision_calibration_pipeline::rig_laserline_device::RigLaserlineDeviceConfig;
    let config = RigLaserlineDeviceConfig::default();
    let json = serde_json::to_string(&config).unwrap();
    let _: RigLaserlineDeviceConfig = serde_json::from_str(&json).unwrap();
}
