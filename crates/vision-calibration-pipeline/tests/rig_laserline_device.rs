//! Pipeline integration test for `rig_laserline_device`.
//!
//! Builds a synthetic upstream rig calibration (2 cameras, 4 views) and
//! a rig laserline dataset, then runs both steps and asserts per-camera
//! laser-plane recovery within tight tolerance.

use nalgebra::{Isometry3, Rotation3, Translation3, Unit, Vector3};
use vision_calibration_core::LaserPlane;
use vision_calibration_core::{
    BrownConrady5, Camera, CorrespondenceView, FxFyCxCy, Pinhole, Pt3, Real, ScheimpflugParams,
};
use vision_calibration_optim::{RigLaserlineDataset, RigLaserlineView};
use vision_calibration_pipeline::common::CalibratedCameras;
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
    let intrinsics = FxFyCxCy {
        fx: 600.0,
        fy: 600.0,
        cx: 320.0,
        cy: 240.0,
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
        LaserPlane::new((Vector3::new(0.0, 1.0, 0.1)).normalize(), -0.0).unwrap(),
        LaserPlane::new((Vector3::new(0.0, 1.0, -0.1)).normalize(), -0.0).unwrap(),
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
            let n = plane.normal().into_inner();
            let d = plane.distance();
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
    let camera = vision_calibration_core::CameraParams {
        projection: vision_calibration_core::ProjectionParams::Pinhole,
        distortion: vision_calibration_core::DistortionParams::BrownConrady5 { params: distortion },
        sensor: vision_calibration_core::SensorParams::Scheimpflug { params: sensor },
        intrinsics,
    };
    let upstream = RigUpstreamCalibration {
        cameras: vec![camera.clone(), camera],
        cam_se3_rig: cam_to_rig.iter().map(|t| t.inverse()).collect(),
        rig_se3_target,
        uncertainty: Default::default(),
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
            let n = p_cam.normal().into_inner();
            let perturbed = Unit::new_normalize(Vector3::new(n.x + 0.05, n.y, n.z));
            LaserPlane::new(perturbed.into_inner(), p_cam.distance() + 0.01).unwrap()
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
        let n_dot = gt.normal().dot(&got.normal());
        let ang = n_dot.abs().min(1.0).acos();
        let d_err = (gt.distance() - got.distance()).abs();
        assert!(ang < 0.01, "cam{cam_idx} normal error {ang:.4} rad");
        assert!(d_err < 0.01, "cam{cam_idx} distance error {d_err:.4}");
    }

    // The export's calibration carries each camera's plane.
    let rig = export.calibrated_rig().unwrap();
    for (i, plane) in export.laser_planes_cam.iter().enumerate() {
        assert_eq!(rig.laser_plane(i).unwrap().distance(), plane.distance());
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

    // Viewer contract: the export echoes the frozen upstream rig
    // geometry so the diagnose / 3D workspaces can render it without
    // the hand-eye export. Same field names as
    // RigHandeyeExport.
    assert_eq!(export.cameras.len(), 2);
    assert_eq!(export.cam_se3_rig.len(), 2);
    assert_eq!(export.rig_se3_target.len(), 4, "one T_R_T per view");
    assert!(
        export.mean_reproj_error.is_finite() && export.mean_reproj_error < 1.0,
        "export-level mean reproj: {}",
        export.mean_reproj_error
    );
}

#[test]
fn json_roundtrip_for_config() {
    use vision_calibration_pipeline::rig_laserline_device::RigLaserlineDeviceConfig;
    let config = RigLaserlineDeviceConfig::default();
    let json = serde_json::to_string(&config).unwrap();
    let _: RigLaserlineDeviceConfig = serde_json::from_str(&json).unwrap();
}

/// An untilted rig hand-eye export converts to an upstream calibration whose
/// cameras keep their identity sensor, and the full rig laserline pipeline
/// converges on it: the laser model evaluates an untilted camera as zero
/// tilt, exactly the identity mapping.
#[test]
fn pinhole_upstream_converges_rig_laserline() {
    use vision_calibration_core::PerFeatureResiduals;
    use vision_calibration_optim::HandeyeMode;
    use vision_calibration_pipeline::rig_handeye::RigHandeyeExport;

    let (dataset, upstream, planes_rig_gt) = make_dataset();

    // Rebuild the same upstream through a pinhole RigHandeyeExport.
    // The export type is #[non_exhaustive]; construct it through serde like a
    // downstream consumer loading a stored pinhole export would.
    let cameras: Vec<_> = upstream
        .cameras
        .iter()
        .map(|c| vision_calibration_core::CameraParams {
            sensor: vision_calibration_core::SensorParams::Identity,
            ..c.clone()
        })
        .collect();
    let export: RigHandeyeExport = serde_json::from_value(serde_json::json!({
        "kind": "rig_handeye",
        "cameras": cameras,
        "cam_se3_rig": upstream.cam_se3_rig,
        "rig_se3_target": upstream.rig_se3_target,
        "handeye_mode": HandeyeMode::EyeInHand,
        "gripper_se3_rig": Isometry3::<f64>::identity(),
        "rig_se3_base": null,
        "base_se3_target": Isometry3::<f64>::identity(),
        "gripper_se3_target": null,
        "robot_deltas": null,
        "mean_reproj_error": 0.0,
        "per_cam_reproj_errors": [0.0, 0.0],
        "per_feature_residuals": PerFeatureResiduals::default(),
    }))
    .expect("synthesize pinhole export");

    let converted = export
        .to_upstream_calibration(upstream.rig_se3_target.clone())
        .expect("pinhole export must convert");
    assert_eq!(converted.cameras.len(), 2);
    for camera in &converted.cameras {
        assert!(matches!(
            camera.sensor,
            vision_calibration_core::SensorParams::Identity
        ));
    }

    let input = RigLaserlineDeviceInput {
        dataset,
        upstream: converted,
        initial_planes_cam: None,
    };
    let mut session = CalibrationSession::<RigLaserlineDeviceProblem>::new();
    session.set_input(input).unwrap();
    run_calibration(&mut session).unwrap();
    let result = session.export().unwrap();

    for (cam_idx, (gt, got)) in planes_rig_gt
        .iter()
        .zip(&result.laser_planes_rig)
        .enumerate()
    {
        let n_dot = gt.normal().dot(&got.normal());
        let ang = n_dot.abs().min(1.0).acos();
        let d_err = (gt.distance() - got.distance()).abs();
        assert!(ang < 0.01, "cam{cam_idx} normal error {ang:.4} rad");
        assert!(d_err < 0.01, "cam{cam_idx} distance error {d_err:.4}");
    }
}

// ─── The upstream calibration's covariance in the planes' ─────────────────────

use vision_calibration_core::{ParameterCovariance, Uncertainty, UncertaintyUnavailable};
use vision_calibration_optim::SolverBackend;
use vision_calibration_pipeline::rig_laserline_device::RigLaserlineDeviceConfig;

fn calibrate(
    dataset: &RigLaserlineDataset,
    upstream: RigUpstreamCalibration,
    backend: SolverBackend,
) -> RigLaserlineDeviceExport {
    let mut config = RigLaserlineDeviceConfig::default();
    config.solver.backend = backend;
    let mut session = CalibrationSession::<RigLaserlineDeviceProblem>::new();
    session.set_config(config).unwrap();
    session
        .set_input(RigLaserlineDeviceInput {
            dataset: dataset.clone(),
            upstream,
            initial_planes_cam: None,
        })
        .unwrap();
    run_calibration(&mut session).unwrap();
    session.export().unwrap()
}

fn plane_labels() -> Vec<String> {
    (0..2)
        .flat_map(|c| ["nx", "ny", "nz", "d"].map(|p| format!("cam{c}.plane.{p}")))
        .collect()
}

/// An upstream covariance over cameras, extrinsics and target poses (one
/// view's in its own block): the export lists the cameras' parameters and
/// extrinsics in the upstream's order in front of the planes, holds the rest,
/// and is the same on both backends.
#[test]
fn the_upstream_covariance_reaches_the_export() {
    let (dataset, mut upstream, planes_rig_gt) = make_dataset();
    let pose = |name: &str| -> Vec<String> {
        ["rx", "ry", "rz", "tx", "ty", "tz"]
            .iter()
            .map(|c| format!("{name}.{c}"))
            .collect()
    };
    let mut labels: Vec<String> = vec!["cam0.fx".into(), "cam1.cy".into(), "cam1.k1".into()];
    labels.extend(pose("cam1.cam_se3_rig"));
    for v in [0, 1, 3] {
        labels.extend(pose(&format!("view{v}.rig_se3_target")));
    }
    let n = labels.len();
    let variance = |l: &str| match l.rsplit('.').next().unwrap() {
        "fx" | "cy" => 0.25,
        "k1" => 1e-6,
        "rx" | "ry" | "rz" => 1e-7,
        _ => 1e-8,
    };
    // Correlate the extrinsic and view 0's pose a little.
    let covariance: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    let (a, b) = (&labels[i], &labels[j]);
                    if i == j {
                        variance(a)
                    } else if a.starts_with("cam1.cam_se3_rig") && b.starts_with("view0")
                        || b.starts_with("cam1.cam_se3_rig") && a.starts_with("view0")
                    {
                        if a.ends_with(".tz") && b.ends_with(".tz") {
                            0.5e-8
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect();
    upstream.uncertainty = Uncertainty::Available(ParameterCovariance {
        labels: labels.clone(),
        covariance,
        views: vec![vision_calibration_core::ViewCovariance {
            view: 2,
            labels: pose("rig_se3_target"),
            covariance: (0..6)
                .map(|i| (0..6).map(|j| if i == j { 1e-8 } else { 0.0 }).collect())
                .collect(),
        }],
        fixed: vec!["cam0.k3".into()],
        noise: Vec::new(),
        num_residuals: 0,
        dof: 0,
        view_scatter: None,
    });

    let export = calibrate(&dataset, upstream.clone(), SolverBackend::TinySolver);
    let cov = export.uncertainty.covariance().expect("covariance");
    let reported: Vec<&String> = labels.iter().filter(|l| !l.starts_with("view")).collect();
    let mut expected: Vec<String> = reported.iter().map(|l| l.to_string()).collect();
    expected.extend(plane_labels());
    assert_eq!(cov.labels, expected);
    for (i, a) in reported.iter().enumerate() {
        assert_eq!(cov.covariance[i][i], variance(a), "{a}");
    }
    // What the upstream did not list stays held; what it listed does not.
    for label in ["cam0.fy", "cam1.fx", "cam0.cam_se3_rig", "cam0.k3"] {
        assert!(
            cov.fixed.iter().any(|l| l == label),
            "{label}: {:?}",
            cov.fixed
        );
    }
    for label in [
        "cam0.fx",
        "cam1.cam_se3_rig",
        "view0.rig_se3_target",
        "view2.rig_se3_target",
    ] {
        assert!(
            !cov.fixed.iter().any(|l| l == label),
            "{label}: {:?}",
            cov.fixed
        );
    }
    // The planes of the two cameras are correlated through the target poses.
    let (p0, p1) = (
        cov.index("cam0.plane.d").unwrap(),
        cov.index("cam1.plane.d").unwrap(),
    );
    assert!(cov.covariance[p0][p1].abs() > 0.0);
    // Symmetric and positive semi-definite.
    let m = nalgebra::DMatrix::from_fn(cov.labels.len(), cov.labels.len(), |i, j| {
        cov.covariance[i][j]
    });
    assert_eq!(m, m.transpose());
    let min = m.symmetric_eigenvalues().min();
    assert!(min > -1e-12 * m.amax(), "min eigenvalue {min:e}");

    // The factrs backend reaches the same covariance.
    let factrs = calibrate(&dataset, upstream, SolverBackend::Factrs);
    let other = factrs.uncertainty.covariance().unwrap();
    assert_eq!(other.labels, cov.labels);
    for (row, other_row) in cov.covariance.iter().zip(&other.covariance) {
        for (a, b) in row.iter().zip(other_row) {
            assert!((a - b).abs() <= 1e-6 * m.amax(), "{a:e} vs {b:e}");
        }
    }

    // The calibration a measurement reads carries it whole.
    let rig = export.calibrated_rig().unwrap();
    assert_eq!(rig.uncertainty(), &export.uncertainty);
    let _ = planes_rig_gt;
}

/// Without an upstream covariance (an input that predates it, or one stated
/// unavailable) the upstream is held exact: the planes' covariance is
/// conditional on it and lists the planes alone.
#[test]
fn an_upstream_without_covariance_is_held_exact() {
    let (dataset, upstream, _) = make_dataset();
    let input = RigLaserlineDeviceInput {
        dataset,
        upstream,
        initial_planes_cam: None,
    };
    let mut json = serde_json::to_value(&input).unwrap();
    json["upstream"]
        .as_object_mut()
        .unwrap()
        .remove("uncertainty");
    let parsed: RigLaserlineDeviceInput = serde_json::from_value(json).unwrap();
    assert_eq!(
        parsed.upstream.uncertainty,
        Uncertainty::Unavailable {
            reason: UncertaintyUnavailable::NotComputed
        }
    );
    let export = calibrate(&parsed.dataset, parsed.upstream, SolverBackend::TinySolver);
    let cov = export.uncertainty.covariance().expect("the planes' own");
    assert_eq!(cov.labels, plane_labels());
    for label in ["cam0.fx", "cam1.cam_se3_rig", "view3.rig_se3_target"] {
        assert!(
            cov.fixed.iter().any(|l| l == label),
            "{label}: {:?}",
            cov.fixed
        );
    }
}
