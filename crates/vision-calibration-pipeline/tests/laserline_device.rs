#![allow(missing_docs)]

use nalgebra::{Point3, Rotation3, Translation3, Vector3};
use vision_calibration_core::{
    BrownConrady5, Camera, CameraModelSpec, CorrespondenceView, DistortionKind, FxFyCxCy, Iso3,
    Pinhole, Pt2, Pt3, ScheimpflugParams, SensorKind, SensorModel, SensorParams, View,
};
use vision_calibration_optim::{LaserPlane, LaserlineMeta, LaserlineView};
use vision_calibration_pipeline::common::config::CameraConfig;
use vision_calibration_pipeline::laserline_device::{
    DeviceManualInit, LaserlineDeviceConfig, LaserlineDeviceExport, LaserlineDeviceProblem,
    run_calibration, step_init_with_seed, step_optimize,
};
use vision_calibration_pipeline::session::CalibrationSession;

fn make_target_points() -> Vec<Pt3> {
    let spacing = 0.03;
    let mut points = Vec::new();
    for i in 0..6 {
        for j in 0..5 {
            points.push(Pt3::new(i as f64 * spacing, j as f64 * spacing, 0.0));
        }
    }
    points
}

fn make_poses() -> Vec<Iso3> {
    vec![
        Iso3::from_parts(
            Translation3::new(0.0, 0.0, 0.5),
            Rotation3::from_euler_angles(0.0, 0.0, 0.0).into(),
        ),
        Iso3::from_parts(
            Translation3::new(0.05, -0.02, 0.55),
            Rotation3::from_euler_angles(0.15, -0.05, 0.0).into(),
        ),
        Iso3::from_parts(
            Translation3::new(-0.04, 0.03, 0.6),
            Rotation3::from_euler_angles(-0.1, 0.08, 0.02).into(),
        ),
        Iso3::from_parts(
            Translation3::new(0.02, 0.06, 0.52),
            Rotation3::from_euler_angles(0.05, 0.12, -0.04).into(),
        ),
        Iso3::from_parts(
            Translation3::new(-0.06, -0.04, 0.58),
            Rotation3::from_euler_angles(-0.12, -0.04, 0.06).into(),
        ),
    ]
}

fn make_laser_plane(poses: &[Iso3]) -> LaserPlane {
    let laser_normal = Vector3::new(0.1, 0.05, 0.99).normalize();
    let target_center = Pt3::new(0.075, 0.06, 0.0);
    let target_center_cam = poses[0].transform_point(&target_center);
    LaserPlane::new(laser_normal, -laser_normal.dot(&target_center_cam.coords))
}

fn laser_pixels_for_view<Sm>(
    pose: &Iso3,
    plane: &LaserPlane,
    camera: &Camera<f64, Pinhole, BrownConrady5<f64>, Sm, FxFyCxCy<f64>>,
) -> Vec<Pt2>
where
    Sm: SensorModel<f64>,
{
    let n_c = plane.normal.as_ref();
    let n_t = pose.rotation.inverse_transform_vector(n_c);
    let d_t = n_c.dot(&pose.translation.vector) + plane.distance;

    let dir = Vector3::new(n_t.y, -n_t.x, 0.0);
    let dir_norm = dir.norm();
    if dir_norm < 1e-9 {
        return Vec::new();
    }
    let dir = dir / dir_norm;
    let (x0, y0) = if n_t.x.abs() > n_t.y.abs() {
        (-d_t / n_t.x, 0.0)
    } else {
        (0.0, -d_t / n_t.y)
    };

    let mut pixels = Vec::new();
    for i in 0..40 {
        let s = (i as f64 / 39.0) * 0.2 - 0.1;
        let pt_target = Point3::new(x0 + s * dir.x, y0 + s * dir.y, 0.0);
        let pt_camera = pose.transform_point(&pt_target);
        if let Some(proj) = camera.project_point(&pt_camera) {
            let pixel = Pt2::new(proj.x, proj.y);
            let ray = camera.backproject_pixel(&pixel).direction;
            if !ray.x.is_finite() || !ray.y.is_finite() || !ray.z.is_finite() {
                continue;
            }
            if ray.z.abs() < 1e-6 {
                continue;
            }
            let ray_dir_camera = ray.normalize();
            let ray_dir_target = pose.inverse_transform_vector(&ray_dir_camera);
            if ray_dir_target.z.abs() > 1e-3 {
                pixels.push(pixel);
            }
        }
    }
    pixels
}

fn make_dataset(sensor: ScheimpflugParams) -> (Vec<LaserlineView>, FxFyCxCy<f64>, LaserPlane) {
    make_dataset_with(sensor, BrownConrady5::default(), make_poses())
}

/// Ten views tilted up to ±0.25 rad about both axes, the board centred.
fn make_oblique_poses() -> Vec<Iso3> {
    [
        (0.0, 0.0),
        (0.25, 0.0),
        (-0.25, 0.0),
        (0.0, 0.25),
        (0.0, -0.25),
        (0.18, 0.18),
        (-0.18, 0.18),
        (0.18, -0.18),
        (-0.18, -0.18),
        (0.1, -0.05),
    ]
    .iter()
    .enumerate()
    .map(|(i, &(rx, ry))| {
        Iso3::from_parts(
            Translation3::new(-0.075, -0.06, 0.5 + 0.03 * (i % 3) as f64),
            Rotation3::from_euler_angles(rx, ry, 0.05 * i as f64).into(),
        )
    })
    .collect()
}

fn make_dataset_with(
    sensor: ScheimpflugParams,
    distortion: BrownConrady5<f64>,
    poses: Vec<Iso3>,
) -> (Vec<LaserlineView>, FxFyCxCy<f64>, LaserPlane) {
    let intrinsics = FxFyCxCy {
        fx: 800.0,
        fy: 780.0,
        cx: 640.0,
        cy: 360.0,
    };

    let camera = Camera::new(Pinhole, distortion, sensor.compile(), intrinsics);
    let points_3d = make_target_points();
    let plane = make_laser_plane(&poses);

    let mut views = Vec::new();
    for pose in &poses {
        let mut points_2d = Vec::new();
        let mut points_3d_view = Vec::new();
        for pw in &points_3d {
            let p_cam = pose.transform_point(pw);
            if let Some(px) = camera.project_point(&p_cam) {
                points_3d_view.push(*pw);
                points_2d.push(Pt2::new(px.x, px.y));
            }
        }

        let laser_pixels = laser_pixels_for_view(pose, &plane, &camera);
        if points_2d.len() >= 4 && !laser_pixels.is_empty() {
            let obs = CorrespondenceView::new(points_3d_view, points_2d).unwrap();
            views.push(View::new(
                obs,
                LaserlineMeta {
                    laser_pixels,
                    laser_weights: Vec::new(),
                },
            ));
        }
    }

    (views, intrinsics, plane)
}

#[test]
fn pipeline_converges_pinhole() {
    let sensor = ScheimpflugParams::default();
    let (dataset, intrinsics_gt, plane_gt) = make_dataset(sensor);

    let mut config = LaserlineDeviceConfig::default();
    config.optimize.fix_plane = true;

    let mut session = CalibrationSession::<LaserlineDeviceProblem>::new();
    session.set_input(dataset).unwrap();
    session.set_config(config).unwrap();
    run_calibration(&mut session).unwrap();

    let export = session.export().unwrap();
    let params = &export.params;

    let json = serde_json::to_string(&export).unwrap();
    let _: LaserlineDeviceExport = serde_json::from_str(&json).unwrap();

    assert!(matches!(params.camera.sensor, SensorParams::Identity));
    let fx_err = (params.camera.intrinsics.fx - intrinsics_gt.fx).abs() / intrinsics_gt.fx;
    let fy_err = (params.camera.intrinsics.fy - intrinsics_gt.fy).abs() / intrinsics_gt.fy;
    assert!(fx_err < 0.05, "fx error too large: {:.3}%", fx_err * 100.0);
    assert!(fy_err < 0.05, "fy error too large: {:.3}%", fy_err * 100.0);

    let normal_dot = params.plane.normal.dot(&plane_gt.normal);
    assert!(normal_dot.is_finite(), "plane normal dot is not finite");
    let normal_angle = normal_dot.abs().min(1.0).acos().to_degrees();
    assert!(
        normal_angle < 5.0,
        "plane normal error too large: {:.2}°",
        normal_angle
    );

    assert!(export.stats.mean_reproj_error < 1.0);
    assert_eq!(export.per_cam_reproj_errors.len(), 1);
    assert!((export.per_cam_reproj_errors[0] - export.mean_reproj_error).abs() < 1.0e-12);
}

#[test]
fn pipeline_converges_scheimpflug() {
    // Radial distortion centres on the optical axis, which makes the tilt
    // observable: without it a tilt is a principal-point shift and a pose
    // rotation.
    let sensor_gt = ScheimpflugParams {
        tilt_x: 0.04,
        tilt_y: -0.03,
    };
    let distortion = BrownConrady5 {
        k1: -0.25,
        k2: 0.08,
        ..BrownConrady5::default()
    };
    let (dataset, _intrinsics_gt, _plane_gt) =
        make_dataset_with(sensor_gt, distortion, make_oblique_poses());

    // Estimated tilt, cold start: an explicit (empty) sensor fix list opts
    // into estimating the tilt; the linear tilt sweep seeds it and the
    // bundle refines it.
    let mut config = LaserlineDeviceConfig::default();
    config.camera = CameraConfig::new(tilted());
    config.camera.fix.sensor = Some(Vec::new());

    let mut session = CalibrationSession::<LaserlineDeviceProblem>::new();
    session.set_input(dataset.clone()).unwrap();
    session.set_config(config.clone()).unwrap();
    run_calibration(&mut session).unwrap();
    let export = session.export().unwrap();
    assert!(
        export.mean_reproj_error < 1e-4,
        "{}",
        export.mean_reproj_error
    );
    assert_tilt(&export, sensor_gt);

    // Default: a laserline device's mount tilt is held at its seed, as is
    // an explicit hold.
    for hold in [None, Some(vec!["tilt_x".into(), "tilt_y".into()])] {
        let mut config = config.clone();
        config.camera.fix.sensor = hold;
        let mut session = CalibrationSession::<LaserlineDeviceProblem>::new();
        session.set_input(dataset.clone()).unwrap();
        session.set_config(config).unwrap();
        let mut seed = DeviceManualInit::default();
        seed.sensor = Some(sensor_gt);
        step_init_with_seed(&mut session, seed, None).unwrap();
        step_optimize(&mut session, None).unwrap();
        let export = session.export().unwrap();
        let SensorParams::Scheimpflug { params } = export.params.camera.sensor else {
            panic!("tilted camera expected");
        };
        assert_eq!(
            (params.tilt_x, params.tilt_y),
            (sensor_gt.tilt_x, sensor_gt.tilt_y),
            "a held tilt keeps its seed"
        );
    }

    // Seeded: a known mount tilt, held fixed.
    config.camera.fix.sensor = Some(vec!["tilt_x".into(), "tilt_y".into()]);
    let mut session = CalibrationSession::<LaserlineDeviceProblem>::new();
    session.set_input(dataset).unwrap();
    session.set_config(config).unwrap();
    let mut seed = DeviceManualInit::default();
    seed.sensor = Some(sensor_gt);
    step_init_with_seed(&mut session, seed, None).unwrap();
    step_optimize(&mut session, None).unwrap();
    let export = session.export().unwrap();
    let SensorParams::Scheimpflug { params } = export.params.camera.sensor else {
        panic!("tilted camera expected");
    };
    assert_eq!(
        (params.tilt_x, params.tilt_y),
        (sensor_gt.tilt_x, sensor_gt.tilt_y),
        "a held tilt keeps its seed"
    );
}

fn tilted() -> CameraModelSpec {
    CameraModelSpec::pinhole(DistortionKind::BrownConrady5, SensorKind::Scheimpflug)
}

fn assert_tilt(export: &LaserlineDeviceExport, sensor_gt: ScheimpflugParams) {
    let SensorParams::Scheimpflug { params: est } = export.params.camera.sensor else {
        panic!("tilted camera expected");
    };
    assert!(
        (est.tilt_x - sensor_gt.tilt_x).abs() < 1e-3,
        "tilt_x error too large: {est:?}"
    );
    assert!(
        (est.tilt_y - sensor_gt.tilt_y).abs() < 1e-3,
        "tilt_y error too large: {est:?}"
    );
}

#[test]
fn tilt_seed_for_an_untilted_camera_must_be_zero() {
    let (dataset, _k, _plane) = make_dataset(ScheimpflugParams::default());
    let mut session = CalibrationSession::<LaserlineDeviceProblem>::new();
    session.set_input(dataset).unwrap();
    let mut seed = DeviceManualInit::default();
    seed.sensor = Some(ScheimpflugParams::default());
    assert!(step_init_with_seed(&mut session, seed.clone(), None).is_ok());
    seed.sensor = Some(ScheimpflugParams {
        tilt_x: 0.02,
        tilt_y: 0.0,
    });
    let err = step_init_with_seed(&mut session, seed, None)
        .unwrap_err()
        .to_string();
    assert!(err.contains("Scheimpflug"), "{err}");
}

#[test]
fn every_distortion_kind_calibrates() {
    let (dataset, intrinsics_gt, _plane) = make_dataset(ScheimpflugParams::default());
    for kind in [
        DistortionKind::None,
        DistortionKind::BrownConrady5,
        DistortionKind::Rational8,
        DistortionKind::ThinPrism9,
        DistortionKind::Division1,
    ] {
        let mut config = LaserlineDeviceConfig::default();
        config.camera = CameraConfig::new(CameraModelSpec::pinhole(kind, SensorKind::Identity));
        let mut session = CalibrationSession::<LaserlineDeviceProblem>::new();
        session.set_input(dataset.clone()).unwrap();
        session.set_config(config).unwrap();
        run_calibration(&mut session).unwrap_or_else(|e| panic!("{kind:?}: {e}"));
        let export = session.export().unwrap();
        let camera = &export.params.camera;
        assert_eq!(camera.spec().map(|s| s.distortion), Some(kind));
        let fx_err = (camera.intrinsics.fx - intrinsics_gt.fx).abs() / intrinsics_gt.fx;
        assert!(fx_err < 0.05, "{kind:?}: fx error {:.3}%", fx_err * 100.0);
    }
}

#[test]
fn pipeline_rejects_insufficient_views() {
    let sensor = ScheimpflugParams::default();
    let (mut dataset, _k, _plane) = make_dataset(sensor);
    dataset.truncate(1);

    let mut session = CalibrationSession::<LaserlineDeviceProblem>::new();
    let err = session.set_input(dataset).unwrap_err().to_string();
    assert!(err.contains("need 3"));
}

#[test]
fn json_roundtrip_for_input_and_config() {
    let sensor = ScheimpflugParams::default();
    let (dataset, _k, _plane) = make_dataset(sensor);
    let input_json = serde_json::to_string(&dataset).unwrap();
    let _: Vec<LaserlineView> = serde_json::from_str(&input_json).unwrap();

    let config = LaserlineDeviceConfig::default();
    let config_json = serde_json::to_string(&config).unwrap();
    let _: LaserlineDeviceConfig = serde_json::from_str(&config_json).unwrap();
}
