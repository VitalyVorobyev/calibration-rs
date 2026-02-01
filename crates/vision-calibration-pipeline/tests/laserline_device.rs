use nalgebra::{Point3, Rotation3, Translation3, Vector3};
use vision_calibration_core::{
    BrownConrady5, Camera, CorrespondenceView, FxFyCxCySkew, Iso3, Pinhole, Pt2, Pt3,
    ScheimpflugParams, SensorModel, View,
};
use vision_calibration_optim::{LaserPlane, LaserlineMeta, LaserlineView};
use vision_calibration_pipeline::laserline_device::{
    LaserlineDeviceConfig, LaserlineDeviceProblem, run_calibration,
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
    camera: &Camera<f64, Pinhole, BrownConrady5<f64>, Sm, FxFyCxCySkew<f64>>,
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
            let ray = camera.backproject_pixel(&pixel);
            if !ray.point.x.is_finite() || !ray.point.y.is_finite() || !ray.point.z.is_finite() {
                continue;
            }
            if ray.point.z.abs() < 1e-6 {
                continue;
            }
            let ray_dir_camera = ray.point.normalize();
            let ray_dir_target = pose.inverse_transform_vector(&ray_dir_camera);
            if ray_dir_target.z.abs() > 1e-3 {
                pixels.push(pixel);
            }
        }
    }
    pixels
}

fn make_dataset(sensor: ScheimpflugParams) -> (Vec<LaserlineView>, FxFyCxCySkew<f64>, LaserPlane) {
    let intrinsics = FxFyCxCySkew {
        fx: 800.0,
        fy: 780.0,
        cx: 640.0,
        cy: 360.0,
        skew: 0.0,
    };
    let distortion = BrownConrady5 {
        k1: 0.0,
        k2: 0.0,
        k3: 0.0,
        p1: 0.0,
        p2: 0.0,
        iters: 8,
    };

    let camera = Camera::new(Pinhole, distortion, sensor.compile(), intrinsics);
    let points_3d = make_target_points();
    let poses = make_poses();
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
                    laser_weights: None,
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

    let config = LaserlineDeviceConfig {
        fix_plane: true,
        ..Default::default()
    };

    let mut session = CalibrationSession::<LaserlineDeviceProblem>::new();
    session.set_input(dataset).unwrap();
    run_calibration(&mut session, Some(config)).unwrap();

    let export = session.export().unwrap();
    let params = &export.estimate.params;

    let json = serde_json::to_string(&export).unwrap();
    let _: vision_calibration_pipeline::laserline_device::LaserlineDeviceOutput =
        serde_json::from_str(&json).unwrap();

    let fx_err = (params.intrinsics.fx - intrinsics_gt.fx).abs() / intrinsics_gt.fx;
    let fy_err = (params.intrinsics.fy - intrinsics_gt.fy).abs() / intrinsics_gt.fy;
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
}

#[test]
fn pipeline_converges_scheimpflug() {
    let sensor_gt = ScheimpflugParams {
        tilt_x: 0.01,
        tilt_y: -0.008,
    };
    let (dataset, _intrinsics_gt, _plane_gt) = make_dataset(sensor_gt);

    let config = LaserlineDeviceConfig {
        sensor_init: ScheimpflugParams {
            tilt_x: 0.005,
            tilt_y: -0.004,
        },
        fix_sensor: false,
        ..Default::default()
    };

    let mut session = CalibrationSession::<LaserlineDeviceProblem>::new();
    session.set_input(dataset).unwrap();
    run_calibration(&mut session, Some(config)).unwrap();

    let export = session.export().unwrap();
    let est_sensor = export.estimate.params.sensor;

    assert!(
        (est_sensor.tilt_x - sensor_gt.tilt_x).abs() < 0.01,
        "tilt_x error too large"
    );
    assert!(
        (est_sensor.tilt_y - sensor_gt.tilt_y).abs() < 0.01,
        "tilt_y error too large"
    );
}

#[test]
fn pipeline_rejects_insufficient_views() {
    let sensor = ScheimpflugParams::default();
    let (mut dataset, _k, _plane) = make_dataset(sensor);
    dataset.truncate(1);

    let mut session = CalibrationSession::<LaserlineDeviceProblem>::new();
    let err = session.set_input(dataset).unwrap_err().to_string();
    assert!(err.contains("need at least 3 views"));
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
