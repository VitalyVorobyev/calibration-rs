//! Example: single laserline device calibration (camera + laser plane).

use calib::core::ScheimpflugParams;
use calib::core::SensorModel;
use calib::laserline_device::{run_calibration, LaserlineDeviceConfig, LaserlineDeviceProblem};
use calib::optim::{LaserPlane, LaserlineMeta, LaserlineView};
use calib::prelude::*;

use nalgebra::{Point3, Rotation3, Translation3, Vector3};

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

fn laser_pixels_for_view<Sm: SensorModel<f64>>(
    pose: &Iso3,
    plane: &LaserPlane,
    camera: &Camera<f64, Pinhole, BrownConrady5<f64>, Sm, FxFyCxCySkew<f64>>,
) -> Vec<Pt2> {
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

fn main() -> anyhow::Result<()> {
    // Ground truth camera (pinhole + Brown-Conrady + identity sensor)
    let intrinsics = FxFyCxCySkew {
        fx: 800.0,
        fy: 780.0,
        cx: 640.0,
        cy: 360.0,
        skew: 0.0,
    };
    let distortion = BrownConrady5 {
        k1: -0.2,
        k2: 0.05,
        k3: 0.0,
        p1: 0.001,
        p2: -0.001,
        iters: 8,
    };
    let sensor = ScheimpflugParams::default();
    let camera = Camera::new(Pinhole, distortion, sensor.compile(), intrinsics);

    // Synthetic dataset
    let points_3d = make_target_points();
    let poses = make_poses();
    let plane = make_laser_plane(&poses);

    let mut views: Vec<LaserlineView> = Vec::new();
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
            let obs = CorrespondenceView::new(points_3d_view, points_2d)?;
            views.push(View::new(
                obs,
                LaserlineMeta {
                    laser_pixels,
                    laser_weights: None,
                },
            ));
        }
    }

    let mut session = CalibrationSession::<LaserlineDeviceProblem>::new();
    session.set_input(views)?;

    run_calibration(&mut session, Some(LaserlineDeviceConfig::default()))?;
    let export = session.export()?;

    println!(
        "Calibrated intrinsics: fx={:.1}, fy={:.1}, cx={:.1}, cy={:.1}",
        export.estimate.params.intrinsics.fx,
        export.estimate.params.intrinsics.fy,
        export.estimate.params.intrinsics.cx,
        export.estimate.params.intrinsics.cy,
    );
    println!(
        "Mean reproj error: {:.3} px, mean laser error: {:.3}",
        export.stats.mean_reproj_error, export.stats.mean_laser_error
    );
    Ok(())
}
