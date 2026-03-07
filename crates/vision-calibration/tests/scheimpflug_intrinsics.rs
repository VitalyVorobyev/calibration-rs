use nalgebra::{Rotation3, Translation3};
use vision_calibration::core::{
    BrownConrady5, Camera, CorrespondenceView, FxFyCxCySkew, Iso3, Pinhole, PlanarDataset, Pt2,
    Pt3, ScheimpflugParams, View,
};
use vision_calibration::optim::RobustLoss;
use vision_calibration::scheimpflug_intrinsics::{
    ScheimpflugFixMask, ScheimpflugIntrinsicsCalibrationConfig, ScheimpflugIntrinsicsResult,
    run_calibration,
};

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
            Translation3::new(0.0, 0.0, 0.50),
            Rotation3::from_euler_angles(0.0, 0.0, 0.0).into(),
        ),
        Iso3::from_parts(
            Translation3::new(0.05, -0.02, 0.55),
            Rotation3::from_euler_angles(0.15, -0.05, 0.0).into(),
        ),
        Iso3::from_parts(
            Translation3::new(-0.04, 0.03, 0.60),
            Rotation3::from_euler_angles(-0.10, 0.08, 0.02).into(),
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

fn make_dataset(sensor: ScheimpflugParams) -> PlanarDataset {
    let camera = Camera::new(
        Pinhole,
        BrownConrady5 {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            iters: 8,
        },
        sensor.compile(),
        FxFyCxCySkew {
            fx: 800.0,
            fy: 780.0,
            cx: 640.0,
            cy: 360.0,
            skew: 0.0,
        },
    );

    let board = make_target_points();
    let views = make_poses()
        .into_iter()
        .map(|pose| {
            let mut points_3d = Vec::new();
            let mut points_2d = Vec::new();
            for point in &board {
                let point_cam = pose.transform_point(point);
                if let Some(px) = camera.project_point(&point_cam) {
                    points_3d.push(*point);
                    points_2d.push(Pt2::new(px.x, px.y));
                }
            }
            View::without_meta(CorrespondenceView::new(points_3d, points_2d).expect("obs"))
        })
        .collect();
    PlanarDataset::new(views).expect("dataset")
}

fn make_noisy_dataset(sensor: ScheimpflugParams, noise_px: f64) -> PlanarDataset {
    let camera = Camera::new(
        Pinhole,
        BrownConrady5 {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            iters: 8,
        },
        sensor.compile(),
        FxFyCxCySkew {
            fx: 800.0,
            fy: 780.0,
            cx: 640.0,
            cy: 360.0,
            skew: 0.0,
        },
    );

    let board = make_target_points();
    let views = make_poses()
        .into_iter()
        .enumerate()
        .map(|(view_idx, pose)| {
            let mut points_3d = Vec::new();
            let mut points_2d = Vec::new();
            for (point_idx, point) in board.iter().enumerate() {
                let point_cam = pose.transform_point(point);
                if let Some(px) = camera.project_point(&point_cam) {
                    let phase = (view_idx * 101 + point_idx * 37) as f64;
                    let nx = noise_px * phase.sin();
                    let ny = noise_px * phase.cos();
                    points_3d.push(*point);
                    points_2d.push(Pt2::new(px.x + nx, px.y + ny));
                }
            }
            View::without_meta(CorrespondenceView::new(points_3d, points_2d).expect("obs"))
        })
        .collect();
    PlanarDataset::new(views).expect("dataset")
}

#[test]
fn public_api_converges_on_synthetic_scheimpflug_dataset() {
    let sensor_gt = ScheimpflugParams {
        tilt_x: 0.01,
        tilt_y: -0.008,
    };
    let dataset = make_dataset(sensor_gt);
    let config = ScheimpflugIntrinsicsCalibrationConfig {
        fix_scheimpflug: vision_calibration::scheimpflug_intrinsics::ScheimpflugFixMask {
            tilt_x: false,
            tilt_y: false,
        },
        ..Default::default()
    };

    let result = run_calibration(&dataset, &config).expect("scheimpflug calibration");
    let camera = result.params.camera;

    let intrinsics = match camera.intrinsics {
        vision_calibration::core::IntrinsicsParams::FxFyCxCySkew { params } => params,
    };
    let sensor = match camera.sensor {
        vision_calibration::core::SensorParams::Scheimpflug { params } => params,
        other => panic!("unexpected sensor params: {other:?}"),
    };

    assert!(result.mean_reproj_error < 1.0);
    assert!((intrinsics.fx - 800.0).abs() / 800.0 < 0.05);
    assert!((intrinsics.fy - 780.0).abs() / 780.0 < 0.05);
    assert!((sensor.tilt_x - sensor_gt.tilt_x).abs() < 0.01);
    assert!((sensor.tilt_y - sensor_gt.tilt_y).abs() < 0.01);
}

#[test]
fn public_api_converges_with_deterministic_noise() {
    let sensor_gt = ScheimpflugParams {
        tilt_x: 0.01,
        tilt_y: -0.008,
    };
    let dataset = make_noisy_dataset(sensor_gt, 0.15);
    let config = ScheimpflugIntrinsicsCalibrationConfig {
        robust_loss: RobustLoss::Huber { scale: 1.0 },
        fix_scheimpflug: ScheimpflugFixMask {
            tilt_x: false,
            tilt_y: false,
        },
        ..Default::default()
    };

    let result = run_calibration(&dataset, &config).expect("scheimpflug calibration");
    assert!(result.mean_reproj_error < 2.0);

    let sensor = match result.params.camera.sensor {
        vision_calibration::core::SensorParams::Scheimpflug { params } => params,
        other => panic!("unexpected sensor params: {other:?}"),
    };
    assert!((sensor.tilt_x - sensor_gt.tilt_x).abs() < 0.03);
    assert!((sensor.tilt_y - sensor_gt.tilt_y).abs() < 0.03);
}

#[test]
fn public_api_rejects_too_few_views() {
    let mut views = make_dataset(ScheimpflugParams::default()).views;
    views.truncate(2);
    let dataset = PlanarDataset::new(views).expect("dataset with 2 views");
    let err = run_calibration(&dataset, &ScheimpflugIntrinsicsCalibrationConfig::default())
        .expect_err("expected validation error");
    assert!(err.to_string().contains("need at least 3 views"));
}

#[test]
fn public_api_rejects_view_with_too_few_points() {
    let mut dataset = make_dataset(ScheimpflugParams::default());
    let first = &dataset.views[0].obs;
    let reduced = CorrespondenceView::new(
        first.points_3d.iter().take(3).copied().collect(),
        first.points_2d.iter().take(3).copied().collect(),
    )
    .expect("reduced observation");
    dataset.views[0] = View::without_meta(reduced);
    let err = run_calibration(&dataset, &ScheimpflugIntrinsicsCalibrationConfig::default())
        .expect_err("expected validation error");
    assert!(err.to_string().contains("has too few points"));
}

#[test]
fn public_api_rejects_invalid_config() {
    let dataset = make_dataset(ScheimpflugParams::default());
    let config = ScheimpflugIntrinsicsCalibrationConfig {
        init_iterations: 0,
        ..Default::default()
    };
    let err = run_calibration(&dataset, &config).expect_err("expected invalid config");
    assert!(err.to_string().contains("init_iterations must be positive"));
}

#[test]
fn scheimpflug_config_json_roundtrip() {
    let config = ScheimpflugIntrinsicsCalibrationConfig {
        init_iterations: 3,
        max_iters: 75,
        robust_loss: RobustLoss::Cauchy { scale: 0.9 },
        fix_scheimpflug: ScheimpflugFixMask {
            tilt_x: true,
            tilt_y: false,
        },
        ..Default::default()
    };
    let json = serde_json::to_string(&config).expect("serialize config");
    let restored: ScheimpflugIntrinsicsCalibrationConfig =
        serde_json::from_str(&json).expect("deserialize config");
    assert_eq!(restored.init_iterations, 3);
    assert_eq!(restored.max_iters, 75);
    assert!(matches!(
        restored.robust_loss,
        RobustLoss::Cauchy { scale } if (scale - 0.9).abs() < 1e-12
    ));
    assert!(restored.fix_scheimpflug.tilt_x);
    assert!(!restored.fix_scheimpflug.tilt_y);
}

#[test]
fn scheimpflug_result_json_roundtrip() {
    let dataset = make_dataset(ScheimpflugParams {
        tilt_x: 0.008,
        tilt_y: -0.006,
    });
    let result = run_calibration(&dataset, &ScheimpflugIntrinsicsCalibrationConfig::default())
        .expect("scheimpflug calibration");
    let json = serde_json::to_string(&result).expect("serialize result");
    let restored: ScheimpflugIntrinsicsResult =
        serde_json::from_str(&json).expect("deserialize result");
    assert!(restored.mean_reproj_error.is_finite());
    assert_eq!(restored.params.camera_se3_target.len(), dataset.num_views());
}
