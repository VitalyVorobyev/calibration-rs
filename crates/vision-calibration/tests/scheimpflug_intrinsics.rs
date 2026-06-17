#![allow(missing_docs)]

use nalgebra::{Rotation3, Translation3};
use vision_calibration::core::{
    BrownConrady5, Camera, CorrespondenceView, FxFyCxCySkew, Iso3, Pinhole, PlanarDataset, Pt2,
    Pt3, ScheimpflugParams, View,
};
use vision_calibration::optim::RobustLoss;
use vision_calibration::scheimpflug_intrinsics::{
    ScheimpflugFixMask, ScheimpflugIntrinsicsConfig, ScheimpflugIntrinsicsProblem,
    ScheimpflugIntrinsicsResult, ScheimpflugManualInit, run_calibration, step_init_with_seed,
    step_optimize,
};
use vision_calibration::session::CalibrationSession;

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

fn run_pipeline(
    dataset: &PlanarDataset,
    config: ScheimpflugIntrinsicsConfig,
) -> anyhow::Result<ScheimpflugIntrinsicsResult> {
    let mut session = CalibrationSession::<ScheimpflugIntrinsicsProblem>::new();
    session.set_input(dataset.clone())?;
    run_calibration(&mut session, Some(config))?;
    session
        .output()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("missing output after successful calibration"))
}

/// Poses that place the 0.25 m × 0.20 m board across the frame with enough
/// angular diversity to make a ≈−5° sensor tilt observable. Translation centers
/// the board on the optical axis (its origin is a corner, so shift by half its
/// extent) at ~0.45–0.54 m.
fn rtv3d_like_poses() -> Vec<Iso3> {
    let (bx, by) = (-0.125, -0.10);
    let specs = [
        (0.0, 0.0, 0.45, 0.0, 0.0, 0.0),
        (0.03, -0.02, 0.48, 0.12, -0.06, 0.0),
        (-0.03, 0.02, 0.52, -0.10, 0.08, 0.03),
        (0.02, 0.04, 0.46, 0.06, 0.14, -0.04),
        (-0.05, -0.03, 0.50, -0.13, -0.05, 0.06),
        (0.04, 0.01, 0.44, 0.10, 0.02, -0.02),
        (-0.02, -0.05, 0.54, -0.04, -0.12, 0.05),
        (0.05, 0.03, 0.47, 0.15, 0.07, 0.0),
        (-0.04, 0.05, 0.49, -0.08, 0.11, -0.03),
        (0.01, -0.04, 0.53, 0.03, -0.10, 0.04),
        (0.03, 0.05, 0.45, 0.09, 0.13, 0.02),
        (-0.05, 0.0, 0.51, -0.14, 0.0, -0.05),
    ];
    specs
        .iter()
        .map(|&(tx, ty, tz, rx, ry, rz)| {
            Iso3::from_parts(
                Translation3::new(bx + tx, by + ty, tz),
                Rotation3::from_euler_angles(rx, ry, rz).into(),
            )
        })
        .collect()
}

/// rtv3d_ref-like synthetic dataset: strong radial distortion (k1 ≈ −0.43) plus
/// a ≈−5° Scheimpflug tilt — the regime where Zhang-from-scratch underestimates
/// the focal and the LM settles into a wrong tilt/focal basin (see ADR 0022 and
/// the P6 diagnosis). Returns the dataset and the ground-truth intrinsics /
/// distortion / sensor for recovery checks.
fn make_rtv3d_like_dataset() -> (
    PlanarDataset,
    FxFyCxCySkew<f64>,
    BrownConrady5<f64>,
    ScheimpflugParams,
) {
    let intr = FxFyCxCySkew {
        fx: 1153.0,
        fy: 1163.0,
        cx: 369.0,
        cy: 264.0,
        skew: 0.0,
    };
    let dist = BrownConrady5 {
        k1: -0.43,
        k2: 0.34,
        k3: 0.0,
        p1: 0.0,
        p2: 0.0,
        iters: 8,
    };
    let sensor = ScheimpflugParams {
        tilt_x: -0.09,
        tilt_y: 0.006,
    };
    let camera = Camera::new(Pinhole, dist, sensor.compile(), intr);

    let spacing = 0.025;
    let mut board = Vec::new();
    for i in 0..11 {
        for j in 0..9 {
            board.push(Pt3::new(i as f64 * spacing, j as f64 * spacing, 0.0));
        }
    }

    let views = rtv3d_like_poses()
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
    (
        PlanarDataset::new(views).expect("dataset"),
        intr,
        dist,
        sensor,
    )
}

/// The **supported** Scheimpflug-intrinsics workflow (ADR 0022): a *coarse*
/// user-provided focal seed, refined to the true optimum. On this rtv3d-like
/// strong-tilt + strong-distortion data, from-scratch Zhang lands in a wrong
/// basin; a coarse focal seed (≈9 % low, principal point at image center, no
/// distortion, no tilt) must converge well under the 0.5 px gate and recover the
/// true tilt basin and focal. Only the focal seed is load-bearing — the staged
/// sweep finds the tilt.
#[test]
fn seeded_coarse_prior_converges_on_strong_tilt_distortion() {
    let (dataset, gt_intr, _gt_dist, gt_sensor) = make_rtv3d_like_dataset();

    let mut session = CalibrationSession::<ScheimpflugIntrinsicsProblem>::new();
    session.set_input(dataset).expect("input");

    let mut config = ScheimpflugIntrinsicsConfig::default();
    config.fix_scheimpflug = ScheimpflugFixMask {
        tilt_x: false,
        tilt_y: false,
    };
    session.set_config(config).expect("config");

    // `ScheimpflugManualInit` is `#[non_exhaustive]`; build via Default + field
    // assignment. The realistic Scheimpflug prior: a coarse focal (≈9% low, pp at
    // image center) and the nominal mount tilt (≈−5°, a known mechanical spec).
    // distortion/poses stay None (auto). The seeded tilt is trusted directly.
    let mut seed = ScheimpflugManualInit::default();
    seed.intrinsics = Some(FxFyCxCySkew {
        fx: 1050.0,
        fy: 1050.0,
        cx: 360.0,
        cy: 270.0,
        skew: 0.0,
    });
    seed.sensor = Some(ScheimpflugParams {
        tilt_x: -0.087,
        tilt_y: 0.0,
    });
    step_init_with_seed(&mut session, seed, None).expect("seeded init");
    step_optimize(&mut session, None).expect("optimize");

    let result = session.output().expect("output");
    let intrinsics = match &result.params.camera.intrinsics {
        vision_calibration::core::IntrinsicsParams::FxFyCxCySkew { params } => *params,
    };
    let sensor = match &result.params.camera.sensor {
        vision_calibration::core::SensorParams::Scheimpflug { params } => *params,
        other => panic!("unexpected sensor params: {other:?}"),
    };

    assert!(
        result.mean_reproj_error < 0.1,
        "seeded reproj {:.4}px exceeds the noise-free margin (0.5px gate)",
        result.mean_reproj_error
    );
    assert!(
        (intrinsics.fx - gt_intr.fx).abs() / gt_intr.fx < 0.02,
        "fx {:.1} not within 2% of {:.1}",
        intrinsics.fx,
        gt_intr.fx
    );
    assert!(
        (sensor.tilt_x - gt_sensor.tilt_x).abs() < 0.01,
        "tilt_x {:.4} not within 0.01 rad of {:.4}",
        sensor.tilt_x,
        gt_sensor.tilt_x
    );
    assert!(
        (sensor.tilt_y - gt_sensor.tilt_y).abs() < 0.01,
        "tilt_y {:.4} not within 0.01 rad of {:.4}",
        sensor.tilt_y,
        gt_sensor.tilt_y
    );
}

#[test]
fn public_api_converges_on_synthetic_scheimpflug_dataset() {
    let sensor_gt = ScheimpflugParams {
        tilt_x: 0.01,
        tilt_y: -0.008,
    };
    let dataset = make_dataset(sensor_gt);
    let mut config = ScheimpflugIntrinsicsConfig::default();
    config.fix_scheimpflug = vision_calibration::scheimpflug_intrinsics::ScheimpflugFixMask {
        tilt_x: false,
        tilt_y: false,
    };

    let result = run_pipeline(&dataset, config).expect("scheimpflug calibration");
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
    assert!((sensor.tilt_x - sensor_gt.tilt_x).abs() < 0.03);
    assert!((sensor.tilt_y - sensor_gt.tilt_y).abs() < 0.03);
}

#[test]
fn public_api_converges_with_deterministic_noise() {
    let sensor_gt = ScheimpflugParams {
        tilt_x: 0.01,
        tilt_y: -0.008,
    };
    let dataset = make_noisy_dataset(sensor_gt, 0.15);
    let mut config = ScheimpflugIntrinsicsConfig::default();
    config.robust_loss = RobustLoss::Huber { scale: 1.0 };
    config.fix_scheimpflug = ScheimpflugFixMask {
        tilt_x: false,
        tilt_y: false,
    };

    let result = run_pipeline(&dataset, config).expect("scheimpflug calibration");
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
    let err = run_pipeline(&dataset, ScheimpflugIntrinsicsConfig::default())
        .expect_err("expected validation error");
    assert!(err.to_string().contains("need 3"));
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
    let err = run_pipeline(&dataset, ScheimpflugIntrinsicsConfig::default())
        .expect_err("expected validation error");
    assert!(err.to_string().contains("has too few points"));
}

#[test]
fn public_api_rejects_invalid_config() {
    let dataset = make_dataset(ScheimpflugParams::default());
    let mut config = ScheimpflugIntrinsicsConfig::default();
    config.init_iterations = 0;
    let err = run_pipeline(&dataset, config).expect_err("expected invalid config");
    assert!(err.to_string().contains("init_iterations must be positive"));
}

#[test]
fn scheimpflug_config_json_roundtrip() {
    let mut config = ScheimpflugIntrinsicsConfig::default();
    config.init_iterations = 3;
    config.max_iters = 75;
    config.robust_loss = RobustLoss::Cauchy { scale: 0.9 };
    config.fix_scheimpflug = ScheimpflugFixMask {
        tilt_x: true,
        tilt_y: false,
    };
    let json = serde_json::to_string(&config).expect("serialize config");
    let restored: ScheimpflugIntrinsicsConfig =
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
    let result = run_pipeline(&dataset, ScheimpflugIntrinsicsConfig::default())
        .expect("scheimpflug calibration");
    let json = serde_json::to_string(&result).expect("serialize result");
    let restored: ScheimpflugIntrinsicsResult =
        serde_json::from_str(&json).expect("deserialize result");
    assert!(restored.mean_reproj_error.is_finite());
    assert_eq!(restored.params.camera_se3_target.len(), dataset.num_views());
}
