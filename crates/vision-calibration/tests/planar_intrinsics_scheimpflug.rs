//! Planar intrinsics with a Scheimpflug camera, through the facade.

#![allow(missing_docs)]

use nalgebra::{Rotation3, Translation3};
use vision_calibration::common::IntrinsicsManualInit;
use vision_calibration::core::{
    BrownConrady5, Camera, CameraFix, CorrespondenceView, FxFyCxCy, Iso3, Pinhole, PlanarDataset,
    Pt2, Pt3, ScheimpflugParams, SensorKind, View,
};
use vision_calibration::optim::RobustLoss;
use vision_calibration::planar_intrinsics::{
    PlanarIntrinsicsConfig, PlanarIntrinsicsEstimate, PlanarIntrinsicsProblem, run_calibration,
    step_init_with_seed, step_optimize,
};

/// The default planar config with a Scheimpflug sensor.
fn tilted_config() -> PlanarIntrinsicsConfig {
    let mut config = PlanarIntrinsicsConfig::default();
    config.camera.model.sensor = SensorKind::Scheimpflug;
    config
}
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
        FxFyCxCy {
            fx: 800.0,
            fy: 780.0,
            cx: 640.0,
            cy: 360.0,
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

/// [`make_dataset`]'s views with deterministic noise, through a lens with
/// barrel distortion (`k1 = −0.1`): only the distortion, centred on the
/// lens axis, observes a sensor tilt, so without it the noisy tilt is
/// undetermined (its σ about 0.25 rad here).
fn make_noisy_dataset(sensor: ScheimpflugParams, noise_px: f64) -> PlanarDataset {
    let camera = Camera::new(
        Pinhole,
        BrownConrady5 {
            k1: -0.1,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            iters: 8,
        },
        sensor.compile(),
        FxFyCxCy {
            fx: 800.0,
            fy: 780.0,
            cx: 640.0,
            cy: 360.0,
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
    config: PlanarIntrinsicsConfig,
) -> anyhow::Result<PlanarIntrinsicsEstimate> {
    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    session.set_input(dataset.clone())?;
    session.set_config(config)?;
    run_calibration(&mut session)?;
    session
        .output()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("missing output after successful calibration"))
}

/// Poses that place the 0.25 m × 0.20 m board across the frame with enough
/// angular diversity to make a ≈−5° sensor tilt observable. Translation centers
/// the board on the optical axis (its origin is a corner, so shift by half its
/// extent) at ~0.45–0.54 m.
fn scheimpflug_rig_like_poses() -> Vec<Iso3> {
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

/// scheimpflug_rig_ref-like synthetic dataset: strong radial distortion (k1 ≈ −0.43) plus
/// a ≈−5° Scheimpflug tilt — the regime where Zhang-from-scratch underestimates
/// the focal and the LM settles into a wrong tilt/focal basin (see ADR 0011).
/// Returns the dataset and the ground-truth intrinsics /
/// distortion / sensor for recovery checks.
fn make_scheimpflug_rig_like_dataset() -> (
    PlanarDataset,
    FxFyCxCy<f64>,
    BrownConrady5<f64>,
    ScheimpflugParams,
) {
    let intr = FxFyCxCy {
        fx: 1153.0,
        fy: 1163.0,
        cx: 369.0,
        cy: 264.0,
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

    let views = scheimpflug_rig_like_poses()
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

/// The **supported** Scheimpflug workflow (ADR 0011): a *coarse*
/// user-provided focal seed, refined to the true optimum. On this scheimpflug_rig-like
/// strong-tilt + strong-distortion data, from-scratch Zhang lands in a wrong
/// basin; a coarse focal seed (≈9 % low, principal point at image center, no
/// distortion, no tilt) must converge well under the 0.5 px gate and recover the
/// true tilt basin and focal. Only the focal seed is load-bearing — the staged
/// sweep finds the tilt.
#[test]
fn seeded_coarse_prior_converges_on_strong_tilt_distortion() {
    let (dataset, gt_intr, _gt_dist, gt_sensor) = make_scheimpflug_rig_like_dataset();

    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    session.set_input(dataset).expect("input");
    session.set_config(tilted_config()).expect("config");

    // `IntrinsicsManualInit` is `#[non_exhaustive]`; build via Default + field
    // assignment. The realistic Scheimpflug prior: a coarse focal (≈9% low, pp at
    // image center) and the nominal mount tilt (≈−5°, a known mechanical spec).
    // distortion/poses stay None (auto). The seeded tilt is trusted directly.
    let mut seed = IntrinsicsManualInit::default();
    seed.intrinsics = Some(FxFyCxCy {
        fx: 1050.0,
        fy: 1050.0,
        cx: 360.0,
        cy: 270.0,
    });
    seed.sensor = Some(ScheimpflugParams {
        tilt_x: -0.087,
        tilt_y: 0.0,
    });
    step_init_with_seed(&mut session, seed, None).expect("seeded init");
    step_optimize(&mut session, None).expect("optimize");

    let result = session.output().expect("output");
    let intrinsics = &result.params.camera.intrinsics;
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

/// A tilt seed with automatically fitted intrinsics: the linear stage fits
/// the intrinsics at the seeded tilt. Fitted at the tilt its own sweep
/// finds (about zero on this 11° sensor), the start puts the principal point
/// 200 px off, the bounded solve ends on its `cy` bound with the tilt
/// 0.045 rad short and no covariance.
#[test]
fn tilt_seed_without_intrinsics_fits_the_intrinsics_at_the_seeded_tilt() {
    use vision_calibration::core::{
        CameraParams, DistortionParams, ProjectionParams, SensorParams,
    };
    use vision_calibration::synthetic::laser::BoardExtent;
    use vision_calibration::synthetic::noise::UniformPixelNoise;
    use vision_calibration::synthetic::{planar, poses};

    let gt_sensor = ScheimpflugParams {
        tilt_x: -0.20,
        tilt_y: 0.03,
    };
    let truth = CameraParams::new(
        ProjectionParams::Pinhole,
        DistortionParams::BrownConrady5 {
            params: BrownConrady5 {
                k1: -0.12,
                k2: 0.04,
                k3: 0.0,
                p1: 0.0,
                p2: 0.0,
                iters: 8,
            },
        },
        SensorParams::Scheimpflug { params: gt_sensor },
        FxFyCxCy {
            fx: 1000.0,
            fy: 990.0,
            cx: 640.0,
            cy: 480.0,
        },
    );
    let (nx, ny, spacing, views) = (13, 9, 0.04, 20);
    let center = BoardExtent::from_grid(nx, ny, spacing).center();
    let board_poses = poses::tilted_board_poses(
        &poses::seeded_tilts(views, 0.5, 31_007),
        (center[0], center[1]),
        0.6,
        0.4 / views as f64,
    );
    let observed = planar::project_views_noisy(
        &truth.build().expect("camera"),
        &planar::grid_points(nx, ny, spacing),
        &board_poses,
        &UniformPixelNoise {
            seed: 1000,
            max_abs_px: 0.5 * 3.0_f64.sqrt(),
        },
    )
    .expect("views");
    let dataset = PlanarDataset::new(observed.into_iter().map(View::without_meta).collect())
        .expect("dataset");

    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    session.set_input(dataset).expect("input");
    session.set_config(tilted_config()).expect("config");
    let mut seed = IntrinsicsManualInit::default();
    seed.sensor = Some(gt_sensor);
    let init = step_init_with_seed(&mut session, seed, None).expect("seeded init");
    assert!(
        (init.camera.intrinsics.cy - 480.0).abs() < 100.0,
        "initial cy {:.1}",
        init.camera.intrinsics.cy
    );
    step_optimize(&mut session, None).expect("optimize");

    let result = session.output().expect("output");
    let vision_calibration::core::SensorParams::Scheimpflug { params } =
        result.params.camera.sensor
    else {
        panic!("a tilted camera");
    };
    assert!(
        (params.tilt_x - gt_sensor.tilt_x).abs() < 0.005,
        "tilt_x {:.4}",
        params.tilt_x
    );
    assert!(
        (result.params.camera.intrinsics.fx - 1000.0).abs() < 5.0,
        "fx {:.2}",
        result.params.camera.intrinsics.fx
    );
    assert!(
        result.uncertainty.covariance().is_some(),
        "{:?}",
        result.uncertainty
    );
}

#[test]
fn public_api_converges_on_synthetic_scheimpflug_dataset() {
    let sensor_gt = ScheimpflugParams {
        tilt_x: 0.01,
        tilt_y: -0.008,
    };
    let dataset = make_dataset(sensor_gt);
    let result = run_pipeline(&dataset, tilted_config()).expect("scheimpflug calibration");
    let camera = result.params.camera;

    let intrinsics = camera.intrinsics;
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
    let mut config = tilted_config();
    config.optimize.reprojection_loss = RobustLoss::Huber { scale: 1.0 };

    let result = run_pipeline(&dataset, config).expect("scheimpflug calibration");
    assert!(result.mean_reproj_error < 2.0);

    let sensor = match result.params.camera.sensor {
        vision_calibration::core::SensorParams::Scheimpflug { params } => params,
        other => panic!("unexpected sensor params: {other:?}"),
    };
    assert!((sensor.tilt_x - sensor_gt.tilt_x).abs() < 0.03);
    assert!((sensor.tilt_y - sensor_gt.tilt_y).abs() < 0.03);
    let cov = result
        .uncertainty
        .covariance()
        .expect("a converged solve's covariance");
    for (label, est, truth) in [
        ("tilt_x", sensor.tilt_x, sensor_gt.tilt_x),
        ("tilt_y", sensor.tilt_y, sensor_gt.tilt_y),
    ] {
        let z = (est - truth).abs() / cov.std_dev(label).unwrap();
        assert!(z < 5.0, "{label}: {z:.2} σ off the truth");
    }
}

/// A free tilt with every distortion term held at zero is not determined
/// (without distortion a tilt is a rotation of the poses and a shift of the
/// principal point): refused before the solve, naming both ways out. Held
/// tilt, the same camera calibrates.
#[test]
fn a_free_tilt_with_the_distortion_held_at_zero_is_refused() {
    let dataset = make_dataset(ScheimpflugParams {
        tilt_x: 0.01,
        tilt_y: -0.008,
    });
    let all_distortion = || {
        ["k1", "k2", "k3", "p1", "p2"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>()
    };
    let mut config = tilted_config();
    config.camera.fix.distortion = Some(all_distortion());
    let err = run_pipeline(&dataset, config).expect_err("an undetermined tilt");
    let message = format!("{err:#}");
    assert!(message.contains("without lens distortion"), "{message}");
    assert!(
        message.contains("`k1`") && message.contains("tilt_x"),
        "{message}"
    );

    let mut held = tilted_config();
    held.camera.fix.distortion = Some(all_distortion());
    held.camera.fix.sensor = Some(vec!["tilt_x".into(), "tilt_y".into()]);
    run_pipeline(&dataset, held).expect("a held tilt calibrates");
}

#[test]
fn public_api_rejects_too_few_views() {
    let mut views = make_dataset(ScheimpflugParams::default()).views;
    views.truncate(2);
    let dataset = PlanarDataset::new(views).expect("dataset with 2 views");
    let err = run_pipeline(&dataset, tilted_config()).expect_err("expected validation error");
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
    let err = run_pipeline(&dataset, tilted_config()).expect_err("expected validation error");
    assert!(err.to_string().contains("has too few points"));
}

#[test]
fn public_api_rejects_invalid_config() {
    let dataset = make_dataset(ScheimpflugParams::default());
    let mut config = tilted_config();
    config.intrinsics_init.iterations = 0;
    let err = run_pipeline(&dataset, config).expect_err("expected invalid config");
    assert!(err.to_string().contains("iterations must be positive"));
}

#[test]
fn scheimpflug_config_json_roundtrip() {
    let mut config = tilted_config();
    config.intrinsics_init.iterations = 3;
    config.solver.max_iters = 75;
    config.optimize.reprojection_loss = RobustLoss::Cauchy { scale: 0.9 };
    config.camera.fix = CameraFix::default();
    config.camera.fix.sensor = Some(vec!["tilt_x".into()]);
    let json = serde_json::to_string(&config).expect("serialize config");
    let restored: PlanarIntrinsicsConfig = serde_json::from_str(&json).expect("deserialize config");
    assert_eq!(restored.intrinsics_init.iterations, 3);
    assert_eq!(restored.solver.max_iters, 75);
    assert!(matches!(
        restored.optimize.reprojection_loss,
        RobustLoss::Cauchy { scale } if (scale - 0.9).abs() < 1e-12
    ));
    assert_eq!(restored.camera, config.camera);
}

#[test]
fn scheimpflug_result_json_roundtrip() {
    let dataset = make_dataset(ScheimpflugParams {
        tilt_x: 0.008,
        tilt_y: -0.006,
    });
    let result = run_pipeline(&dataset, tilted_config()).expect("scheimpflug calibration");
    let json = serde_json::to_string(&result).expect("serialize result");
    let restored: PlanarIntrinsicsEstimate =
        serde_json::from_str(&json).expect("deserialize result");
    assert!(restored.mean_reproj_error.is_finite());
    assert_eq!(restored.params.cam_se3_target.len(), dataset.num_views());
}

/// The photometric centre's blur-gradient bias under a plane-of-focus blur
/// law is mostly absorbed by the camera and the poses, with a definite
/// aspect-ratio signature.
///
/// A ring-grid photometric fit lands `b = −k·σ·∇σ` (`k = 1.24`) off the
/// true centre under a blur gradient. With the defocus law
/// `σ² = σ0² + κ²·D²`, `D = 1/z − π·(x_n, y_n, 1)` (`π` the plane of sharp
/// focus in the camera frame, 1/m; `κ` px·m), `σ·∇σ = κ²·D·∇D` and
/// `∇D = −(π_x, π_y)/f`, so `b = (k·κ²/f)·(π_x, π_y)·D`. Its part linear in
/// `y_n`, `−(k·κ²/f)·π_y²·y_n`, is a rescaling of `v`: to first order
/// `Δ(fy/fx) = −k·κ²·π_y²/fx²`. On an exact scene close to the studied
/// rig, solving the seeded route on exact and on exact + `b` observations
/// must leave only a small share of `|b|²` in the residual, move `fy/fx` by
/// the predicted amount, and move the tilt little.
#[test]
fn blur_gradient_bias_is_absorbed_with_an_aspect_ratio_signature() {
    use vision_calibration::core::{CameraModelSpec, DistortionKind, SensorParams};

    let intr = FxFyCxCy {
        fx: 1145.0,
        fy: 1155.0,
        cx: 362.0,
        cy: 268.0,
    };
    let dist = BrownConrady5 {
        k1: -0.41,
        k2: 0.2,
        k3: 0.0,
        p1: 0.0,
        p2: 0.0,
        iters: 8,
    };
    let sensor = ScheimpflugParams {
        tilt_x: (-5.0_f64).to_radians(),
        tilt_y: 0.0,
    };
    let camera = Camera::new(Pinhole, dist, sensor.compile(), intr);
    let (w, h) = (720.0, 540.0);

    // A 0.24 m × 0.18 m board (12 mm pitch), centred on the line of sight
    // `(cx_off, cy_off)` at depth `z`, tilted about its centre.
    let spacing = 0.012;
    let board: Vec<Pt3> = (0..=20)
        .flat_map(|i| (0..=15).map(move |j| Pt3::new(i as f64 * spacing, j as f64 * spacing, 0.0)))
        .collect();
    let centre = nalgebra::Vector3::new(0.12, 0.09, 0.0);
    let specs: [(f64, f64, f64, f64, f64, f64); 16] = [
        (0.00, 0.00, 0.25, 0.00, 0.00, 0.00),
        (0.02, -0.01, 0.17, 0.30, -0.10, 0.05),
        (-0.02, 0.01, 0.19, -0.35, 0.15, -0.05),
        (0.01, 0.02, 0.21, 0.10, 0.40, 0.10),
        (-0.01, -0.02, 0.23, -0.15, -0.42, -0.08),
        (0.03, 0.01, 0.27, 0.45, 0.05, 0.00),
        (-0.03, -0.01, 0.29, -0.45, -0.05, 0.12),
        (0.02, 0.03, 0.31, 0.20, 0.30, -0.10),
        (-0.02, -0.03, 0.33, -0.25, -0.30, 0.06),
        (0.00, 0.02, 0.18, 0.40, 0.25, 0.00),
        (0.01, -0.02, 0.22, -0.40, 0.25, -0.04),
        (-0.01, 0.00, 0.26, 0.05, -0.45, 0.08),
        (0.02, 0.00, 0.30, -0.10, 0.45, -0.12),
        (-0.02, 0.02, 0.20, 0.30, 0.35, 0.03),
        (0.00, -0.03, 0.28, -0.30, -0.35, -0.03),
        (0.03, -0.02, 0.32, 0.15, -0.20, 0.15),
    ];
    let poses: Vec<Iso3> = specs
        .iter()
        .map(|&(tx, ty, tz, rx, ry, rz)| {
            let r = Rotation3::from_euler_angles(rx, ry, rz);
            let t = nalgebra::Vector3::new(tx, ty, tz) - r * centre;
            Iso3::from_parts(Translation3::from(t), r.into())
        })
        .collect();

    // The bias law: π = (0, −15, +3) /m, κ = 1.4 px·m, k = 1.24.
    let (pi_x, pi_y, pi_z) = (0.0, -15.0, 3.0);
    let (k, kappa) = (1.24, 1.4);
    let c = k * kappa * kappa / intr.fx;
    let mut exact = Vec::new();
    let mut biased = Vec::new();
    let mut bias: Vec<Vec<[f64; 2]>> = Vec::new();
    for pose in &poses {
        let (mut p3, mut p2, mut p2b, mut bv) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for point in &board {
            let pc = pose.transform_point(point);
            let Some(px) = camera.project_point(&pc) else {
                continue;
            };
            if !(0.0..w).contains(&px.x) || !(0.0..h).contains(&px.y) {
                continue;
            }
            let (xn, yn) = ((px.x - intr.cx) / intr.fx, (px.y - intr.cy) / intr.fy);
            let d = 1.0 / pc.z - (pi_x * xn + pi_y * yn + pi_z);
            let b = [c * pi_x * d, c * pi_y * d];
            p3.push(*point);
            p2.push(Pt2::new(px.x, px.y));
            p2b.push(Pt2::new(px.x + b[0], px.y + b[1]));
            bv.push(b);
        }
        exact.push(View::without_meta(
            CorrespondenceView::new(p3.clone(), p2).expect("obs"),
        ));
        biased.push(View::without_meta(
            CorrespondenceView::new(p3, p2b).expect("obs"),
        ));
        bias.push(bv);
    }
    let exact = PlanarDataset::new(exact).expect("dataset");
    let biased = PlanarDataset::new(biased).expect("dataset");

    // The accept path: the seeded route, bc5 under a tilted sensor, Huber.
    let solve = |ds: &PlanarDataset| {
        let mut config = tilted_config();
        config.camera.model =
            CameraModelSpec::pinhole(DistortionKind::BrownConrady5, SensorKind::Scheimpflug);
        config.solver.max_iters = 120;
        config.optimize.reprojection_loss = RobustLoss::Huber { scale: 1.0 };
        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
        session.set_input(ds.clone()).expect("input");
        session.set_config(config).expect("config");
        let mut seed = IntrinsicsManualInit::default();
        seed.intrinsics = Some(FxFyCxCy {
            fx: 1050.0,
            fy: 1050.0,
            cx: 360.0,
            cy: 270.0,
        });
        seed.sensor = Some(ScheimpflugParams {
            tilt_x: -0.087,
            tilt_y: 0.0,
        });
        step_init_with_seed(&mut session, seed, None).expect("seeded init");
        step_optimize(&mut session, None).expect("optimize");
        session.output().cloned().expect("output")
    };
    let residuals = |ds: &PlanarDataset, est: &PlanarIntrinsicsEstimate| -> Vec<[f64; 2]> {
        let model = est.params.camera.build().expect("camera");
        ds.views
            .iter()
            .zip(&est.params.cam_se3_target)
            .flat_map(|(view, pose)| {
                let model = &model;
                view.obs
                    .points_3d
                    .iter()
                    .zip(&view.obs.points_2d)
                    .map(move |(p3, p2)| {
                        let p = model
                            .project_point(&pose.transform_point(p3))
                            .expect("projects");
                        [p2.x - p.x, p2.y - p.y]
                    })
            })
            .collect()
    };
    let tilt = |est: &PlanarIntrinsicsEstimate| match est.params.camera.sensor {
        SensorParams::Scheimpflug { params } => [params.tilt_x, params.tilt_y],
        _ => panic!("a Scheimpflug sensor"),
    };

    let e0 = solve(&exact);
    let eb = solve(&biased);
    let (r0, rb) = (residuals(&exact, &e0), residuals(&biased, &eb));
    let bb: f64 = bias
        .iter()
        .flatten()
        .map(|b| b[0] * b[0] + b[1] * b[1])
        .sum();
    let left: f64 = r0
        .iter()
        .zip(&rb)
        .map(|(a, b)| (b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2))
        .sum::<f64>()
        / bb;
    let ratio = |e: &PlanarIntrinsicsEstimate| {
        e.params.camera.intrinsics.fy / e.params.camera.intrinsics.fx
    };
    let d_ratio = ratio(&eb) - ratio(&e0);
    let predicted = -k * kappa * kappa * pi_y * pi_y / (intr.fx * intr.fx);
    let (t0, tb) = (tilt(&e0), tilt(&eb));
    let d_tilt_deg = [(tb[0] - t0[0]).to_degrees(), (tb[1] - t0[1]).to_degrees()];
    let n = r0.len();
    let rms0 = (r0.iter().map(|r| r[0] * r[0] + r[1] * r[1]).sum::<f64>() / n as f64).sqrt();
    eprintln!(
        "points {n}, |b| rms {:.4} px, exact rms {rms0:.2e} px, share left {left:.3e}, \
         d(fy/fx) {d_ratio:+.4e} vs predicted {predicted:+.4e} (ratio {:.3}), d tilt {:+.4} / {:+.4} deg",
        (bb / n as f64).sqrt(),
        d_ratio / predicted,
        d_tilt_deg[0],
        d_tilt_deg[1]
    );
    // Measured: exact rms 1.3e-8 px, share left 8.0e-5, d(fy/fx) −4.263e-4
    // against −4.171e-4 predicted (ratio 1.022), d tilt −0.0062° / −0.0001°.
    // Margins: share 25×, the ratio's error 7×, the tilt about 5×.
    assert!(
        rms0 < 1e-6,
        "the exact scene is not fitted exactly: {rms0:.2e} px"
    );
    assert!(left < 2e-3, "share of |b|² left in the residual {left:.3e}");
    assert!(
        (d_ratio / predicted - 1.0).abs() < 0.15,
        "d(fy/fx) {d_ratio:+.4e} against the first-order {predicted:+.4e}"
    );
    assert!(
        d_tilt_deg[0].abs() < 0.03 && d_tilt_deg[1].abs() < 0.03,
        "the tilt moved {:+.4} / {:+.4} deg",
        d_tilt_deg[0],
        d_tilt_deg[1]
    );
}
