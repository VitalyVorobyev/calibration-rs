//! Tests of the quality checks: each check on hand-made inputs, the image-size
//! rules, and two synthetic calibrations that differ only in how much of the
//! image their views covered.

use std::path::PathBuf;

use nalgebra::{Rotation3, Translation3, Vector3};
use vision_calibration_core::{
    BrownConrady5, CorrespondenceView, FrameRef, FxFyCxCy, Iso3, PinholeCamera, PixelRect,
    PlanarDataset, PointCovariance, View, ViewCovariance, make_pinhole_camera, synthetic::planar,
};

use super::*;
use crate::planar_intrinsics::{PlanarIntrinsicsExport, PlanarIntrinsicsProblem, run_calibration};
use crate::session::CalibrationSession;

// ─────────────────────────────────────────────────────────────────────────────
// Fixtures
// ─────────────────────────────────────────────────────────────────────────────

const IMAGE: [u32; 2] = [1280, 960];

fn covariance(labels: &[&str], matrix: Vec<Vec<f64>>, sigma: f64) -> ParameterCovariance {
    ParameterCovariance {
        labels: labels.iter().map(|l| l.to_string()).collect(),
        covariance: matrix,
        views: Vec::<ViewCovariance>::new(),
        fixed: Vec::new(),
        noise: vec![ResidualNoise {
            kind: ResidualKind::Reprojection,
            level: NoiseLevel::Sigma(sigma),
            count: 2000,
            camera: None,
        }],
        num_residuals: 2000,
        dof: 1900,
    }
}

fn unavailable(reason: UncertaintyUnavailable) -> Uncertainty {
    Uncertainty::Unavailable { reason }
}

#[allow(clippy::field_reassign_with_default)]
fn residual(pose: usize, camera: usize, observed: [f64; 2], error: f64) -> TargetFeatureResidual {
    let mut r = TargetFeatureResidual::default();
    r.pose = pose;
    r.camera = camera;
    r.observed_px = observed;
    r.projected_px = Some(observed);
    r.error_px = Some(error);
    r
}

#[allow(clippy::field_reassign_with_default)]
fn frame(pose: usize, camera: usize, roi: Option<[u32; 2]>) -> FrameRef {
    let mut f = FrameRef::default();
    f.pose = pose;
    f.camera = camera;
    f.path = PathBuf::from(format!("pose{pose}.png"));
    f.roi = roi.map(|[w, h]| {
        let mut r = PixelRect::default();
        r.x = camera as u32 * w;
        r.w = w;
        r.h = h;
        r
    });
    f
}

fn manifest(frames: Vec<FrameRef>) -> ImageManifest {
    let mut m = ImageManifest::default();
    m.frames = frames;
    m
}

fn camera_model() -> PinholeCamera {
    make_pinhole_camera(
        FxFyCxCy {
            fx: 1000.0,
            fy: 1000.0,
            cx: 640.0,
            cy: 480.0,
        },
        BrownConrady5 {
            k1: -0.08,
            k2: 0.02,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            iters: 8,
        },
    )
}

/// A 9 × 7 board of 30 mm cells, posed so its centre lands on pixel `at`
/// at depth `z`, tilted by `(pitch, yaw)`.
fn board_pose(at: [f64; 2], z: f64, pitch: f64, yaw: f64) -> Iso3 {
    let center = Vector3::new(0.12, 0.09, 0.0);
    let r = Rotation3::from_euler_angles(pitch, yaw, 0.0);
    let ray = Vector3::new((at[0] - 640.0) / 1000.0, (at[1] - 480.0) / 1000.0, 1.0);
    Iso3::from_parts(Translation3::from(ray * z - r * center), r.into())
}

/// Per-axis σ of the synthetic corner noise, pixels.
const NOISE_PX: f64 = 0.1;

/// Calibrate a synthetic camera from board poses, with seeded Gaussian corner
/// noise of `NOISE_PX` per axis.
fn calibrate(poses: &[Iso3]) -> PlanarIntrinsicsExport {
    calibrate_stating(poses, None)
}

/// [`calibrate`], every corner stating `cov` when given.
fn calibrate_stating(poses: &[Iso3], cov: Option<PointCovariance>) -> PlanarIntrinsicsExport {
    let camera = camera_model();
    let board = planar::grid_points(9, 7, 0.03);
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    // Box–Muller.
    let mut noise = move || {
        let (u, v) = (uniform(), uniform());
        NOISE_PX * (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    };
    let views = planar::project_views_all(&camera, &board, poses)
        .expect("every corner projects")
        .into_iter()
        .map(|mut view| {
            for p in &mut view.points_2d {
                p.x += noise();
                p.y += noise();
            }
            let view = match cov {
                None => view,
                Some(c) => CorrespondenceView::new_with_covariances(
                    view.points_3d.clone(),
                    view.points_2d.clone(),
                    vec![c; view.len()],
                )
                .expect("one covariance per corner"),
            };
            View::without_meta(view)
        })
        .collect();
    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    session
        .set_input(PlanarDataset::new(views).expect("views"))
        .expect("input");
    run_calibration(&mut session).expect("calibration");
    session.export().expect("export")
}

/// Board poses on a grid of image positions, each tilted toward the image
/// centre and alternately rolled about it, at depth `z`.
fn grid_poses(us: &[f64], vs: &[f64], z: f64) -> Vec<Iso3> {
    let mut poses = Vec::new();
    for (i, &v) in vs.iter().enumerate() {
        for (j, &u) in us.iter().enumerate() {
            let sign = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
            let pitch = 0.35 * sign;
            let yaw = 0.35 * (j as f64 / (us.len() - 1) as f64 * 2.0 - 1.0);
            poses.push(board_pose([u, v], z, pitch, yaw));
        }
    }
    poses
}

/// Twenty tilted views spread over the whole image.
fn well_covered_poses() -> Vec<Iso3> {
    grid_poses(
        &[260.0, 450.0, 640.0, 830.0, 1020.0],
        &[200.0, 380.0, 580.0, 760.0],
        0.5,
    )
}

/// Twenty views as tilted as the well-covered ones, all in the top-left
/// quadrant of the image.
fn one_quadrant_poses() -> Vec<Iso3> {
    grid_poses(
        &[250.0, 290.0, 330.0, 370.0, 410.0],
        &[190.0, 230.0, 270.0, 310.0],
        1.1,
    )
}

fn check(report: &QualityReport, kind: CheckKind) -> &QualityCheck {
    report
        .checks
        .iter()
        .find(|c| c.kind == kind)
        .unwrap_or_else(|| panic!("no {kind:?} check"))
}

// ─────────────────────────────────────────────────────────────────────────────
// Synthetic proof
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn views_in_one_quadrant_fail_coverage_and_spread_views_pass() {
    let thresholds = QualityThresholds::default();

    let spread = assess(&calibrate(&well_covered_poses()), &[IMAGE], &thresholds).unwrap();
    let coverage = check(&spread, CheckKind::FeatureCoverage);
    assert_eq!(coverage.status, CheckStatus::Pass, "{}", coverage.message);
    assert!(coverage.measurement.unwrap().value >= 0.75);
    assert_eq!(spread.verdict, Verdict::Pass, "{spread:#?}");

    let bunched = assess(&calibrate(&one_quadrant_poses()), &[IMAGE], &thresholds).unwrap();
    let coverage = check(&bunched, CheckKind::FeatureCoverage);
    assert_eq!(coverage.status, CheckStatus::Fail, "{}", coverage.message);
    assert!(coverage.measurement.unwrap().value <= 0.25);
    assert_eq!(bunched.verdict, Verdict::Fail);
    // The map of the bunched views extrapolates over the rest of the image.
    let projection = check(&bunched, CheckKind::ProjectionSigma);
    let spread_projection = check(&spread, CheckKind::ProjectionSigma);
    assert!(
        projection.measurement.unwrap().value > 3.0 * spread_projection.measurement.unwrap().value
    );
}

#[test]
fn stated_noise_is_tested_against_the_estimate() {
    let export = calibrate(&well_covered_poses());
    let true_sigma = NOISE_PX;
    let status = |sigma0: f64| {
        let thresholds = QualityThresholds {
            expected_feature_noise_px: Some(sigma0),
            ..QualityThresholds::default()
        };
        let report = assess(&export, &[IMAGE], &thresholds).unwrap();
        check(&report, CheckKind::VarianceFactor).status
    };
    assert_eq!(status(true_sigma), CheckStatus::Pass);
    assert_eq!(status(true_sigma / 2.0), CheckStatus::Fail);
    assert_eq!(status(true_sigma * 2.0), CheckStatus::Warn);
}

#[test]
fn report_round_trips_through_json() {
    let report = assess(
        &calibrate(&well_covered_poses()),
        &[IMAGE],
        &QualityThresholds::default(),
    )
    .unwrap();
    let json = serde_json::to_value(&report).unwrap();
    assert_eq!(json["verdict"], "pass");
    assert_eq!(json["checks"][0]["kind"], "observability");
    assert!(json["checks"][0].get("camera").is_none());
    let back: QualityReport = serde_json::from_value(json).unwrap();
    assert_eq!(back, report);
}

// ─────────────────────────────────────────────────────────────────────────────
// Image sizes
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn image_size_comes_from_manifest_regions_then_the_caller() {
    let tiled = manifest(vec![
        frame(0, 0, Some([720, 540])),
        frame(0, 1, Some([720, 540])),
    ]);
    assert_eq!(
        resolve_image_sizes(Some(&tiled), 2, &[]).unwrap(),
        vec![Some([720, 540]), Some([720, 540])]
    );
    assert_eq!(
        resolve_image_sizes(Some(&tiled), 2, &[[720, 540], [720, 540]]).unwrap(),
        vec![Some([720, 540]), Some([720, 540])]
    );
    let whole = manifest(vec![frame(0, 0, None)]);
    assert_eq!(
        resolve_image_sizes(Some(&whole), 1, &[]).unwrap(),
        vec![None]
    );
    assert_eq!(
        resolve_image_sizes(Some(&whole), 1, &[IMAGE]).unwrap(),
        vec![Some(IMAGE)]
    );
    assert_eq!(resolve_image_sizes(None, 1, &[]).unwrap(), vec![None]);
}

#[test]
fn ambiguous_image_sizes_are_errors() {
    let tiled = manifest(vec![frame(0, 0, Some([720, 540]))]);
    assert!(resolve_image_sizes(Some(&tiled), 1, &[IMAGE]).is_err());
    assert!(resolve_image_sizes(None, 2, &[IMAGE]).is_err());
    assert!(resolve_image_sizes(None, 1, &[[0, 960]]).is_err());
    let differing = manifest(vec![
        frame(0, 0, Some([720, 540])),
        frame(1, 0, Some([700, 540])),
    ]);
    assert!(resolve_image_sizes(Some(&differing), 1, &[]).is_err());
    let mixed = manifest(vec![frame(0, 0, Some([720, 540])), frame(1, 0, None)]);
    assert!(resolve_image_sizes(Some(&mixed), 1, &[]).is_err());
}

#[test]
fn without_an_image_size_the_image_checks_are_not_applicable() {
    let report = assess(
        &calibrate(&well_covered_poses()),
        &[],
        &QualityThresholds::default(),
    )
    .unwrap();
    for kind in [CheckKind::ProjectionSigma, CheckKind::FeatureCoverage] {
        let c = check(&report, kind);
        assert_eq!(c.status, CheckStatus::NotApplicable);
        assert!(c.measurement.is_none());
        assert!(c.message.contains("image size"), "{}", c.message);
    }
    assert_eq!(report.verdict, Verdict::Pass);
}

// ─────────────────────────────────────────────────────────────────────────────
// Thresholds
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn thresholds_fill_defaults_and_reject_unknown_fields() {
    let parsed: QualityThresholds =
        serde_json::from_str(r#"{"expected_feature_noise_px": 0.2}"#).unwrap();
    let expected = QualityThresholds {
        expected_feature_noise_px: Some(0.2),
        ..QualityThresholds::default()
    };
    assert_eq!(parsed, expected);
    let json = serde_json::to_string(&expected).unwrap();
    assert_eq!(
        serde_json::from_str::<QualityThresholds>(&json).unwrap(),
        expected
    );
    assert!(serde_json::from_str::<QualityThresholds>(r#"{"coverage": 0.5}"#).is_err());
}

#[test]
fn out_of_range_thresholds_are_errors() {
    let export = calibrate(&well_covered_poses());
    let invalid: [fn(&mut QualityThresholds); 6] = [
        |t| t.expected_feature_noise_px = Some(0.0),
        |t| t.variance_test_alpha = 1.0,
        |t| t.max_projection_sigma_px = f64::NAN,
        |t| t.coverage_fail_below = 0.6,
        |t| t.max_view_residual = -1.0,
        |t| t.max_parameter_correlation = 1.5,
    ];
    for set in invalid {
        let mut thresholds = QualityThresholds::default();
        set(&mut thresholds);
        assert!(
            assess(&export, &[IMAGE], &thresholds).is_err(),
            "{thresholds:?}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Individual checks
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn observability_follows_the_uncertainty_status() {
    let available = Uncertainty::Available(covariance(&["fx"], vec![vec![1.0]], 0.1));
    assert_eq!(observability(&available).status, CheckStatus::Pass);

    let rank = observability(&unavailable(UncertaintyUnavailable::RankDeficient {
        unobservable: vec!["handeye.tz".into(), "target.tz".into()],
    }));
    assert_eq!(rank.status, CheckStatus::Fail);
    assert_eq!(rank.subjects, vec!["handeye.tz", "target.tz"]);
    assert!(rank.message.contains("handeye.tz"));

    let bound = observability(&unavailable(UncertaintyUnavailable::AtBound {
        labels: vec!["k1".into()],
    }));
    assert_eq!(bound.status, CheckStatus::Warn);
    assert_eq!(bound.subjects, vec!["k1"]);

    assert_eq!(
        observability(&unavailable(UncertaintyUnavailable::NoRedundancy)).status,
        CheckStatus::Fail
    );
    for reason in [
        UncertaintyUnavailable::NotConverged,
        UncertaintyUnavailable::NotComputed,
    ] {
        assert_eq!(
            observability(&unavailable(reason)).status,
            CheckStatus::NotApplicable
        );
    }
}

#[test]
fn convergence_reads_the_solve_status() {
    assert_eq!(
        convergence(&unavailable(UncertaintyUnavailable::NotConverged)).status,
        CheckStatus::Fail
    );
    assert_eq!(
        convergence(&unavailable(UncertaintyUnavailable::NotComputed)).status,
        CheckStatus::NotApplicable
    );
    // A rank-deficient covariance is analysed only after the solve converged.
    assert_eq!(
        convergence(&unavailable(UncertaintyUnavailable::RankDeficient {
            unobservable: vec![]
        }))
        .status,
        CheckStatus::Pass
    );
}

#[test]
fn variance_factor_needs_a_stated_noise_and_a_covariance() {
    let available = Uncertainty::Available(covariance(&["fx"], vec![vec![1.0]], 0.1));
    let defaults = QualityThresholds::default();
    let checks = variance_factor(&available, &defaults);
    assert_eq!(checks.len(), 1);
    assert_eq!(checks[0].status, CheckStatus::NotApplicable);

    let mut stated = defaults;
    stated.expected_feature_noise_px = Some(0.1);
    let checks = variance_factor(&available, &stated);
    assert_eq!(checks[0].status, CheckStatus::Pass);
    let m = checks[0].measurement.unwrap();
    assert_eq!(m.unit, Unit::Ratio);
    assert!((m.value - 1.0).abs() < 1e-12);

    let checks = variance_factor(&unavailable(UncertaintyUnavailable::NotComputed), &stated);
    assert_eq!(checks[0].status, CheckStatus::NotApplicable);
}

/// Stating the true isotropic covariance `σ² I` on every corner gives the
/// unweighted solve's estimate and covariance, with the noise reported as
/// `σ̂₀ ≈ 1` instead of `σ̂ ≈ σ` px, whitened per-feature errors `|e|/σ`
/// beside the pixel ones, and pixel reprojection statistics; the variance
/// factor is then tested without a stated pixel noise.
#[test]
fn a_stated_isotropic_covariance_reproduces_the_unweighted_solve() {
    let poses = well_covered_poses();
    let plain = calibrate(&poses);
    let var = NOISE_PX * NOISE_PX;
    let stated = calibrate_stating(&poses, Some(PointCovariance::new(var, 0.0, var).unwrap()));

    let (a, b) = (
        plain.params.camera.intrinsics,
        stated.params.camera.intrinsics,
    );
    assert!(
        (a.fx - b.fx).abs() < 1e-6 && (a.cx - b.cx).abs() < 1e-6,
        "{a:?} vs {b:?}"
    );
    assert!((plain.mean_reproj_error - stated.mean_reproj_error).abs() < 1e-9);
    let (Uncertainty::Available(cp), Uncertainty::Available(cs)) =
        (&plain.uncertainty, &stated.uncertainty)
    else {
        panic!("both have a covariance");
    };
    let (sp, ss) = (cp.std_dev("fx").unwrap(), cs.std_dev("fx").unwrap());
    assert!((sp - ss).abs() < 1e-6 * sp, "σ(fx) {sp} vs {ss}");
    let NoiseLevel::Sigma(px) = cp.noise[0].level else {
        panic!("pixel noise without stated covariances");
    };
    let NoiseLevel::Sigma0(s0) = cs.noise[0].level else {
        panic!("whitened noise with stated covariances");
    };
    assert!((s0 - px / NOISE_PX).abs() < 1e-9, "σ̂₀ {s0} vs σ̂ {px} px");
    assert!((s0 - 1.0).abs() < 0.1, "σ̂₀ = {s0}");

    for (p, s) in plain
        .per_feature_residuals
        .target
        .iter()
        .zip(&stated.per_feature_residuals.target)
    {
        assert!((p.error_px.unwrap() - s.error_px.unwrap()).abs() < 1e-9);
        assert_eq!(p.error_whitened, None);
        let (e, w) = (s.error_px.unwrap(), s.error_whitened.unwrap());
        assert!((w - e / NOISE_PX).abs() < 1e-9 * w.max(1.0));
    }
    let report = assess(&stated, &[IMAGE], &QualityThresholds::default()).unwrap();
    assert_eq!(
        check(&report, CheckKind::VarianceFactor).status,
        CheckStatus::Pass
    );
    assert_eq!(
        check(&report, CheckKind::ViewOutliers).status,
        CheckStatus::Pass
    );
}

/// Residuals whitened by stated covariances are tested against 1, with no
/// stated pixel noise; a stated one does not apply to them.
#[test]
fn whitened_noise_is_tested_against_one() {
    let mut cov = covariance(&["fx"], vec![vec![1.0]], 0.0);
    cov.noise[0].level = NoiseLevel::Sigma0(1.02);
    let whitened = Uncertainty::Available(cov.clone());
    for expected in [None, Some(0.3)] {
        let thresholds = QualityThresholds {
            expected_feature_noise_px: expected,
            ..QualityThresholds::default()
        };
        let checks = variance_factor(&whitened, &thresholds);
        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].status, CheckStatus::Pass, "{}", checks[0].message);
        assert!((checks[0].measurement.unwrap().value - 1.02 * 1.02).abs() < 1e-12);
        assert!(checks[0].message.contains("σ̂₀"), "{}", checks[0].message);
    }
    cov.noise[0].level = NoiseLevel::Sigma0(1.5);
    let checks = variance_factor(&Uncertainty::Available(cov), &QualityThresholds::default());
    assert_eq!(checks[0].status, CheckStatus::Fail);
}

/// With a whitened noise level the outlier check normalizes the whitened
/// errors; residuals without one leave it nothing to assess.
#[test]
fn view_outliers_use_whitened_errors_with_a_whitened_noise() {
    let mut cov = covariance(&["fx"], vec![vec![1.0]], 0.0);
    cov.noise[0].level = NoiseLevel::Sigma0(1.0);
    let uncertainty = Uncertainty::Available(cov);
    let mut residuals = Vec::new();
    for view in 0..5 {
        // Pixel errors alike; the whitened ones single out view 2.
        let whitened = if view == 2 { 4.0 } else { 1.0 };
        for i in 0..10 {
            let mut r = residual(view, 0, [i as f64, 0.0], 0.5);
            r.error_whitened = Some(whitened);
            residuals.push(r);
        }
    }
    let thresholds = QualityThresholds::default();
    let c = view_outliers(&residuals, &uncertainty, 0, &thresholds);
    assert_eq!(c.status, CheckStatus::Warn);
    assert_eq!(c.subjects, vec!["view 2"]);
    assert!((c.measurement.unwrap().value - 4.0).abs() < 1e-12);

    for r in &mut residuals {
        r.error_whitened = None;
    }
    assert_eq!(
        view_outliers(&residuals, &uncertainty, 0, &thresholds).status,
        CheckStatus::NotApplicable
    );
}

#[test]
fn coverage_counts_grid_cells_with_a_feature() {
    let thresholds = QualityThresholds::default();
    // One feature in the centre of each of the 48 cells: full coverage.
    let full: Vec<_> = (0..48)
        .map(|i| {
            let (c, r) = ((i % 8) as f64, (i / 8) as f64);
            residual(0, 0, [(c + 0.5) * 160.0, (r + 0.5) * 160.0], 0.1)
        })
        .collect();
    let c = feature_coverage(&full, 0, Some(IMAGE), &thresholds);
    assert_eq!(c.status, CheckStatus::Pass);
    assert_eq!(c.measurement.unwrap().value, 1.0);
    // The same features for camera 1 do not count for camera 0.
    let other: Vec<_> = full
        .iter()
        .cloned()
        .map(|mut r| {
            r.camera = 1;
            r
        })
        .collect();
    assert_eq!(
        feature_coverage(&other, 0, Some(IMAGE), &thresholds).status,
        CheckStatus::NotApplicable
    );
    // Features in 20 of 48 cells warn; in 10, fail.
    let warn = feature_coverage(&full[..20], 0, Some(IMAGE), &thresholds);
    assert_eq!(warn.status, CheckStatus::Warn);
    assert_eq!(warn.measurement.unwrap().threshold, 0.5);
    let fail = feature_coverage(&full[..10], 0, Some(IMAGE), &thresholds);
    assert_eq!(fail.status, CheckStatus::Fail);
    assert_eq!(fail.measurement.unwrap().threshold, 0.3);
    // A point on the far edge falls in the last cell; one outside is ignored.
    let edge = vec![
        residual(0, 0, [1280.0, 960.0], 0.1),
        residual(0, 0, [-5.0, 10.0], 0.1),
        residual(0, 0, [f64::NAN, 10.0], 0.1),
    ];
    let c = feature_coverage(&edge, 0, Some(IMAGE), &thresholds);
    assert!((c.measurement.unwrap().value - 1.0 / 48.0).abs() < 1e-12);
}

#[test]
fn view_outliers_name_views_far_above_the_noise() {
    let uncertainty = Uncertainty::Available(covariance(&["fx"], vec![vec![1.0]], 0.1));
    let mut residuals = Vec::new();
    for view in 0..5 {
        let error = if view == 3 { 0.6 } else { 0.12 };
        for i in 0..10 {
            residuals.push(residual(view, 0, [i as f64, 0.0], error));
        }
    }
    let thresholds = QualityThresholds::default();
    let c = view_outliers(&residuals, &uncertainty, 0, &thresholds);
    assert_eq!(c.status, CheckStatus::Warn);
    assert_eq!(c.subjects, vec!["view 3"]);
    let m = c.measurement.unwrap();
    assert!((m.value - 6.0).abs() < 1e-9);
    assert_eq!(m.unit, Unit::Sigma);

    residuals.retain(|r| r.pose != 3);
    assert_eq!(
        view_outliers(&residuals, &uncertainty, 0, &thresholds).status,
        CheckStatus::Pass
    );
    assert_eq!(
        view_outliers(
            &residuals,
            &unavailable(UncertaintyUnavailable::NotComputed),
            0,
            &thresholds
        )
        .status,
        CheckStatus::NotApplicable
    );
}

#[test]
fn view_outliers_prefer_the_cameras_own_noise() {
    let mut cov = covariance(&["cam0.fx"], vec![vec![1.0]], 0.1);
    cov.noise.push(ResidualNoise {
        kind: ResidualKind::Reprojection,
        level: NoiseLevel::Sigma(1.0),
        count: 100,
        camera: Some(1),
    });
    assert_eq!(reprojection_sigma(&cov, 1), Some(NoiseLevel::Sigma(1.0)));
    assert_eq!(reprojection_sigma(&cov, 0), Some(NoiseLevel::Sigma(0.1)));
    cov.noise.clear();
    assert_eq!(reprojection_sigma(&cov, 0), None);
}

#[test]
fn correlations_above_the_limit_are_named_per_camera() {
    let params = calibrate(&well_covered_poses()).params.camera;
    // cam1's fx and cx correlate at 0.995; its plane and cam0 do not count.
    let labels = ["cam0.fx", "cam1.fx", "cam1.cx", "cam1.plane.d"];
    let m = vec![
        vec![1.0, 0.0, 0.0, 0.999],
        vec![0.0, 4.0, 0.995 * 2.0, 0.999 * 2.0],
        vec![0.0, 0.995 * 2.0, 1.0, 0.0],
        vec![0.999, 0.999 * 2.0, 0.0, 1.0],
    ];
    let uncertainty = Uncertainty::Available(covariance(&labels, m, 0.1));
    let thresholds = QualityThresholds::default();
    let c = parameter_correlation(&params, &uncertainty, 1, &thresholds);
    assert_eq!(c.status, CheckStatus::Warn);
    assert_eq!(c.subjects, vec!["cam1.fx/cam1.cx"]);
    assert!((c.measurement.unwrap().value - 0.995).abs() < 1e-12);
    // Camera 0 has a single estimated parameter.
    assert_eq!(
        parameter_correlation(&params, &uncertainty, 0, &thresholds).status,
        CheckStatus::NotApplicable
    );
}

#[test]
fn the_focal_lengths_correlating_is_not_a_warning() {
    let params = calibrate(&well_covered_poses()).params.camera;
    let m = vec![
        vec![1.0, 0.9999, 0.1],
        vec![0.9999, 1.0, 0.1],
        vec![0.1, 0.1, 1.0],
    ];
    let uncertainty = Uncertainty::Available(covariance(&["fx", "fy", "cx"], m, 0.1));
    let c = parameter_correlation(&params, &uncertainty, 0, &QualityThresholds::default());
    assert_eq!(c.status, CheckStatus::Pass, "{}", c.message);
    assert!((c.measurement.unwrap().value - 0.1).abs() < 1e-12);
}

#[test]
fn a_camera_held_fixed_has_no_projection_check() {
    let mut export = calibrate(&well_covered_poses());
    let Uncertainty::Available(cov) = &mut export.uncertainty else {
        panic!("expected a covariance");
    };
    // Keep only the pose blocks, as a rig solve with fixed intrinsics does.
    cov.labels = vec!["camera_se3_target.rx".into()];
    cov.covariance = vec![vec![1e-6]];
    let report = assess(&export, &[IMAGE], &QualityThresholds::default()).unwrap();
    let c = check(&report, CheckKind::ProjectionSigma);
    assert_eq!(c.status, CheckStatus::NotApplicable);
    assert!(
        c.message.contains("held camera 0's parameters fixed"),
        "{}",
        c.message
    );
}

#[test]
fn a_report_with_nothing_assessed_warns() {
    let mut export = calibrate(&well_covered_poses());
    export.uncertainty = Uncertainty::default();
    let report = assess(&export, &[], &QualityThresholds::default()).unwrap();
    assert!(
        report
            .checks
            .iter()
            .all(|c| c.status == CheckStatus::NotApplicable)
    );
    assert_eq!(report.verdict, Verdict::Warn);
}

#[test]
fn quantiles_match_tables() {
    assert!((normal_quantile(0.975) - 1.959_963_985).abs() < 1e-8);
    assert!((normal_quantile(0.005) + 2.575_829_304).abs() < 1e-8);
    assert!((normal_quantile(0.995) - 2.575_829_304).abs() < 1e-8);
    // χ² table values: (p, k, quantile).
    for (p, k, q) in [
        (0.995, 10.0, 25.188),
        (0.005, 30.0, 13.787),
        (0.99, 100.0, 135.807),
        (0.005, 1000.0, 888.564),
    ] {
        let rel = (chi_square_quantile(p, k) - q).abs() / q;
        assert!(
            rel < 0.01,
            "χ²({p}, {k}): {} vs {q}",
            chi_square_quantile(p, k)
        );
    }
}
