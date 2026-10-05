//! One small scene per problem type through the real pipeline, on both solver
//! backends: the record is `Ok`, the objective is finite, ground-truth errors
//! are small, a second run reproduces the record bit for bit (timing aside),
//! and the two backends end at the same final cost.

use vision_calibration::optim::{RobustLoss, SolverBackend};
use vision_calibration_bench::solver::metrics::GtErrors;
use vision_calibration_bench::solver::record::{RunStatus, SolverRunRecord};
use vision_calibration_bench::solver::run_scene;
use vision_calibration_bench::solver::scenes::{
    OUTLIER_FRACTION, OutlierSpec, Problem, Scale, SceneSpec, SensorKind,
};

fn small(problem: Problem, sensor: SensorKind) -> SceneSpec {
    SceneSpec {
        problem,
        sensor,
        scale: Scale::Small,
        noise_px: 0.1,
        outliers: OutlierSpec::None,
        loss: RobustLoss::None,
    }
}

/// Run twice; assert the quality bar and bit-exact reproducibility.
fn check(spec: SceneSpec, bounds: impl Fn(&GtErrors)) {
    let tiny = check_on(&spec, &bounds, SolverBackend::TinySolver);
    let factrs = check_on(&spec, &bounds, SolverBackend::Factrs);
    // Both backends reach the same minimum of the same objective.
    let (a, b) = (
        tiny.solve_report.as_ref().map(|r| r.final_cost),
        factrs.solve_report.as_ref().map(|r| r.final_cost),
    );
    if let (Some(a), Some(b)) = (a, b) {
        assert!(
            (a - b).abs() <= 1e-3 * a.abs(),
            "{}: final cost tiny-solver {a} vs factrs {b}",
            spec.id()
        );
    }
}

/// One backend: the quality bar, then a second run that must reproduce the
/// record bit for bit (timing aside).
fn check_on(
    spec: &SceneSpec,
    bounds: &impl Fn(&GtErrors),
    backend: SolverBackend,
) -> SolverRunRecord {
    let ctx = format!("{} on {backend:?}", spec.id());
    let first = run_scene(spec, 1, backend);
    assert_eq!(first.status, RunStatus::Ok, "{ctx}: {:?}", first.status);
    let m = first.metrics.as_ref().expect("metrics");
    assert!(m.objective.is_finite() && m.objective > 0.0, "{ctx}");
    assert!(
        m.inlier_rms_px.is_finite() && m.inlier_rms_px < 0.3,
        "{ctx}"
    );
    assert_eq!(m.num_outliers, 0);
    if let Some(f) = m.gt.focal_rel {
        assert!(f < 0.015, "{ctx}: focal error {f}");
    }
    bounds(&m.gt);
    let t = first.timing.as_ref().expect("timing");
    assert!(t.optimize_ms > 0.0 && t.samples.len() == 1);

    let second = run_scene(spec, 1, backend);
    let a = serde_json::to_value(first.clone().without_timing()).unwrap();
    let b = serde_json::to_value(second.without_timing()).unwrap();
    assert_eq!(a, b, "{ctx}: a second run differs");
    first
}

#[test]
fn planar_intrinsics() {
    check(
        small(Problem::PlanarIntrinsics, SensorKind::Pinhole),
        |_| {},
    );
}

#[test]
fn scheimpflug_intrinsics() {
    check(
        small(Problem::ScheimpflugIntrinsics, SensorKind::Scheimpflug),
        |gt| assert!(gt.sensor_tilt_deg.unwrap() < 1.0),
    );
}

#[test]
fn single_cam_handeye() {
    check(
        small(Problem::SingleCamHandeye, SensorKind::Pinhole),
        |gt| {
            assert!(gt.handeye_rot_deg.unwrap() < 1.0);
            assert!(gt.handeye_trans_mm.unwrap() < 20.0);
        },
    );
}

#[test]
fn laserline_device() {
    check(small(Problem::LaserlineDevice, SensorKind::Pinhole), |gt| {
        assert!(gt.laser_normal_deg.unwrap() < 1.0);
        assert!(gt.laser_distance_mm.unwrap() < 10.0);
    });
}

#[test]
fn rig_extrinsics_pinhole() {
    check(small(Problem::RigExtrinsics, SensorKind::Pinhole), |gt| {
        assert!(gt.rig_rot_deg.unwrap() < 1.0);
        assert!(gt.rig_trans_mm.unwrap() < 30.0);
    });
}

#[test]
fn rig_extrinsics_scheimpflug() {
    check(
        small(Problem::RigExtrinsics, SensorKind::Scheimpflug),
        |gt| {
            assert!(gt.sensor_tilt_deg.unwrap() < 3.0);
            assert!(gt.rig_rot_deg.unwrap() < 3.0);
        },
    );
}

#[test]
fn rig_handeye() {
    check(small(Problem::RigHandeye, SensorKind::Pinhole), |gt| {
        assert!(gt.handeye_rot_deg.unwrap() < 1.0);
        assert!(gt.rig_rot_deg.unwrap() < 1.0);
    });
}

#[test]
fn rig_laserline_device() {
    check(
        small(Problem::RigLaserlineDevice, SensorKind::Pinhole),
        |gt| {
            assert!(gt.laser_normal_deg.unwrap() < 1.0);
            assert!(gt.focal_rel.is_none(), "upstream geometry is frozen");
        },
    );
}

#[test]
fn rig_handeye_laserline() {
    check(
        small(Problem::RigHandeyeLaserline, SensorKind::Pinhole),
        |gt| {
            assert!(gt.handeye_rot_deg.unwrap() < 1.0);
            assert!(gt.laser_normal_deg.unwrap() < 2.0);
        },
    );
}

#[test]
fn contaminated_scene_is_scored_against_its_outliers() {
    let spec = SceneSpec {
        outliers: OutlierSpec::Injected {
            fraction: OUTLIER_FRACTION,
        },
        loss: RobustLoss::Cauchy { scale: 1.0 },
        ..small(Problem::PlanarIntrinsics, SensorKind::Pinhole)
    };
    let rec = run_scene(&spec, 1, SolverBackend::TinySolver);
    assert_eq!(rec.status, RunStatus::Ok);
    let m = rec.metrics.unwrap();
    assert_eq!(m.num_outliers, 26); // ceil(0.05 · 8 views · 63 corners)
    // The robust fit tracks the inliers; the outliers stand out in the full RMS.
    assert!(m.inlier_rms_px < 0.5, "inlier RMS {}", m.inlier_rms_px);
    assert!(m.all_rms_px > 2.0 * m.inlier_rms_px);
    assert!(m.gt.focal_rel.unwrap() < 0.03);
}

#[test]
fn unsupported_combination_is_recorded_as_error() {
    let rec = run_scene(
        &small(Problem::ScheimpflugIntrinsics, SensorKind::Pinhole),
        1,
        SolverBackend::TinySolver,
    );
    assert!(matches!(rec.status, RunStatus::Error(_)));
    assert!(rec.metrics.is_none() && rec.timing.is_none());
}
