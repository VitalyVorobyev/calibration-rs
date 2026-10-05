//! One small scene per problem type through the real pipeline: the record is
//! `Ok`, the objective is finite, ground-truth errors are small, and a second
//! run reproduces the record (timing aside).
//!
//! Reproducibility is asserted to a 1e-8 relative tolerance, not bit-for-bit:
//! the backend keys its parameter blocks in a randomly seeded `HashMap`, so the
//! sparse-solver variable order (and with it the rounding of the last few
//! digits) differs between two solves in the same process.

use vision_calibration::optim::RobustLoss;
use vision_calibration_bench::solver::metrics::GtErrors;
use vision_calibration_bench::solver::record::RunStatus;
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
    let first = run_scene(&spec, 1);
    assert_eq!(
        first.status,
        RunStatus::Ok,
        "{}: {:?}",
        spec.id(),
        first.status
    );
    let m = first.metrics.as_ref().expect("metrics");
    assert!(
        m.objective.is_finite() && m.objective > 0.0,
        "{}",
        spec.id()
    );
    assert!(
        m.inlier_rms_px.is_finite() && m.inlier_rms_px < 0.3,
        "{}",
        spec.id()
    );
    assert_eq!(m.num_outliers, 0);
    if let Some(f) = m.gt.focal_rel {
        assert!(f < 0.015, "{}: focal error {f}", spec.id());
    }
    bounds(&m.gt);
    let t = first.timing.as_ref().expect("timing");
    assert!(t.optimize_ms > 0.0 && t.samples.len() == 1);

    let second = run_scene(&spec, 1);
    let a = serde_json::to_value(first.without_timing()).unwrap();
    let b = serde_json::to_value(second.without_timing()).unwrap();
    assert_close(&a, &b, &spec.id());
}

/// Structural equality; numbers match to 1e-8 relative (1e-12 absolute).
fn assert_close(a: &serde_json::Value, b: &serde_json::Value, ctx: &str) {
    use serde_json::Value::{Array, Number, Object};
    match (a, b) {
        (Number(x), Number(y)) if x.is_f64() || y.is_f64() => {
            let (x, y) = (x.as_f64().unwrap(), y.as_f64().unwrap());
            assert!(
                (x - y).abs() <= 1e-12 + 1e-8 * x.abs().max(y.abs()),
                "{ctx}: {x} vs {y}"
            );
        }
        (Array(x), Array(y)) => {
            assert_eq!(x.len(), y.len(), "{ctx}");
            x.iter().zip(y).for_each(|(x, y)| assert_close(x, y, ctx));
        }
        (Object(x), Object(y)) => {
            assert_eq!(
                x.keys().collect::<Vec<_>>(),
                y.keys().collect::<Vec<_>>(),
                "{ctx}"
            );
            x.iter()
                .for_each(|(k, v)| assert_close(v, &y[k], &format!("{ctx}.{k}")));
        }
        _ => assert_eq!(a, b, "{ctx}"),
    }
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
    let rec = run_scene(&spec, 1);
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
    );
    assert!(matches!(rec.status, RunStatus::Error(_)));
    assert!(rec.metrics.is_none() && rec.timing.is_none());
}
