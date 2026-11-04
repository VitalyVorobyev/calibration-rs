//! The second-order error a calibration's uncertainty predicts, against the
//! joint error measured over noise realizations.
//!
//! A calibration's covariance is its first-order error. Weak views (a stereo
//! rig from centred boards tilted by at most 0.2 rad) leave combinations
//! that the views determine only through large errors that cancel, where
//! the second-order term exceeds the first; the result's
//! [`SecondOrderError`] predicts that from the residuals' curvature alone.
//! The Monte-Carlo gates here (`--ignored`, run in release) calibrate each
//! scene over 300 noise realizations, compare the predicted NEES per
//! parameter with the measured one and check that the predictions fall on
//! the quality threshold's side the scene belongs to; the always-on test
//! runs one weak and one spread calibration through the quality report.

#![allow(missing_docs)]

mod common;

use common::{gauss, stereo_board_poses_scaled, stereo_cam1_se3_rig, stereo_cameras};
use nalgebra::{DMatrix, DVector, SymmetricEigen};
use std::time::Instant;
use vision_calibration::analysis::{CandidateGrid, candidate_poses};
use vision_calibration::common::CalibratedCameras;
use vision_calibration::core::{
    BrownConrady5, CameraParams, CorrespondenceView, FxFyCxCy, Iso3, NoMeta, ParameterCovariance,
    PinholeCamera, PlanarDataset, Pt2, Pt3, RigDataset, RigView, RigViewObs, SecondOrderError,
    View, make_pinhole_camera, pinhole_camera_params,
};
use vision_calibration::optim::camera_parameters;
use vision_calibration::planar_intrinsics::PlanarIntrinsicsProblem;
use vision_calibration::rig_extrinsics::{
    RigExtrinsicsConfig, RigExtrinsicsInput, RigExtrinsicsProblem,
};
use vision_calibration::session::CalibrationSession;
use vision_calibration::synthetic::noise::UniformPixelNoise;
use vision_calibration::synthetic::planar;

// ---------------------------------------------------------------------------
// Scenes
// ---------------------------------------------------------------------------

/// The stereo dataset of trial `seed` with Gaussian pixel noise of σ
/// `sigma`.
fn stereo_input(poses: &[Iso3], sigma: f64, seed: u64) -> RigExtrinsicsInput {
    let cameras = stereo_cameras();
    let cam_se3_rig = [Iso3::identity(), stereo_cam1_se3_rig()];
    let board = planar::grid_points(8, 6, 0.03);
    let views: Vec<RigView<NoMeta>> = poses
        .iter()
        .enumerate()
        .map(|(v, rig_se3_target)| RigView {
            meta: NoMeta,
            obs: RigViewObs {
                cameras: (0..2)
                    .map(|c| {
                        let pixels = board
                            .iter()
                            .enumerate()
                            .map(|(i, p)| {
                                let key = (seed << 32) + (c * 1_000_000 + v * 1000 + i) as u64;
                                let q = cameras[c]
                                    .project_point(&(cam_se3_rig[c] * rig_se3_target * p))
                                    .unwrap();
                                Pt2::new(
                                    q.x + sigma * gauss(2 * key),
                                    q.y + sigma * gauss(2 * key + 1),
                                )
                            })
                            .collect();
                        Some(CorrespondenceView::new(board.clone(), pixels).unwrap())
                    })
                    .collect(),
            },
        })
        .collect();
    RigDataset::new(views, 2).unwrap()
}

/// The stereo rig's covariance and its whitened error against the truth.
fn stereo_trial(poses: &[Iso3], sigma: f64, seed: u64) -> Trial {
    let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
    session.set_config(RigExtrinsicsConfig::default()).unwrap();
    session.set_input(stereo_input(poses, sigma, seed)).unwrap();
    vision_calibration::rig_extrinsics::run_calibration(&mut session).unwrap();
    let rig = session.export().unwrap().calibrated_rig().unwrap();
    let cov = rig
        .uncertainty()
        .covariance()
        .expect("rig covariance")
        .clone();
    let truth: Vec<CameraParams> = stereo_cameras().iter().map(pinhole_camera_params).collect();
    let truth_cam_se3_rig = [Iso3::identity(), stereo_cam1_se3_rig()];
    let error = DVector::from_iterator(
        cov.labels.len(),
        cov.labels.iter().map(|label| {
            let (camera, name) = label
                .strip_prefix("cam")
                .and_then(|rest| rest.split_once('.'))
                .map(|(k, name)| (k.parse::<usize>().unwrap(), name))
                .unwrap_or_else(|| panic!("unexpected label {label}"));
            match name.strip_prefix("cam_se3_rig.") {
                Some(axis) => {
                    let at = ["rx", "ry", "rz", "tx", "ty", "tz"]
                        .iter()
                        .position(|a| *a == axis)
                        .unwrap();
                    tangent_error(&truth_cam_se3_rig[camera], &rig.cam_se3_rig()[camera])[at]
                }
                None => parameter(&rig.cameras()[camera], name) - parameter(&truth[camera], name),
            }
        }),
    );
    Trial::new(cov, error)
}

/// A camera of the `capture_planning` example: 1280 × 960, fx = fy =
/// 1000 px, barrel distortion.
fn planar_camera() -> PinholeCamera {
    make_pinhole_camera(
        FxFyCxCy {
            fx: 1000.0,
            fy: 1000.0,
            cx: 640.0,
            cy: 480.0,
        },
        BrownConrady5 {
            k1: -0.15,
            k2: 0.05,
            ..BrownConrady5::default()
        },
    )
}

const PLANAR_IMAGE: [u32; 2] = [1280, 960];

/// `n` views of the 12 × 9 board 0.65 m away, tilted by `tilt` about ±x
/// and ±y in turn, centred on the image quadrants (`quadrants`) or on the
/// image centre at 0.60–0.75 m.
fn planar_poses(board: &[Pt3], n: usize, tilt: f64, quadrants: bool) -> Vec<Iso3> {
    const TILTS: [usize; 4] = [1, 3, 2, 4];
    let camera = pinhole_camera_params(&planar_camera());
    if quadrants {
        let grid = CandidateGrid::new(0.65, tilt, [2, 2]);
        let near = candidate_poses(&camera, board, PLANAR_IMAGE, &grid).unwrap();
        (0..n)
            .map(|k| near[(k % 4) * 5 + TILTS[(k + k / 4) % 4]])
            .collect()
    } else {
        let mut out = Vec::new();
        for d in [0.6, 0.65, 0.7, 0.75] {
            let grid = CandidateGrid::new(d, tilt, [1, 1]);
            let poses = candidate_poses(&camera, board, PLANAR_IMAGE, &grid).unwrap();
            out.extend(TILTS.iter().map(|&t| poses[t]));
        }
        out.truncate(n);
        out
    }
}

/// One planar calibration of trial `seed`, uniform noise of ±`max_abs_px`.
fn planar_trial(board: &[Pt3], plan: &[Iso3], max_abs_px: f64, seed: u64) -> Trial {
    let truth = planar_camera();
    let noise = UniformPixelNoise { seed, max_abs_px };
    let views = planar::project_views_noisy(&truth, board, plan, &noise).unwrap();
    let dataset = PlanarDataset::new(views.into_iter().map(View::without_meta).collect()).unwrap();
    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    session.set_input(dataset).unwrap();
    vision_calibration::planar_intrinsics::run_calibration(&mut session).unwrap();
    let export = session.export().unwrap();
    let cov = export
        .parameter_uncertainty()
        .covariance()
        .expect("planar covariance")
        .clone();
    let estimate = &export.camera_params()[0];
    let truth = pinhole_camera_params(&truth);
    let error = DVector::from_iterator(
        cov.labels.len(),
        cov.labels
            .iter()
            .map(|label| parameter(estimate, label) - parameter(&truth, label)),
    );
    Trial::new(cov, error)
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// One calibration: its whitened error and its second-order prediction.
struct Trial {
    whitened: DVector<f64>,
    second: SecondOrderError,
}

impl Trial {
    fn new(cov: ParameterCovariance, error: DVector<f64>) -> Self {
        let n = cov.labels.len();
        let sigma = DMatrix::from_fn(n, n, |a, b| cov.covariance[a][b]);
        let l = sigma.cholesky().expect("positive definite covariance");
        Self {
            whitened: l.l().solve_lower_triangular(&error).unwrap(),
            second: cov.second_order.expect("a second-order prediction"),
        }
    }
}

/// Measured against predicted over a scene's trials.
#[derive(Debug)]
struct Summary {
    /// Mean NEES per parameter, measured.
    nees: f64,
    /// Its standard error.
    nees_se: f64,
    /// The largest eigenvalue of the whitened errors' second moment, as a
    /// scatter ratio.
    max_scatter: f64,
    /// Mean predicted NEES per parameter.
    predicted_nees: f64,
    /// Median predicted worst scatter ratio, and its 5 % and 95 % points
    /// over the trials.
    predicted_max: [f64; 3],
    /// Median predicted NEES per parameter.
    predicted_nees_median: f64,
    /// Median measured NEES per parameter.
    nees_median: f64,
    /// The first trial's combination.
    combination: String,
}

fn summarize(trials: &[Trial]) -> Summary {
    let n = trials.len() as f64;
    let k = trials[0].whitened.len();
    let nees: Vec<f64> = trials
        .iter()
        .map(|t| t.whitened.norm_squared() / k as f64)
        .collect();
    let mean = nees.iter().sum::<f64>() / n;
    let var = nees.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let mut moment = DMatrix::<f64>::zeros(k, k);
    for t in trials {
        moment += &t.whitened * t.whitened.transpose();
    }
    moment /= n;
    let top = SymmetricEigen::new(moment)
        .eigenvalues
        .iter()
        .copied()
        .fold(0.0, f64::max);
    let quantile = |mut v: Vec<f64>, q: f64| {
        v.sort_by(f64::total_cmp);
        v[((q * (v.len() - 1) as f64).round()) as usize]
    };
    let max: Vec<f64> = trials.iter().map(|t| t.second.max_scatter_ratio).collect();
    let at = |q: f64| quantile(max.clone(), q);
    Summary {
        nees: mean,
        nees_se: (var / n).sqrt(),
        max_scatter: top.sqrt(),
        predicted_nees: trials.iter().map(|t| t.second.nees_ratio).sum::<f64>() / n,
        predicted_max: [at(0.5), at(0.05), at(0.95)],
        predicted_nees_median: quantile(trials.iter().map(|t| t.second.nees_ratio).collect(), 0.5),
        nees_median: quantile(nees.clone(), 0.5),
        combination: trials[0]
            .second
            .combination
            .iter()
            .map(|t| format!("{:+.2} {}", t.weight, t.label))
            .collect::<Vec<_>>()
            .join(", "),
    }
}

/// What a scene's prediction must show.
#[derive(Clone, Copy, PartialEq)]
enum Expect {
    /// The covariance describes the error: every calibration's prediction
    /// (to its 95th percentile) stays under the quality threshold.
    Linear,
    /// The error departs from the covariance: every calibration's
    /// prediction (from its 5th percentile) is above the threshold.
    Beyond,
}

/// The quality check's default limit on the worst scatter ratio.
const THRESHOLD: f64 = 1.2;

/// Prints the scene's line and checks it: the predicted mean NEES within
/// three standard errors of the measured one (when `nees` is set; a scene
/// whose errors are heavy-tailed has no usable mean), and the predictions
/// on the side of the threshold that `expect` says.
fn gate(name: &str, trials: &[Trial], seconds: f64, expect: Expect, nees: bool) {
    let s = report(name, trials, seconds);
    if nees {
        assert!(
            (s.predicted_nees - s.nees).abs() <= 3.0 * s.nees_se,
            "{name}: predicted NEES/k {:.3}, measured {:.3} ± {:.3}",
            s.predicted_nees,
            s.nees,
            s.nees_se
        );
    }
    match expect {
        Expect::Linear => assert!(
            s.predicted_max[2] < THRESHOLD,
            "{name}: predicted worst ratio up to {:.2}",
            s.predicted_max[2]
        ),
        Expect::Beyond => assert!(
            s.predicted_max[1] > THRESHOLD,
            "{name}: predicted worst ratio down to {:.2}",
            s.predicted_max[1]
        ),
    }
}

fn report(name: &str, trials: &[Trial], seconds: f64) -> Summary {
    let s = summarize(trials);
    eprintln!(
        "{name:<34} NEES/k {:.3} ± {:.3} (predicted {:.3}; medians {:.3}, {:.3}); worst \
         scatter {:.2} (predicted median {:.2}, 5–95 % {:.2}–{:.2}); {} trials, {seconds:.1} s; \
         [{}]",
        s.nees,
        s.nees_se,
        s.predicted_nees,
        s.nees_median,
        s.predicted_nees_median,
        s.max_scatter,
        s.predicted_max[0],
        s.predicted_max[1],
        s.predicted_max[2],
        trials.len(),
        s.combination
    );
    s
}

/// `estimate = truth · Exp(δ)`: δ as `[rx ry rz tx ty tz]`.
fn tangent_error(truth: &Iso3, estimate: &Iso3) -> [f64; 6] {
    let w = (truth.rotation.inverse() * estimate.rotation).scaled_axis();
    let t = truth.rotation.inverse() * (estimate.translation.vector - truth.translation.vector);
    [w.x, w.y, w.z, t.x, t.y, t.z]
}

fn parameter(camera: &CameraParams, name: &str) -> f64 {
    camera_parameters(camera)
        .unwrap()
        .into_iter()
        .find_map(|(label, value)| (label == name).then_some(value))
        .unwrap_or_else(|| panic!("camera has no parameter {name}"))
}

// ---------------------------------------------------------------------------
// Studies
// ---------------------------------------------------------------------------

/// Calibrations per scene: the mean NEES per parameter of 22 has a standard
/// error of about `√(2 / (22 · 300))`, 1.7 %, before heavy tails.
const STUDY_TRIALS: u64 = 300;

/// The stereo rig over capture strength and noise: the rig's eight views
/// with the tilts scaled and the board moved off the centre.
#[test]
#[ignore = "Monte-Carlo gate, minutes in release: cargo test --release -p vision-calibration --test second_order_error -- --ignored"]
fn stereo_second_order_matches_the_monte_carlo_error() {
    use Expect::{Beyond, Linear};
    // (name, tilt scale, offset scale, pixel σ, expectation)
    let scenes: [(&str, f64, f64, f64, Expect); 7] = [
        ("centred, tilt ×1, 0.3 px", 1.0, 0.0, 0.3, Beyond),
        ("centred, tilt ×1, 0.1 px", 1.0, 0.0, 0.1, Beyond),
        ("centred, tilt ×1, 0.03 px", 1.0, 0.0, 0.03, Linear),
        ("centred, tilt ×1.5, 0.3 px", 1.5, 0.0, 0.3, Beyond),
        ("centred, tilt ×2.5, 0.3 px", 2.5, 0.0, 0.3, Beyond),
        ("off centre, tilt ×1, 0.3 px", 1.0, 1.0, 0.3, Beyond),
        ("off centre, tilt ×2.5, 0.3 px", 2.5, 1.0, 0.3, Linear),
    ];
    for (name, tilt, offset, sigma, expect) in scenes {
        let clock = Instant::now();
        let poses = stereo_board_poses_scaled(tilt, offset);
        let trials: Vec<Trial> = (0..STUDY_TRIALS)
            .map(|t| stereo_trial(&poses, sigma, 500 + t))
            .collect();
        gate(name, &trials, clock.elapsed().as_secs_f64(), expect, true);
    }
}

/// A single camera over tilt and placement (the `capture_planning` scene,
/// ±0.35 px uniform noise). At 0.05 rad a few calibrations land far off, so
/// the mean NEES is not compared there.
#[test]
#[ignore = "Monte-Carlo gate, minutes in release: cargo test --release -p vision-calibration --test second_order_error -- --ignored"]
fn planar_second_order_matches_the_monte_carlo_error() {
    use Expect::{Beyond, Linear};
    let board = planar::grid_points(12, 9, 0.03);
    // (name, tilt, views in the quadrants, expectation, compare the NEES)
    let scenes: [(&str, f64, bool, Expect, bool); 6] = [
        ("quadrants, tilt 0.05 rad", 0.05, true, Beyond, false),
        ("quadrants, tilt 0.1 rad", 0.1, true, Linear, true),
        ("quadrants, tilt 0.2 rad", 0.2, true, Linear, true),
        ("quadrants, tilt 0.35 rad", 0.35, true, Linear, true),
        ("centre, tilt 0.35 rad", 0.35, false, Linear, true),
        ("centre, tilt 0.1 rad", 0.1, false, Linear, true),
    ];
    for (name, tilt, quadrants, expect, nees) in scenes {
        let clock = Instant::now();
        let plan = planar_poses(&board, 16, tilt, quadrants);
        let trials: Vec<Trial> = (0..STUDY_TRIALS)
            .map(|t| planar_trial(&board, &plan, 0.35, 500 + t))
            .collect();
        gate(name, &trials, clock.elapsed().as_secs_f64(), expect, nees);
    }
}

// ---------------------------------------------------------------------------
// The check
// ---------------------------------------------------------------------------

/// One weak and one spread stereo calibration through the quality report:
/// the weak one's covariance-linearity check warns and names camera 1's
/// rotation with the principal points, the spread one's passes.
#[test]
fn weak_stereo_views_warn_and_spread_views_pass() {
    use vision_calibration::analysis::{CheckKind, CheckStatus, QualityThresholds, assess};
    let calibrate = |poses: &[Iso3]| {
        let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
        session.set_config(RigExtrinsicsConfig::default()).unwrap();
        session.set_input(stereo_input(poses, 0.3, 7)).unwrap();
        vision_calibration::rig_extrinsics::run_calibration(&mut session).unwrap();
        session.export().unwrap()
    };
    let linearity = |poses: &[Iso3]| {
        let export = calibrate(poses);
        let report = assess(&export, &[[1280, 960]; 2], &QualityThresholds::default()).unwrap();
        let check = report
            .checks
            .into_iter()
            .find(|c| c.kind == CheckKind::CovarianceLinearity)
            .expect("a covariance-linearity check");
        eprintln!("{:?}: {}", check.status, check.message);
        check
    };

    let weak = linearity(&common::stereo_board_poses());
    assert_eq!(weak.status, CheckStatus::Warn, "{}", weak.message);
    let ratio = weak.measurement.unwrap().value;
    assert!((2.5..=4.0).contains(&ratio), "weak views: {ratio:.2}");
    assert!(
        weak.subjects
            .iter()
            .any(|s| s.starts_with("cam1.cam_se3_rig"))
            && weak.subjects.iter().any(|s| s.ends_with(".cx")),
        "{:?}",
        weak.subjects
    );

    let spread = linearity(&common::stereo_board_poses_spread());
    assert_eq!(spread.status, CheckStatus::Pass, "{}", spread.message);
    assert!(spread.subjects.is_empty());
    let ratio = spread.measurement.unwrap().value;
    assert!(ratio < 1.1, "spread views: {ratio:.2}");
}
