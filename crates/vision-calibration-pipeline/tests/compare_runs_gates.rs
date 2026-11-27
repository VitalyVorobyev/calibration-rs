//! Monte-Carlo gates of the tests `compare_runs` reports, on synthetic
//! captures (a 12 × 9 board, 0.3 px of noise unless a sweep says
//! otherwise): under the null hypothesis (both runs estimate one device,
//! the model is right, nothing left out by its residuals) each test's
//! rejection rate at the 5 % level must be 5 % within the Monte-Carlo
//! error, and no test may go missing silently.
//!
//! - **Independent** (`z = Δ / √(σ_ref² + σ_other²)`): two captures of one
//!   camera with their own poses and noise.
//! - **Nested** (the Hausman form on one noise level): a capture against
//!   itself with 30 % of its views, 20 % of its features, 1 of 8 or 1 of 25
//!   views, or 6 features left out; least squares and a Huber loss.
//! - Each cell's **joint** `χ²` over the camera's parameters.
//! - The 6-dof `χ²` of a two-camera rig's relative pose: independent, and
//!   2 of 8 views left out.
//! - The **alternative**: gross outliers in the larger run, which the
//!   smaller leaves out, must be detected.
//!
//! The gates run on one backend: the two reach the same optimum (the
//! regression suite checks them against each other), so the same seeds on
//! the other would repeat the draw. A sweep over the number of views, the
//! noise and the camera model (pinhole Brown-Conrady, a Kannala-Brandt
//! fisheye, a Scheimpflug sensor) runs at half the trials. The trials of a
//! cell run on their own threads. Release-only:
//! `python3 scripts/uncertainty_gates.py compare_runs_gates`.

#![allow(missing_docs)]

use std::collections::BTreeMap;

use nalgebra::{Rotation3, Translation3, Vector2, Vector3};
use vision_calibration_core::{
    BrownConrady5, CameraModel, CameraModelSpec, CameraParams, CameraProject, CorrespondenceView,
    DistortionKind, DistortionParams, FxFyCxCy, Iso3, NoMeta, PlanarDataset, ProjectionKind,
    ProjectionParams, Pt2, Pt3, RigDataset, RigView, RigViewObs, ScheimpflugParams, SensorKind,
    SensorParams, View,
};
use vision_calibration_optim::{RobustLoss, SolverBackend};
use vision_calibration_pipeline::analysis::{
    CompareRunsOptions, ComparedRun, DataRelation, DifferenceTest, JointTest, ParameterComparison,
    RunDifference, RunSide, compare_runs,
};
use vision_calibration_pipeline::common::IntrinsicsManualInit;
use vision_calibration_pipeline::planar_intrinsics::{
    PlanarIntrinsicsConfig, PlanarIntrinsicsExport, PlanarIntrinsicsProblem, step_init_with_seed,
    step_optimize,
};
use vision_calibration_pipeline::rig_extrinsics::{
    RigExtrinsicsConfig, RigExtrinsicsExport, RigExtrinsicsProblem, run_calibration,
};
use vision_calibration_pipeline::session::CalibrationSession;

/// The two-sided 5 % point of a standard normal.
const Z_95: f64 = 1.959_963_984_540_054;
/// The 95 % point of a χ² of 6 degrees of freedom.
const CHI2_6_95: f64 = 12.591_587_243_743_977;
/// The nominal level.
const LEVEL: f64 = 0.05;

/// SplitMix64.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x2545_F491_4F6C_DD1D)
    }

    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }

    fn normal(&mut self) -> f64 {
        let u = 1.0 - self.uniform();
        let v = self.uniform();
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

/// Trials per gate: `COMPARE_GATE_TRIALS`, default 400.
fn trials() -> usize {
    std::env::var("COMPARE_GATE_TRIALS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(400)
}

const SIZE: [u32; 2] = [1280, 960];
const COLS: usize = 12;
const ROWS: usize = 9;
const PITCH: f64 = 0.025;

fn board() -> Vec<Pt3> {
    (0..ROWS)
        .flat_map(|j| (0..COLS).map(move |i| Pt3::new(i as f64 * PITCH, j as f64 * PITCH, 0.0)))
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Model {
    Brown,
    Fisheye,
    Scheimpflug,
}

fn truth(model: Model) -> CameraParams {
    let k = |f: f64| FxFyCxCy {
        fx: f,
        fy: f * 1.002,
        cx: 641.0,
        cy: 478.0,
    };
    match model {
        Model::Brown => CameraParams::new(
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
            SensorParams::Identity,
            k(900.0),
        ),
        Model::Fisheye => CameraParams::new(
            ProjectionParams::Equidistant,
            DistortionParams::from_values(
                DistortionKind::KannalaBrandt4,
                &[-0.02, 0.002, 0.0, 0.0],
            )
            .expect("KB4"),
            SensorParams::Identity,
            k(500.0),
        ),
        Model::Scheimpflug => CameraParams::new(
            ProjectionParams::Pinhole,
            DistortionParams::BrownConrady5 {
                params: BrownConrady5 {
                    k1: -0.08,
                    k2: 0.0,
                    k3: 0.0,
                    p1: 0.0,
                    p2: 0.0,
                    iters: 8,
                },
            },
            SensorParams::Scheimpflug {
                params: ScheimpflugParams {
                    tilt_x: 0.04,
                    tilt_y: -0.02,
                },
            },
            k(1800.0),
        ),
    }
}

fn spec(model: Model) -> CameraModelSpec {
    match model {
        Model::Brown => {
            CameraModelSpec::pinhole(DistortionKind::BrownConrady5, SensorKind::Identity)
        }
        Model::Fisheye => CameraModelSpec::new(
            ProjectionKind::Equidistant,
            DistortionKind::KannalaBrandt4,
            SensorKind::Identity,
        ),
        Model::Scheimpflug => {
            CameraModelSpec::pinhole(DistortionKind::BrownConrady5, SensorKind::Scheimpflug)
        }
    }
}

/// A board pose whose centre lands at a random pixel, at a random distance
/// along that pixel's ray, tilted up to 0.45 rad.
fn random_pose(camera: &CameraModel, rng: &mut Rng, distance: (f64, f64)) -> Iso3 {
    let u = Vector2::new(
        rng.range(0.15, 0.85) * f64::from(SIZE[0]),
        rng.range(0.15, 0.85) * f64::from(SIZE[1]),
    );
    let ray = camera
        .backproject_pixel(&Pt2::new(u.x, u.y))
        .direction
        .normalize();
    let z = rng.range(distance.0, distance.1);
    let r = Rotation3::from_euler_angles(
        rng.range(-0.45, 0.45),
        rng.range(-0.45, 0.45),
        rng.range(-0.6, 0.6),
    );
    let centre = Vector3::new(
        (COLS - 1) as f64 * PITCH / 2.0,
        (ROWS - 1) as f64 * PITCH / 2.0,
        0.0,
    );
    let t = ray * z - r * centre;
    Iso3::from_parts(Translation3::from(t), r.into())
}

/// The board seen from `pose` through `camera`, with Gaussian noise of
/// `noise` px per axis; `None` when fewer than 40 points land in the image.
fn observe(
    camera: &CameraModel,
    pose: &Iso3,
    noise: f64,
    rng: &mut Rng,
) -> Option<CorrespondenceView> {
    let (mut p3, mut p2) = (Vec::new(), Vec::new());
    for pw in board() {
        let pc = pose.transform_point(&pw);
        let Some(q) = camera.project_camera_point(&pc.coords) else {
            continue;
        };
        let inside = q.x >= 0.0
            && q.y >= 0.0
            && q.x <= f64::from(SIZE[0] - 1)
            && q.y <= f64::from(SIZE[1] - 1);
        let (nx, ny) = (rng.normal() * noise, rng.normal() * noise);
        if inside {
            p3.push(pw);
            p2.push(Pt2::new(q.x + nx, q.y + ny));
        }
    }
    (p3.len() >= 40).then(|| CorrespondenceView::new(p3, p2).expect("view"))
}

fn distance(model: Model) -> (f64, f64) {
    match model {
        Model::Brown => (0.45, 0.75),
        Model::Fisheye => (0.25, 0.45),
        Model::Scheimpflug => (0.9, 1.4),
    }
}

/// `views` views of one camera, seeded.
fn capture(model: Model, views: usize, noise: f64, seed: u64) -> PlanarDataset {
    let camera = truth(model).build().expect("truth");
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(views);
    while out.len() < views {
        let pose = random_pose(&camera, &mut rng, distance(model));
        if let Some(view) = observe(&camera, &pose, noise, &mut rng) {
            out.push(View::without_meta(view));
        }
    }
    PlanarDataset::new(out).expect("dataset")
}

/// The first `keep` views.
fn first_views(data: &PlanarDataset, keep: usize) -> PlanarDataset {
    PlanarDataset::new(data.views[..keep].to_vec()).expect("subset")
}

/// Each feature dropped with probability `fraction`, seeded.
fn drop_features(data: &PlanarDataset, fraction: f64, seed: u64) -> PlanarDataset {
    let mut rng = Rng::new(seed ^ 0xFEA7);
    let views = data
        .views
        .iter()
        .map(|v| {
            let (mut p3, mut p2) = (Vec::new(), Vec::new());
            for (a, b) in v.obs.points_3d.iter().zip(&v.obs.points_2d) {
                if rng.uniform() >= fraction {
                    p3.push(*a);
                    p2.push(*b);
                }
            }
            View::without_meta(CorrespondenceView::new(p3, p2).expect("view"))
        })
        .collect();
    PlanarDataset::new(views).expect("subset")
}

fn calibrate(
    data: &PlanarDataset,
    model: Model,
    backend: SolverBackend,
    loss: RobustLoss,
) -> Option<PlanarIntrinsicsExport> {
    let truth = truth(model);
    let mut config = PlanarIntrinsicsConfig::default();
    config.camera.model = spec(model);
    config.solver.backend = backend;
    config.solver.max_iters = 200;
    config.optimize.reprojection_loss = loss;
    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    session.set_config(config).ok()?;
    session.set_input(data.clone()).ok()?;
    // Initialization is not under test: start near the truth.
    let mut seed = IntrinsicsManualInit::default();
    let k = truth.intrinsics;
    seed.intrinsics = Some(FxFyCxCy {
        fx: k.fx * 1.01,
        fy: k.fy * 1.01,
        cx: k.cx + 2.0,
        cy: k.cy - 2.0,
    });
    if let SensorParams::Scheimpflug { params } = truth.sensor {
        seed.sensor = Some(ScheimpflugParams {
            tilt_x: params.tilt_x * 0.8,
            tilt_y: params.tilt_y * 0.8,
        });
    }
    step_init_with_seed(&mut session, seed, None).ok()?;
    step_optimize(&mut session, None).ok()?;
    let export = session.export().ok()?;
    export.uncertainty.covariance()?;
    Some(export)
}

/// The two cameras of the synthetic rig.
fn rig_cameras() -> [CameraParams; 2] {
    let mut b = truth(Model::Brown);
    b.intrinsics.fx = 880.0;
    b.intrinsics.fy = 878.0;
    b.intrinsics.cx = 650.0;
    [truth(Model::Brown), b]
}

fn cam1_se3_cam0() -> Iso3 {
    Iso3::from_parts(
        Translation3::new(-0.12, 0.005, 0.01),
        Rotation3::from_euler_angles(0.01, 0.12, -0.005).into(),
    )
}

/// `views` views of the two-camera rig (camera 0 the rig frame); a view is
/// kept when both cameras see at least 40 points.
fn rig_capture(views: usize, noise: f64, seed: u64) -> RigDataset<NoMeta> {
    let cameras = rig_cameras().map(|c| c.build().expect("camera"));
    let mounts = [Iso3::identity(), cam1_se3_cam0()];
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(views);
    while out.len() < views {
        let mut pose = random_pose(&cameras[0], &mut rng, (0.6, 0.9));
        // Between the two cameras.
        pose.translation.vector.x -= 0.06;
        let obs: Vec<Option<CorrespondenceView>> = cameras
            .iter()
            .zip(&mounts)
            .map(|(c, m)| observe(c, &(m * pose), noise, &mut rng))
            .collect();
        if obs.iter().all(Option::is_some) {
            out.push(RigView::new(RigViewObs::new(obs), NoMeta));
        }
    }
    RigDataset::new(out, 2).expect("rig dataset")
}

fn calibrate_rig(data: &RigDataset<NoMeta>, backend: SolverBackend) -> Option<RigExtrinsicsExport> {
    let mut config = RigExtrinsicsConfig::default();
    config.solver.backend = backend;
    let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
    session.set_config(config).ok()?;
    session.set_input(data.clone()).ok()?;
    run_calibration(&mut session).ok()?;
    let export = session.export().ok()?;
    export.uncertainty.covariance()?;
    Some(export)
}

// ── Statistics ──────────────────────────────────────────────────────────────

/// One comparison's tests: each parameter's `z` (or why it has none) and the
/// joint `χ²` with its degrees of freedom.
struct Outcome {
    z: Vec<(String, Result<f64, String>)>,
    joint: Result<(f64, usize), String>,
}

fn outcome_of(difference: &RunDifference) -> Outcome {
    let ParameterComparison::SameModel { parameters, joint } = &difference.cameras[0].parameters
    else {
        panic!("same model");
    };
    let z = parameters
        .iter()
        .filter(|p| p.reference_sigma.is_some())
        .map(|p| {
            let z = match &p.test {
                DifferenceTest::Independent { z, .. } | DifferenceTest::Nested { z, .. } => Ok(*z),
                DifferenceTest::NotAvailable { reason } => Err(format!("{reason:?}")),
                other => Err(format!("{other:?}")),
            };
            (p.label.clone(), z)
        })
        .collect();
    Outcome {
        z,
        joint: joint_of(joint),
    }
}

fn joint_of(test: &JointTest) -> Result<(f64, usize), String> {
    match test {
        JointTest::Independent { chi2, dof, .. } | JointTest::Nested { chi2, dof, .. } => {
            Ok((*chi2, *dof))
        }
        JointTest::NotAvailable { reason } => Err(format!("{reason:?}")),
        other => Err(format!("{other:?}")),
    }
}

fn compare(
    reference: &PlanarIntrinsicsExport,
    other: &PlanarIntrinsicsExport,
    relation: DataRelation,
) -> RunDifference {
    let ids = vec!["cam".to_string()];
    let runs = [
        ComparedRun::new("reference", reference, ids.clone(), vec![SIZE]),
        ComparedRun::new("other", other, ids, vec![SIZE]).with_relation(relation),
    ];
    let mut comparison = compare_runs(&runs, &CompareRunsOptions::new()).expect("comparison");
    let difference = comparison.differences.remove(0);
    assert_eq!(
        difference.relation_refused, None,
        "the exports contradict the stated relation"
    );
    difference
}

/// The Monte-Carlo standard error of a rejection rate at the nominal level
/// over `n` trials.
fn mc_error(n: usize) -> f64 {
    (LEVEL * (1.0 - LEVEL) / n as f64).sqrt()
}

/// The mean of `values` and its bootstrap standard error (500 resamples,
/// seeded).
fn bootstrap_mean(values: &[f64]) -> (f64, f64) {
    let n = values.len();
    let mean = values.iter().sum::<f64>() / n as f64;
    let mut rng = Rng::new(0xB007);
    let resampled: Vec<f64> = (0..500)
        .map(|_| {
            (0..n)
                .map(|_| values[((rng.uniform() * n as f64) as usize).min(n - 1)])
                .sum::<f64>()
                / n as f64
        })
        .collect();
    let m = resampled.iter().sum::<f64>() / 500.0;
    let se = (resampled.iter().map(|v| (v - m).powi(2)).sum::<f64>() / 499.0).sqrt();
    (mean, se)
}

/// Runs `trial` for `0..n` on every core; `None` results (a failed
/// calibration) are dropped and counted.
fn run_trials<T: Send>(n: usize, trial: impl Fn(u64) -> Option<T> + Sync) -> (Vec<T>, usize) {
    let threads = std::thread::available_parallelism().map_or(4, |p| p.get());
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    let r = trial(i as u64);
                    results.lock().expect("lock").push((i, r));
                }
            });
        }
    });
    let mut results = results.into_inner().expect("lock");
    results.sort_by_key(|(i, _)| *i);
    let failed = results.iter().filter(|(_, r)| r.is_none()).count();
    (results.into_iter().filter_map(|(_, r)| r).collect(), failed)
}

/// Each parameter's statistics over a cell's trials: its `z` values and the
/// reasons of the tests that are missing.
fn by_label(trials: &[Outcome]) -> BTreeMap<&str, (Vec<f64>, BTreeMap<String, usize>)> {
    let mut out: BTreeMap<&str, (Vec<f64>, BTreeMap<String, usize>)> = BTreeMap::new();
    for t in trials {
        for (label, z) in &t.z {
            let entry = out.entry(label).or_default();
            match z {
                Ok(z) => entry.0.push(*z),
                Err(reason) => *entry.1.entry(reason.clone()).or_default() += 1,
            }
        }
    }
    out
}

fn moments(z: &[f64]) -> (f64, f64, f64) {
    let n = z.len() as f64;
    let rate = z.iter().filter(|z| z.abs() > Z_95).count() as f64 / n;
    let mean = z.iter().sum::<f64>() / n;
    let std = (z.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    (rate, mean, std)
}

/// A cell's gate under the null, printing what it measures:
///
/// - each parameter's rejection rate at 5 % within 3.5 Monte-Carlo errors,
///   its z standard deviation within `3.4 / √(2n)` of 1 (the sample
///   standard deviation's own error is `1 / √(2n)`: a family-wise false
///   alarm of about 1 % over eight parameters), and at most 1 % of its
///   tests missing, so none is dropped silently;
/// - the pooled mean `z²` over parameters and trials within 3 bootstrap
///   standard errors (over trials) of 1;
/// - independent captures: the joint `χ²` (see [`gate_joint`]); nested
///   runs: no joint test, with its reason.
fn gate_null(name: &str, trials: &[Outcome], failed: usize, independent: bool) {
    eprintln!(
        "{name}: {} trials, {failed} failed calibrations",
        trials.len()
    );
    assert!(
        failed * 100 <= trials.len(),
        "{name}: {failed} failed calibrations"
    );
    for (label, (z, missing)) in &by_label(trials) {
        let n = z.len();
        let (rate, _, std) = moments(z);
        let tolerance = 3.4 / (2.0 * n as f64).sqrt();
        let missing_count: usize = missing.values().sum();
        eprintln!(
            "  {label:>8}: n {n:5}  reject {rate:.4} ± {:.4}  z std {std:.3} (±{tolerance:.3})  missing {missing:?}",
            mc_error(n)
        );
        assert!(
            (rate - LEVEL).abs() <= 3.5 * mc_error(n),
            "{name} {label}: rejection rate {rate}"
        );
        assert!(
            (std - 1.0).abs() <= tolerance,
            "{name} {label}: z std {std}"
        );
        assert!(
            missing_count * 100 <= trials.len(),
            "{name} {label}: {missing:?} tests missing"
        );
    }
    let (mean_z2, se) = bootstrap_mean(&pooled_z2(trials));
    eprintln!("  pooled mean z² {mean_z2:.4} ± {se:.4}");
    assert!(
        (mean_z2 - 1.0).abs() <= 3.0 * se,
        "{name}: pooled mean z² {mean_z2}"
    );
    if independent {
        gate_joint(name, trials.iter().map(|t| &t.joint), trials.len());
    } else {
        assert!(
            trials
                .iter()
                .all(|t| t.joint == Err("NestedJoint".to_string())),
            "{name}: a nested joint test"
        );
    }
}

/// Each trial's mean `z²` over its parameters.
fn pooled_z2(trials: &[Outcome]) -> Vec<f64> {
    trials
        .iter()
        .filter_map(|t| {
            let z: Vec<f64> =
                t.z.iter()
                    .filter_map(|(_, z)| z.as_ref().ok().copied())
                    .collect();
            (!z.is_empty()).then(|| z.iter().map(|z| z * z).sum::<f64>() / z.len() as f64)
        })
        .collect()
}

/// The joint `χ²` gate: its rejection rate within 3.5 Monte-Carlo errors,
/// its mean over its degrees of freedom within 3 bootstrap errors of 1, at
/// most 1 % missing.
fn gate_joint<'a>(
    name: &str,
    joints: impl Iterator<Item = &'a Result<(f64, usize), String>>,
    n: usize,
) {
    let joints: Vec<&Result<(f64, usize), String>> = joints.collect();
    let available: Vec<(f64, usize)> = joints
        .iter()
        .filter_map(|j| j.as_ref().ok().copied())
        .collect();
    let missing = joints.len() - available.len();
    let rate = available
        .iter()
        .filter(|(chi2, dof)| chi2_quantile_95(*dof) < *chi2)
        .count() as f64
        / available.len() as f64;
    let ratios: Vec<f64> = available
        .iter()
        .map(|(chi2, dof)| chi2 / *dof as f64)
        .collect();
    let (mean, se) = bootstrap_mean(&ratios);
    let err = mc_error(available.len());
    let mut dofs: BTreeMap<usize, usize> = BTreeMap::new();
    for (_, dof) in &available {
        *dofs.entry(*dof).or_default() += 1;
    }
    eprintln!(
        "  joint χ²: reject {rate:.4} ± {err:.4}, mean χ²/dof {mean:.4} ± {se:.4}, dof {dofs:?}, missing {missing}"
    );
    assert!(
        (rate - LEVEL).abs() <= 3.5 * err,
        "{name}: joint rejection rate {rate}"
    );
    assert!(
        (mean - 1.0).abs() <= 3.0 * se,
        "{name}: joint mean χ²/dof {mean}"
    );
    assert!(missing * 100 <= n, "{name}: {missing} joint tests missing");
}

/// The 95 % point of a χ² of `dof` degrees of freedom (the gates' degrees
/// of freedom).
fn chi2_quantile_95(dof: usize) -> f64 {
    match dof {
        1 => 3.841_458_820_694_124,
        2 => 5.991_464_547_107_979,
        3 => 7.814_727_903_251_178,
        4 => 9.487_729_036_781_154,
        5 => 11.070_497_693_516_351,
        6 => CHI2_6_95,
        7 => 14.067_140_449_340_169,
        8 => 15.507_313_055_865_453,
        9 => 16.918_977_604_620_448,
        10 => 18.307_038_053_275_146,
        _ => panic!("no 95 % point of a χ² of {dof} degrees of freedom"),
    }
}

// ── Planar cells ────────────────────────────────────────────────────────────

/// How the other run of a cell is made from the reference capture.
#[derive(Clone, Copy, Debug)]
enum Plan {
    /// Another capture of as many views.
    Independent,
    /// The capture without its last `n` views.
    LeaveViews(usize),
    /// The capture without each feature with probability `p`.
    DropFeatures(f64),
    /// The capture without `n` features of random views.
    LeaveFeatures(usize),
}

/// Leaves out `n` features, each from a random view at a random position.
fn leave_features(data: &PlanarDataset, n: usize, seed: u64) -> PlanarDataset {
    let mut rng = Rng::new(seed ^ 0x5EED);
    let mut views: Vec<(Vec<Pt3>, Vec<Pt2>)> = data
        .views
        .iter()
        .map(|v| (v.obs.points_3d.clone(), v.obs.points_2d.clone()))
        .collect();
    for _ in 0..n {
        let v = ((rng.uniform() * views.len() as f64) as usize).min(views.len() - 1);
        let k = ((rng.uniform() * views[v].0.len() as f64) as usize).min(views[v].0.len() - 1);
        views[v].0.remove(k);
        views[v].1.remove(k);
    }
    PlanarDataset::new(
        views
            .into_iter()
            .map(|(p3, p2)| View::without_meta(CorrespondenceView::new(p3, p2).expect("view")))
            .collect(),
    )
    .expect("subset")
}

/// The comparison of trial `i` of a cell.
fn planar_pair(
    plan: Plan,
    model: Model,
    views: usize,
    noise: f64,
    loss: RobustLoss,
    i: u64,
) -> Option<Outcome> {
    // Each cell draws its own captures.
    let cell = ((views as u64) << 32) ^ (noise.to_bits() >> 32) ^ ((model as u64) << 48);
    let seed = (1000 * i + 7) ^ cell;
    let full = capture(model, views, noise, seed);
    let reference = calibrate(&full, model, BACKEND, loss)?;
    let nested = DataRelation::Nested {
        smaller: RunSide::Other,
    };
    let (other_data, relation) = match plan {
        Plan::Independent => (
            capture(model, views, noise, seed + 500_000),
            DataRelation::Independent,
        ),
        Plan::LeaveViews(n) => (first_views(&full, views - n), nested),
        Plan::DropFeatures(p) => (drop_features(&full, p, seed), nested),
        Plan::LeaveFeatures(n) => (leave_features(&full, n, seed), nested),
    };
    let other = calibrate(&other_data, model, BACKEND, loss)?;
    Some(outcome_of(&compare(&reference, &other, relation)))
}

fn planar_trials(
    plan: Plan,
    model: Model,
    views: usize,
    noise: f64,
    loss: RobustLoss,
    n: usize,
) -> (Vec<Outcome>, usize) {
    run_trials(n, |i| planar_pair(plan, model, views, noise, loss, i))
}

fn planar_cell(plan: Plan, model: Model, views: usize, noise: f64, loss: RobustLoss, n: usize) {
    let (trials, failed) = planar_trials(plan, model, views, noise, loss, n);
    gate_null(
        &format!("{plan:?} {model:?} {views} views {noise} px {loss:?}"),
        &trials,
        failed,
        matches!(plan, Plan::Independent),
    );
}

/// The backend of the Monte-Carlo gates: the two backends reach the same
/// optimum (the regression suite checks them against each other), so a
/// second run with the same seeds would repeat the draw.
const BACKEND: SolverBackend = SolverBackend::TinySolver;

/// Trials of a nested cell.
fn nested_trials() -> usize {
    trials().max(1000)
}

#[test]
#[ignore = "Monte-Carlo gate; release only"]
fn independent_captures_reject_at_the_nominal_rate() {
    planar_cell(
        Plan::Independent,
        Model::Brown,
        20,
        0.3,
        RobustLoss::None,
        trials(),
    );
}

#[test]
#[ignore = "Monte-Carlo gate; release only"]
fn nested_runs_reject_at_the_nominal_rate() {
    let n = nested_trials();
    for (plan, views) in [
        (Plan::LeaveViews(6), 20),
        (Plan::DropFeatures(0.2), 20),
        (Plan::LeaveViews(1), 8),
    ] {
        planar_cell(plan, Model::Brown, views, 0.3, RobustLoss::None, n);
    }
}

#[test]
#[ignore = "Monte-Carlo gate; release only"]
fn nested_runs_under_a_huber_loss() {
    let n = nested_trials();
    for plan in [Plan::LeaveViews(6), Plan::DropFeatures(0.2)] {
        planar_cell(
            plan,
            Model::Brown,
            20,
            0.3,
            RobustLoss::Huber { scale: 0.5 },
            n,
        );
    }
}

/// A limitation, pinned: when the left-out data are a small share of the
/// information (1 of 25 views, 6 of about 2000 features), the nested `z` of
/// some parameters has heavier tails than a standard normal even on one
/// noise level. Each fit's information is evaluated at its own estimate,
/// and the difference of two nearly equal informations is then dominated
/// by that mismatch. The test prints the cells and checks that the excess
/// is there and bounded; a fix that removes it makes this test fail, to be
/// replaced by the strict gate.
#[test]
#[ignore = "Monte-Carlo characterization; release only"]
fn nested_small_exclusions_have_heavier_tails() {
    let n = nested_trials();
    let mut worst: f64 = 0.0;
    for (plan, views) in [(Plan::LeaveViews(1), 25), (Plan::LeaveFeatures(6), 20)] {
        let (trials, failed) = planar_trials(plan, Model::Brown, views, 0.3, RobustLoss::None, n);
        eprintln!(
            "{plan:?} {views} views: {} trials, {failed} failed",
            trials.len()
        );
        for (label, (z, missing)) in &by_label(&trials) {
            let (rate, _, std) = moments(z);
            eprintln!(
                "  {label:>8}: n {:5}  reject {rate:.4} ± {:.4}  z std {std:.3}  missing {missing:?}",
                z.len(),
                mc_error(z.len())
            );
            worst = worst.max(std);
            // The rejection rate at 5 % stays within 0.05 of the level.
            assert!(rate < 0.10, "{plan:?} {label}: rejection rate {rate}");
        }
    }
    assert!(
        worst > 1.1,
        "the tails are no longer heavier: z std {worst}"
    );
}

/// The alternative: the larger run holds gross outliers (1 % of its
/// features 8 px off), which the smaller run leaves out. The comparison
/// must reject. The left-out share is small, so some parameters' variance
/// difference comes out not positive (measured: 18 % of the tests, typed
/// `NestedVarianceNotPositive`); at least three in four must be available.
#[test]
#[ignore = "Monte-Carlo gate; release only"]
fn nested_runs_detect_left_out_outliers() {
    let n = trials();
    let (trials, failed) = run_trials(n, |i| {
        let seed = 31 * i + 11;
        let clean = capture(Model::Brown, 20, 0.3, seed);
        let mut rng = Rng::new(seed ^ 0x0BAD);
        // The full run: 1 % of the features moved 8 px; the smaller run: the
        // others, untouched.
        let mut full_views = Vec::new();
        let mut kept_views = Vec::new();
        for v in &clean.views {
            let (mut f3, mut f2, mut k3, mut k2) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for (p3, p2) in v.obs.points_3d.iter().zip(&v.obs.points_2d) {
                if rng.uniform() < 0.01 {
                    let a = rng.uniform() * std::f64::consts::TAU;
                    f3.push(*p3);
                    f2.push(Pt2::new(p2.x + 8.0 * a.cos(), p2.y + 8.0 * a.sin()));
                } else {
                    f3.push(*p3);
                    f2.push(*p2);
                    k3.push(*p3);
                    k2.push(*p2);
                }
            }
            full_views.push(View::without_meta(
                CorrespondenceView::new(f3, f2).expect("view"),
            ));
            kept_views.push(View::without_meta(
                CorrespondenceView::new(k3, k2).expect("view"),
            ));
        }
        let full = PlanarDataset::new(full_views).expect("full");
        let kept = PlanarDataset::new(kept_views).expect("kept");
        let reference = calibrate(&full, Model::Brown, BACKEND, RobustLoss::None)?;
        let other = calibrate(&kept, Model::Brown, BACKEND, RobustLoss::None)?;
        let d = compare(
            &reference,
            &other,
            DataRelation::Nested {
                smaller: RunSide::Other,
            },
        );
        Some(outcome_of(&d))
    });
    let tests: usize = trials.iter().map(|t| t.z.len()).sum();
    let available: usize = trials
        .iter()
        .map(|t| t.z.iter().filter(|(_, z)| z.is_ok()).count())
        .sum();
    let rejecting = trials
        .iter()
        .filter(|t| {
            t.z.iter()
                .any(|(_, z)| z.as_ref().is_ok_and(|z| z.abs() > Z_95))
        })
        .count() as f64
        / trials.len() as f64;
    let (mean_z2, _) = bootstrap_mean(&pooled_z2(&trials));
    eprintln!(
        "outliers left out: {} trials, {failed} failed, {available} of {tests} tests available, \
         some parameter rejects in {rejecting:.4}, pooled mean z² {mean_z2:.1}",
        trials.len()
    );
    assert!(available * 4 >= 3 * tests, "tests are missing");
    assert!(
        rejecting >= 0.8,
        "the comparison rejects in only {rejecting}"
    );
}

#[test]
#[ignore = "Monte-Carlo sweep; release only"]
fn sweep_over_views_noise_and_models() {
    let n = trials() / 2;
    let mut cells = Vec::new();
    for views in [8usize, 20, 60] {
        for noise in [0.05, 0.3] {
            cells.push((Model::Brown, views, noise));
        }
    }
    cells.push((Model::Fisheye, 20, 0.3));
    cells.push((Model::Scheimpflug, 20, 0.3));
    for (model, views, noise) in cells {
        for plan in [
            Plan::Independent,
            Plan::LeaveViews((views * 3).div_ceil(10)),
        ] {
            planar_cell(plan, model, views, noise, RobustLoss::None, n);
        }
    }
}

// ── Rig ─────────────────────────────────────────────────────────────────────

fn rig_chi2(
    reference: &RigExtrinsicsExport,
    other: &RigExtrinsicsExport,
    relation: DataRelation,
) -> Result<(f64, usize), String> {
    let ids = vec!["left".to_string(), "right".to_string()];
    let runs = [
        ComparedRun::new("reference", reference, ids.clone(), vec![SIZE; 2]),
        ComparedRun::new("other", other, ids, vec![SIZE; 2]).with_relation(relation),
    ];
    let comparison = compare_runs(&runs, &CompareRunsOptions::new()).expect("comparison");
    assert_eq!(comparison.differences[0].relation_refused, None);
    joint_of(&comparison.differences[0].rig[0].test)
}

#[test]
#[ignore = "Monte-Carlo gate; release only"]
fn rig_pose_chi2_rejects_at_the_nominal_rate() {
    // Independent captures of 20 views.
    let (results, failed) = run_trials(trials(), |i| {
        let seed = 77 * i + 3;
        let reference = calibrate_rig(&rig_capture(20, 0.3, seed), BACKEND)?;
        let other = calibrate_rig(&rig_capture(20, 0.3, seed + 900_000), BACKEND)?;
        Some(rig_chi2(&reference, &other, DataRelation::Independent))
    });
    assert!(
        failed * 100 <= results.len(),
        "{failed} failed calibrations"
    );
    gate_joint("rig independent", results.iter(), results.len());
    // Nested: 2 of 8 views left out.
    let (results, failed) = run_trials(nested_trials(), |i| {
        let seed = 91 * i + 5;
        let full = rig_capture(8, 0.3, seed);
        let mut sub = full.clone();
        sub.views.truncate(6);
        let reference = calibrate_rig(&full, BACKEND)?;
        let other = calibrate_rig(&sub, BACKEND)?;
        Some(rig_chi2(
            &reference,
            &other,
            DataRelation::Nested {
                smaller: RunSide::Other,
            },
        ))
    });
    assert!(
        failed * 100 <= results.len(),
        "{failed} failed calibrations"
    );
    gate_joint("rig nested, 2 of 8 views", results.iter(), results.len());
}
