//! Monte-Carlo gates of the tests `compare_runs` reports, on synthetic
//! captures under the null hypothesis (both runs estimate one device, the
//! model is right): each test's rejection rate at the 5 % level must be
//! 5 % within the Monte-Carlo error.
//!
//! - **Independent** (`z = Δ / √(σ_ref² + σ_other²)`): two captures of one
//!   camera with their own poses and noise.
//! - **Nested** (`z = Δ / √(σ_smaller² − σ_larger²)`): a capture against the
//!   same capture with 30 % of its views, or 20 % of its features, left
//!   out; plain least squares and a Huber loss.
//! - The 6-dof `χ²` of a two-camera rig's relative pose, independent and
//!   nested.
//!
//! Every gate runs on both solver backends; a sweep over the number of
//! views, the noise and the camera model (pinhole Brown-Conrady, a
//! Kannala-Brandt fisheye, a Scheimpflug sensor) runs on the default one.
//! The trials of a gate run on their own threads. Release-only:
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
    CompareRunsOptions, ComparedRun, DataRelation, DifferenceTest, ParameterComparison, PoseTest,
    RunSide, compare_runs,
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

/// The z of every parameter's test, by label, and the reasons of the
/// parameters without one.
fn parameter_z(
    reference: &PlanarIntrinsicsExport,
    other: &PlanarIntrinsicsExport,
    relation: DataRelation,
) -> Vec<(String, Result<f64, String>)> {
    let ids = vec!["cam".to_string()];
    let runs = [
        ComparedRun::new("reference", reference, ids.clone(), vec![SIZE]),
        ComparedRun::new("other", other, ids, vec![SIZE]),
    ];
    let options = CompareRunsOptions::new().with_relations(vec![relation]);
    let comparison = compare_runs(&runs, &options).expect("comparison");
    let ParameterComparison::SameModel { parameters } =
        &comparison.differences[0].cameras[0].parameters
    else {
        panic!("same model");
    };
    parameters
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
        .collect()
}

/// The statistics of one parameter's z over the trials.
#[derive(Default)]
struct Tally {
    z: Vec<f64>,
    missing: BTreeMap<String, usize>,
}

impl Tally {
    fn rate(&self) -> f64 {
        self.z.iter().filter(|z| z.abs() > Z_95).count() as f64 / self.z.len().max(1) as f64
    }

    fn std(&self) -> f64 {
        let n = self.z.len() as f64;
        let mean = self.z.iter().sum::<f64>() / n;
        (self.z.iter().map(|z| (z - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
    }
}

/// The Monte-Carlo standard error of a rejection rate at the nominal level
/// over `n` trials.
fn mc_error(n: usize) -> f64 {
    (LEVEL * (1.0 - LEVEL) / n as f64).sqrt()
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

/// Pools trials into tallies per label and prints them; returns the worst
/// rate deviation in Monte-Carlo errors and the z-std range.
fn report(
    name: &str,
    trials: &[Vec<(String, Result<f64, String>)>],
    failed: usize,
) -> (f64, f64, f64) {
    let mut tallies: BTreeMap<String, Tally> = BTreeMap::new();
    for trial in trials {
        for (label, z) in trial {
            let t = tallies.entry(label.clone()).or_default();
            match z {
                Ok(z) => t.z.push(*z),
                Err(reason) => *t.missing.entry(reason.clone()).or_default() += 1,
            }
        }
    }
    let (mut worst, mut lo, mut hi) = (0.0f64, f64::INFINITY, 0.0f64);
    eprintln!(
        "{name}: {} trials, {failed} failed calibrations",
        trials.len()
    );
    for (label, t) in &tallies {
        if t.z.len() < 20 {
            eprintln!("  {label:>8}: {} tests, missing {:?}", t.z.len(), t.missing);
            continue;
        }
        let err = mc_error(t.z.len());
        let dev = (t.rate() - LEVEL).abs() / err;
        worst = worst.max(dev);
        lo = lo.min(t.std());
        hi = hi.max(t.std());
        eprintln!(
            "  {label:>8}: n {:4}  reject {:.3} ± {:.3}  z std {:.3}  missing {:?}",
            t.z.len(),
            t.rate(),
            err,
            t.std(),
            t.missing
        );
    }
    (worst, lo, hi)
}

#[derive(Clone, Copy, Debug)]
enum Plan {
    Independent,
    NestedViews,
    NestedFeatures,
}

fn planar_gate(
    plan: Plan,
    model: Model,
    views: usize,
    noise: f64,
    backend: SolverBackend,
    loss: RobustLoss,
    n: usize,
) -> (f64, f64, f64) {
    let (trials, failed) = run_trials(n, |i| {
        // Each cell draws its own captures: cells that differ only in the
        // noise level would otherwise repeat one draw, scaled.
        let cell = ((views as u64) << 32) ^ (noise.to_bits() >> 32) ^ ((model as u64) << 48);
        let seed = (1000 * i + 7) ^ cell;
        let full = capture(model, views, noise, seed);
        let reference = calibrate(&full, model, backend, loss)?;
        let (other_data, relation) = match plan {
            Plan::Independent => (
                capture(model, views, noise, seed + 500_000),
                DataRelation::Independent,
            ),
            Plan::NestedViews => (
                first_views(&full, (views * 7).div_ceil(10)),
                DataRelation::Nested {
                    smaller: RunSide::Other,
                },
            ),
            Plan::NestedFeatures => (
                drop_features(&full, 0.2, seed),
                DataRelation::Nested {
                    smaller: RunSide::Other,
                },
            ),
        };
        let other = calibrate(&other_data, model, backend, loss)?;
        Some(parameter_z(&reference, &other, relation))
    });
    report(
        &format!("{plan:?} {model:?} {views} views {noise} px {backend:?} {loss:?}"),
        &trials,
        failed,
    )
}

fn backends() -> [SolverBackend; 2] {
    [SolverBackend::TinySolver, SolverBackend::Factrs]
}

/// The gate over the parameters of one camera, tested together: each
/// rejection rate within 3.5 Monte-Carlo errors of 5 % and each z standard
/// deviation within `3.4 / √(2n)` of 1 (the sample standard deviation's own
/// error is `1 / √(2n)`): a family-wise false-alarm rate of about 1 % over
/// eight parameters.
fn assert_gate(result: (f64, f64, f64), n: usize, what: &str) {
    let (worst, lo, hi) = result;
    let tolerance = 3.4 / (2.0 * n as f64).sqrt();
    assert!(
        worst <= 3.5,
        "{what}: a rejection rate is {worst:.1} Monte-Carlo errors off 5 %"
    );
    assert!(
        lo >= 1.0 - tolerance && hi <= 1.0 + tolerance,
        "{what}: z std in [{lo:.3}, {hi:.3}], allowed 1 ± {tolerance:.3}"
    );
}

#[test]
#[ignore = "Monte-Carlo gate; release only"]
fn independent_captures_reject_at_the_nominal_rate() {
    for backend in backends() {
        let r = planar_gate(
            Plan::Independent,
            Model::Brown,
            20,
            0.3,
            backend,
            RobustLoss::None,
            trials(),
        );
        assert_gate(r, trials(), &format!("independent {backend:?}"));
    }
}

#[test]
#[ignore = "Monte-Carlo gate; release only"]
fn nested_runs_reject_at_the_nominal_rate() {
    for backend in backends() {
        for plan in [Plan::NestedViews, Plan::NestedFeatures] {
            let r = planar_gate(
                plan,
                Model::Brown,
                20,
                0.3,
                backend,
                RobustLoss::None,
                trials(),
            );
            assert_gate(r, trials(), &format!("{plan:?} {backend:?}"));
        }
    }
}

#[test]
#[ignore = "Monte-Carlo gate; release only"]
fn nested_runs_under_a_huber_loss() {
    for backend in backends() {
        for plan in [Plan::NestedViews, Plan::NestedFeatures] {
            let r = planar_gate(
                plan,
                Model::Brown,
                20,
                0.3,
                backend,
                RobustLoss::Huber { scale: 0.5 },
                trials(),
            );
            assert_gate(r, trials(), &format!("Huber {plan:?} {backend:?}"));
        }
    }
}

#[test]
#[ignore = "Monte-Carlo sweep; release only"]
fn sweep_over_views_noise_and_models() {
    let n = trials() / 2;
    let mut cells = Vec::new();
    for views in [8, 20, 60] {
        for noise in [0.05, 0.3] {
            cells.push((Model::Brown, views, noise));
        }
    }
    cells.push((Model::Fisheye, 20, 0.3));
    cells.push((Model::Scheimpflug, 20, 0.3));
    for (model, views, noise) in cells {
        for plan in [Plan::Independent, Plan::NestedViews] {
            let r = planar_gate(
                plan,
                model,
                views,
                noise,
                SolverBackend::TinySolver,
                RobustLoss::None,
                n,
            );
            eprintln!(
                "  -> worst {:.2} MC errors, z std [{:.3}, {:.3}]",
                r.0, r.1, r.2
            );
            assert_gate(
                r,
                n,
                &format!("{plan:?} {model:?} {views} views {noise} px"),
            );
        }
    }
}

// ── Rig ─────────────────────────────────────────────────────────────────────

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

fn rig_chi2(
    reference: &RigExtrinsicsExport,
    other: &RigExtrinsicsExport,
    relation: DataRelation,
) -> Result<f64, String> {
    let ids = vec!["left".to_string(), "right".to_string()];
    let runs = [
        ComparedRun::new("reference", reference, ids.clone(), vec![SIZE; 2]),
        ComparedRun::new("other", other, ids, vec![SIZE; 2]),
    ];
    let options = CompareRunsOptions::new().with_relations(vec![relation]);
    let comparison = compare_runs(&runs, &options).expect("comparison");
    match &comparison.differences[0].rig[0].test {
        PoseTest::Independent { chi2, .. } | PoseTest::Nested { chi2, .. } => Ok(*chi2),
        PoseTest::NotAvailable { reason } => Err(format!("{reason:?}")),
        other => Err(format!("{other:?}")),
    }
}

#[test]
#[ignore = "Monte-Carlo gate; release only"]
fn rig_pose_chi2_rejects_at_the_nominal_rate() {
    let n = trials();
    for backend in backends() {
        for nested in [false, true] {
            let (results, failed) = run_trials(n, |i| {
                let seed = 77 * i + 3;
                let full = rig_capture(20, 0.3, seed);
                let reference = calibrate_rig(&full, backend)?;
                let (other, relation) = if nested {
                    let mut sub = full.clone();
                    sub.views.truncate(14);
                    (
                        sub,
                        DataRelation::Nested {
                            smaller: RunSide::Other,
                        },
                    )
                } else {
                    (
                        rig_capture(20, 0.3, seed + 900_000),
                        DataRelation::Independent,
                    )
                };
                let other = calibrate_rig(&other, backend)?;
                Some(rig_chi2(&reference, &other, relation))
            });
            let chi2: Vec<f64> = results
                .iter()
                .filter_map(|r| r.as_ref().ok().copied())
                .collect();
            let missing = results.len() - chi2.len();
            let rate = chi2.iter().filter(|c| **c > CHI2_6_95).count() as f64 / chi2.len() as f64;
            let mean = chi2.iter().sum::<f64>() / chi2.len() as f64;
            let err = mc_error(chi2.len());
            eprintln!(
                "rig {} {backend:?}: {} tests ({missing} without, {failed} failed calibrations), \
                 reject {rate:.3} ± {err:.3}, mean χ² {mean:.2} (6)",
                if nested { "nested" } else { "independent" },
                chi2.len()
            );
            assert!(
                (rate - LEVEL).abs() <= 3.0 * err,
                "rig rejection rate {rate}"
            );
            assert!((5.4..=6.6).contains(&mean), "rig mean χ² {mean}");
        }
    }
}
