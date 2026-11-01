//! Tests of next-view planning: the prediction against a re-solve with the
//! view added, on synthetic planar scenes, and the API's contracts.
//!
//! The gate that compares predictions with re-solves over several distortion
//! models, seeds and candidates solves a few dozen calibrations, so it is
//! ignored in the default (debug) run; CI runs it in release:
//! `cargo test --release -p vision-calibration-pipeline --lib next_view -- --ignored --nocapture`.

use nalgebra::{Rotation3, Translation3, Vector3};
use vision_calibration_core::{
    BrownConrady5, CameraModelSpec, CorrespondenceView, DistortionKind, DistortionParams, FxFyCxCy,
    PlanarDataset, ProjectionParams, ScheimpflugParams, SensorKind, SensorParams,
    UncertaintyUnavailable, View, synthetic::planar,
};

use super::*;
use crate::planar_intrinsics::{PlanarIntrinsicsConfig, PlanarIntrinsicsProblem, run_calibration};
use crate::session::CalibrationSession;

// ─────────────────────────────────────────────────────────────────────────────
// Fixtures
// ─────────────────────────────────────────────────────────────────────────────

const IMAGE: [u32; 2] = [1280, 960];

/// The camera models the tests calibrate.
#[derive(Debug, Clone, Copy)]
enum Model {
    /// Brown-Conrady, `k3` fixed (the default).
    BrownConrady,
    /// The one-parameter division model.
    Division,
    /// Brown-Conrady behind a tilted (Scheimpflug) sensor, the low-order
    /// radial terms and the tilt free (the default).
    Scheimpflug,
}

impl Model {
    fn spec(self) -> CameraModelSpec {
        let (distortion, sensor) = match self {
            Self::BrownConrady => (DistortionKind::BrownConrady5, SensorKind::Identity),
            Self::Division => (DistortionKind::Division1, SensorKind::Identity),
            Self::Scheimpflug => (DistortionKind::BrownConrady5, SensorKind::Scheimpflug),
        };
        CameraModelSpec::pinhole(distortion, sensor)
    }
}

/// The true camera of `model`.
fn truth(model: Model) -> CameraParams {
    let brown_conrady = DistortionParams::BrownConrady5 {
        params: BrownConrady5 {
            k1: -0.08,
            k2: 0.02,
            k3: 0.0,
            p1: 0.0005,
            p2: -0.0003,
            iters: 8,
        },
    };
    let (distortion, sensor) = match model {
        Model::BrownConrady => (brown_conrady, SensorParams::Identity),
        Model::Division => (
            DistortionParams::Division { lambda: -0.15 },
            SensorParams::Identity,
        ),
        Model::Scheimpflug => (
            DistortionParams::BrownConrady5 {
                params: BrownConrady5 {
                    k1: -0.08,
                    k2: 0.02,
                    k3: 0.0,
                    p1: 0.0,
                    p2: 0.0,
                    iters: 8,
                },
            },
            SensorParams::Scheimpflug {
                params: ScheimpflugParams {
                    tilt_x: 0.03,
                    tilt_y: -0.02,
                },
            },
        ),
    };
    CameraParams {
        projection: ProjectionParams::Pinhole,
        distortion,
        sensor,
        intrinsics: FxFyCxCy {
            fx: 1000.0,
            fy: 1000.0,
            cx: 640.0,
            cy: 480.0,
        },
    }
}

/// A 9 × 7 board of 30 mm cells.
fn board() -> Vec<Pt3> {
    planar::grid_points(9, 7, 0.03)
}

/// The board posed so its centre lands near pixel `at` at depth `z`, tilted
/// by `(pitch, yaw)`.
fn board_pose(at: [f64; 2], z: f64, pitch: f64, yaw: f64) -> Iso3 {
    let center = Vector3::new(0.12, 0.09, 0.0);
    let r = Rotation3::from_euler_angles(pitch, yaw, 0.0);
    let ray = Vector3::new((at[0] - 640.0) / 1000.0, (at[1] - 480.0) / 1000.0, 1.0);
    Iso3::from_parts(Translation3::from(ray * z - r * center), r.into())
}

/// Six tilted views over the central part of the image: the corners are
/// left to extrapolation, so there is something to plan for.
fn base_poses() -> Vec<Iso3> {
    vec![
        board_pose([420.0, 330.0], 0.6, 0.35, -0.3),
        board_pose([860.0, 330.0], 0.6, -0.35, 0.3),
        board_pose([420.0, 630.0], 0.6, -0.3, -0.35),
        board_pose([860.0, 630.0], 0.6, 0.3, 0.35),
        board_pose([640.0, 480.0], 0.65, 0.0, 0.4),
        board_pose([640.0, 480.0], 0.65, 0.4, 0.0),
    ]
}

/// The candidates every test ranks: 3 × 3 regions, ±0.4 rad tilts, 0.55 m.
fn candidates(camera: &CameraParams) -> Vec<Iso3> {
    candidate_poses(
        camera,
        &board(),
        IMAGE,
        &CandidateGrid::new(0.55, 0.4, [3, 3]),
    )
    .expect("candidates")
}

/// Seeded Gaussian pixel noise (xorshift and Box–Muller).
struct Noise {
    state: u64,
    sigma: f64,
}

impl Noise {
    fn new(seed: u64, sigma: f64) -> Self {
        Self {
            state: 0x9E37_79B9_7F4A_7C15 ^ seed.wrapping_mul(0x2545_F491_4F6C_DD1D),
            sigma,
        }
    }

    fn uniform(&mut self) -> f64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        ((self.state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn sample(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        self.sigma * (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

/// The view of the board at `pose` through the true camera: the corners that
/// land in the image, with noise.
fn observe(
    truth: &CameraParams,
    pose: &Iso3,
    noise: &mut Noise,
) -> View<vision_calibration_core::NoMeta> {
    let camera = truth.build().expect("camera");
    let [w, h] = IMAGE.map(f64::from);
    let (points_3d, points_2d): (Vec<Pt3>, Vec<Pt2>) = board()
        .into_iter()
        .filter_map(|p| {
            let q = camera.project_point_c(&pose.transform_point(&p).coords)?;
            ((0.0..=w).contains(&q.x) && (0.0..=h).contains(&q.y))
                .then(|| (p, Pt2::new(q.x + noise.sample(), q.y + noise.sample())))
        })
        .unzip();
    View::without_meta(CorrespondenceView::new(points_3d, points_2d).expect("view"))
}

fn calibrate(
    model: Model,
    views: Vec<View<vision_calibration_core::NoMeta>>,
) -> PlanarIntrinsicsExport {
    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    let mut config = PlanarIntrinsicsConfig::default();
    config.camera.model = model.spec();
    session.set_config(config).expect("config");
    session
        .set_input(PlanarDataset::new(views).expect("views"))
        .expect("input");
    run_calibration(&mut session).expect("calibration");
    session.export().expect("export")
}

/// A calibration from the base views, and the views and noise behind it.
struct Scene {
    model: Model,
    truth: CameraParams,
    views: Vec<View<vision_calibration_core::NoMeta>>,
    noise: Noise,
    export: PlanarIntrinsicsExport,
}

impl Scene {
    fn new(model: Model, seed: u64, sigma: f64) -> Self {
        let truth = truth(model);
        let mut noise = Noise::new(seed, sigma);
        let views: Vec<_> = base_poses()
            .iter()
            .map(|pose| observe(&truth, pose, &mut noise))
            .collect();
        let export = calibrate(model, views.clone());
        Self {
            model,
            truth,
            views,
            noise,
            export,
        }
    }

    /// The calibration re-solved with views of the board at `poses` added.
    fn resolve(&mut self, poses: &[Iso3]) -> PlanarIntrinsicsExport {
        let mut views = self.views.clone();
        for pose in poses {
            views.push(observe(&self.truth, pose, &mut self.noise));
        }
        calibrate(self.model, views)
    }
}

/// The board's points, for planners that borrow them.
static BOARD: std::sync::OnceLock<Vec<Pt3>> = std::sync::OnceLock::new();

fn planner(export: &PlanarIntrinsicsExport) -> Planner<'_> {
    Planner::new(export, BOARD.get_or_init(board), IMAGE).expect("planner")
}

/// The prediction for `candidate` against the re-solve with it, at the
/// current result's pixel σ (the re-solve's covariance rescaled from its own
/// σ̂, which fluctuates with the noise of the added view).
struct Agreement {
    labels: Vec<String>,
    /// Predicted σ / re-solved σ, per label.
    sigma_ratio: Vec<f64>,
    /// Predicted / re-solved maximum projection σ.
    projection_ratio: f64,
    /// The re-solve's own pixel σ over the current one's.
    noise_ratio: f64,
    /// Current maximum projection σ minus the re-solved one (rescaled), px.
    real_reduction: f64,
    /// The current maximum projection σ, px.
    current_max: f64,
}

fn agreement(scene: &mut Scene, candidates: &[Iso3]) -> Agreement {
    let predicted =
        predict_with_views(&scene.export, candidates, &board(), IMAGE).expect("predict");
    let base = scene.export.clone();
    let current = planner(&base);
    let current_max = current.projection(&current.covariance).expect("map").max;
    let resolved = scene.resolve(candidates);
    let after = planner(&resolved);
    let scale = current.sigma2 / after.sigma2;
    let cov = &after.covariance;
    assert_eq!(cov.labels, predicted.covariance.labels);
    let sigma_ratio = (0..cov.labels.len())
        .map(|i| (predicted.covariance.covariance[i][i] / (cov.covariance[i][i] * scale)).sqrt())
        .collect();
    let resolved_max = after.projection(cov).expect("map").max * scale.sqrt();
    Agreement {
        labels: cov.labels.clone(),
        sigma_ratio,
        projection_ratio: predicted.projection.max / resolved_max,
        noise_ratio: (after.sigma2 / current.sigma2).sqrt(),
        real_reduction: current_max - resolved_max,
        current_max,
    }
}

fn mean_abs_deviation(ratios: &[f64]) -> f64 {
    ratios.iter().map(|r| (r - 1.0).abs()).sum::<f64>() / ratios.len() as f64
}

// ─────────────────────────────────────────────────────────────────────────────
// Scale and identity
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn the_covariance_is_its_rebuilt_information_inverse_on_one_scale() {
    let scene = Scene::new(Model::BrownConrady, 1, 0.2);
    let planner = planner(&scene.export);
    let covariance = scene.export.uncertainty.covariance().expect("covariance");
    let existing = existing_information(&scene.export, covariance, &planner.free).unwrap();
    // `Σ S_A / σ²` is the identity: the rebuilt information is the one the
    // covariance inverted, so one scalar relates them. (Ten seeds put the
    // largest deviation between 4e-7 and 3e-6, the precision of the solve's
    // own inversion.)
    let product = &planner.covariance_matrix * existing / planner.sigma2;
    let n = product.nrows();
    let deviation = (&product - DMatrix::<f64>::identity(n, n)).abs().max();
    assert!(deviation < 1e-5, "Σ S_A / σ² − I up to {deviation:e}");
    // The implied pixel σ is near the noise (0.2 px).
    let sigma = planner.sigma2.sqrt();
    assert!((0.17..0.23).contains(&sigma), "σ = {sigma}");
}

#[test]
fn a_view_off_the_image_changes_nothing() {
    let scene = Scene::new(Model::BrownConrady, 2, 0.2);
    let behind = Iso3::translation(0.0, 0.0, -1.0);
    let predicted = predict_with_views(&scene.export, &[behind], &board(), IMAGE).unwrap();
    assert_eq!(predicted.visible_points, [0]);
    let current = scene.export.uncertainty.covariance().unwrap();
    for (p, c) in predicted
        .covariance
        .covariance
        .iter()
        .zip(&current.covariance)
    {
        for (a, b) in p.iter().zip(c) {
            assert!((a - b).abs() <= 1e-9 * b.abs().max(1e-12), "{a} vs {b}");
        }
    }
    assert_eq!(predicted.covariance.dof, current.dof);
}

// ─────────────────────────────────────────────────────────────────────────────
// Prediction against a re-solve (fast)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn prediction_matches_a_resolve() {
    let mut scene = Scene::new(Model::BrownConrady, 3, 0.2);
    let candidates = candidates(&scene.export.params.camera);
    // A view tilted about its x axis in the top-left region.
    let a = agreement(&mut scene, &candidates[1..2]);
    let deviation = mean_abs_deviation(&a.sigma_ratio);
    assert!(
        deviation < 0.05,
        "σ ratios {:?} for {:?}",
        a.sigma_ratio,
        a.labels
    );
    assert!(
        (a.projection_ratio - 1.0).abs() < 0.1,
        "projection σ ratio {}",
        a.projection_ratio
    );
    assert!(a.real_reduction > 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Ranking and candidates
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn ranking_is_ordered_and_deterministic() {
    let scene = Scene::new(Model::BrownConrady, 4, 0.2);
    let candidates = candidates(&scene.export.params.camera);
    assert_eq!(candidates.len(), 9 * 5);
    let ranked = rank_candidates(&scene.export, &candidates, &board(), IMAGE).unwrap();
    assert_eq!(ranked.len(), candidates.len());
    for pair in ranked.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        assert!(
            a.max_projection_sigma_px < b.max_projection_sigma_px
                || (a.max_projection_sigma_px == b.max_projection_sigma_px && a.index < b.index)
        );
    }
    for r in &ranked {
        assert!(r.projection_sigma_reduction_px >= -1e-9, "{r:?}");
        assert!(r.log_det_reduction >= -1e-9, "{r:?}");
    }
    // The best view reaches a corner the base views left to extrapolation.
    assert!(ranked[0].index / 5 != 4, "{:?}", ranked[0]);
    let again = rank_candidates(&scene.export, &candidates, &board(), IMAGE).unwrap();
    assert_eq!(ranked, again);
}

#[test]
fn candidate_poses_put_the_target_on_each_region() {
    let camera = truth(Model::BrownConrady);
    let poses = candidate_poses(
        &camera,
        &board(),
        IMAGE,
        &CandidateGrid::new(0.5, 0.0, [2, 1]),
    )
    .unwrap();
    assert_eq!(poses.len(), 2);
    let model = camera.build().unwrap();
    let centroid = Pt3::new(0.12, 0.09, 0.0);
    for (pose, expected) in poses.iter().zip([[320.0, 480.0], [960.0, 480.0]]) {
        let c = pose.transform_point(&centroid);
        assert!((c.coords.norm() - 0.5).abs() < 1e-12);
        let q = model.project_point_c(&c.coords).unwrap();
        assert!((q.x - expected[0]).abs() < 1e-6 && (q.y - expected[1]).abs() < 1e-6);
    }
    let bad = CandidateGrid::new(0.5, 2.0, [2, 1]);
    assert!(candidate_poses(&camera, &board(), IMAGE, &bad).is_err());
    let bad = CandidateGrid::new(0.0, 0.3, [2, 1]);
    assert!(candidate_poses(&camera, &board(), IMAGE, &bad).is_err());
    let bad = CandidateGrid::new(0.5, 0.3, [0, 1]);
    assert!(candidate_poses(&camera, &board(), IMAGE, &bad).is_err());
}

#[test]
fn inputs_are_checked() {
    let scene = Scene::new(Model::BrownConrady, 5, 0.2);
    let pose = candidates(&scene.export.params.camera)[0];
    let err = predict_with_views(&scene.export, &[], &board(), IMAGE).unwrap_err();
    assert!(
        matches!(err, Error::InsufficientData { need: 1, got: 0 }),
        "{err}"
    );
    let err = rank_candidates(&scene.export, &[pose], &[], IMAGE).unwrap_err();
    assert!(matches!(err, Error::InsufficientData { .. }), "{err}");
    let err = predict_with_views(&scene.export, &[pose], &board(), [1, 960]).unwrap_err();
    assert!(matches!(err, Error::InvalidInput { .. }), "{err}");

    let mut unavailable = scene.export.clone();
    unavailable.uncertainty = Uncertainty::Unavailable {
        reason: UncertaintyUnavailable::NotConverged,
    };
    let err = predict_with_views(&unavailable, &[pose], &board(), IMAGE).unwrap_err();
    assert!(
        matches!(
            err,
            Error::CovarianceUnavailable {
                reason: UncertaintyUnavailable::NotConverged
            }
        ),
        "{err}"
    );

    let mut stripped = scene.export.clone();
    stripped.diagnostics.per_feature_residuals.target.clear();
    let err = predict_with_views(&stripped, &[pose], &board(), IMAGE).unwrap_err();
    assert!(matches!(err, Error::InvalidInput { .. }), "{err}");
}

#[test]
fn results_round_trip_through_json() {
    let scene = Scene::new(Model::BrownConrady, 6, 0.2);
    let candidates = candidates(&scene.export.params.camera);
    let predicted = predict_with_views(&scene.export, &candidates[..2], &board(), IMAGE).unwrap();
    let back: PredictedUncertainty =
        serde_json::from_value(serde_json::to_value(&predicted).unwrap()).unwrap();
    assert_eq!(back, predicted);
    let ranked = rank_candidates(&scene.export, &candidates[..3], &board(), IMAGE).unwrap();
    let back: Vec<RankedCandidate> =
        serde_json::from_value(serde_json::to_value(&ranked).unwrap()).unwrap();
    assert_eq!(back, ranked);
    let grid = CandidateGrid::new(0.5, 0.3, [3, 2]);
    let json = serde_json::to_value(grid).unwrap();
    assert_eq!(json["regions"], serde_json::json!([3, 2]));
    assert_eq!(serde_json::from_value::<CandidateGrid>(json).unwrap(), grid);
}

// ─────────────────────────────────────────────────────────────────────────────
// The gate: prediction against re-solves over models, seeds and candidates
// ─────────────────────────────────────────────────────────────────────────────

/// Over Brown-Conrady, the division model and a Scheimpflug camera, five seeds each, at
/// 0.1–0.3 px noise: for the best-, median- and worst-ranked candidates that
/// see most of the board, and for the best three together, the predicted σ
/// of every parameter and the predicted maximum projection σ against the
/// re-solve's at the current pixel σ, within 10 % on average; the best
/// candidate's real reduction above the worst one's in every scene.
#[test]
#[ignore = "re-solves a few dozen calibrations; run in release with --ignored"]
fn prediction_matches_resolves_across_models() {
    let models = [Model::BrownConrady, Model::Division, Model::Scheimpflug];
    let sigmas = [0.1, 0.2, 0.3, 0.2, 0.1];
    println!(
        "| model | seed | noise px | candidate | mean abs dev σ | σ ratio min–max | max proj σ ratio | re-solve σ̂ / current | current max σ px | real reduction px |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|");
    for model in models {
        let mut deviations = Vec::new();
        let mut projection = Vec::new();
        for (seed, &sigma) in sigmas.iter().enumerate() {
            let mut scene = Scene::new(model, 100 + seed as u64, sigma);
            let candidates = candidates(&scene.export.params.camera);
            let ranked = rank_candidates(&scene.export, &candidates, &board(), IMAGE).unwrap();
            // Candidates that see most of the board: a re-solve would
            // accept them, and the ranking is between real views.
            let full: Vec<&RankedCandidate> =
                ranked.iter().filter(|r| r.visible_points >= 40).collect();
            let picks = [
                ("best", vec![full[0].index]),
                ("median", vec![full[full.len() / 2].index]),
                ("worst", vec![full[full.len() - 1].index]),
                (
                    "best three",
                    vec![full[0].index, full[1].index, full[2].index],
                ),
            ];
            let mut reductions = Vec::new();
            for (name, indices) in picks {
                let poses: Vec<Iso3> = indices.iter().map(|&i| candidates[i]).collect();
                let a = agreement(&mut scene, &poses);
                let deviation = mean_abs_deviation(&a.sigma_ratio);
                let (lo, hi) = a
                    .sigma_ratio
                    .iter()
                    .fold((f64::INFINITY, 0.0_f64), |(lo, hi), r| {
                        (lo.min(*r), hi.max(*r))
                    });
                println!(
                    "| {model:?} | {seed} | {sigma} | {name} {indices:?} | {:.1} % | {lo:.3}–{hi:.3} | {:.3} | {:.3} | {:.3} | {:.3} |",
                    100.0 * deviation,
                    a.projection_ratio,
                    a.noise_ratio,
                    a.current_max,
                    a.real_reduction
                );
                deviations.push(deviation);
                projection.push(a.projection_ratio);
                reductions.push(a.real_reduction);
            }
            assert!(
                reductions[0] > reductions[2],
                "{model:?} seed {seed}: best reduces {:.3} px, worst {:.3} px",
                reductions[0],
                reductions[2]
            );
        }
        let mean_deviation = deviations.iter().sum::<f64>() / deviations.len() as f64;
        let projection_deviation = mean_abs_deviation(&projection);
        println!(
            "{model:?}: mean abs deviation of σ {:.1} %, of max projection σ {:.1} %",
            100.0 * mean_deviation,
            100.0 * projection_deviation
        );
        assert!(mean_deviation < 0.1, "{model:?}: {mean_deviation}");
        assert!(
            projection_deviation < 0.1,
            "{model:?}: {projection_deviation}"
        );
    }
}
