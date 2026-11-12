//! Tests of robot pose planning for a rig hand-eye calibration: the
//! prediction against a re-solve with the views added, on synthetic
//! two-camera rigs in both modes, and the API's contracts.
//!
//! The gate that compares predictions with re-solves over both modes, with
//! and without robot pose corrections, several seeds and candidates solves a
//! few dozen calibrations, so it is ignored in the default (debug) run:
//! `cargo test --release -p vision-calibration-pipeline --lib next_view::rig -- --ignored --nocapture`.

use nalgebra::{Translation3, UnitQuaternion, Vector3};
use vision_calibration_core::synthetic::{planar, poses};
use vision_calibration_core::{
    BrownConrady5, CameraParams, CorrespondenceView, DistortionParams, FxFyCxCy, ProjectionParams,
    Pt2, RigDataset, RigView, RigViewObs, SensorParams,
};
use vision_calibration_optim::RobotPoseMeta;

use super::*;
use crate::common::config::{RobotPoseConfig, RobotPosePrior};
use crate::rig_handeye::{RigHandeyeConfig, RigHandeyeProblem, run_calibration};
use crate::session::CalibrationSession;

// ─────────────────────────────────────────────────────────────────────────────
// Fixtures
// ─────────────────────────────────────────────────────────────────────────────

const IMAGE: [u32; 2] = [1280, 960];
const SIZES: [[u32; 2]; 2] = [IMAGE, IMAGE];
/// The robot's true pose error, about and along each gripper axis.
const ROBOT_ROT: f64 = 2e-4;
const ROBOT_TRANS: f64 = 2e-4;

fn cameras() -> Vec<CameraParams> {
    let camera = |f: f64, k1: f64| CameraParams {
        projection: ProjectionParams::Pinhole,
        distortion: DistortionParams::BrownConrady5 {
            params: BrownConrady5 {
                k1,
                k2: 0.02,
                k3: 0.0,
                p1: 0.0005,
                p2: -0.0003,
                iters: 8,
            },
        },
        sensor: SensorParams::Identity,
        intrinsics: FxFyCxCy {
            fx: f,
            fy: f,
            cx: 640.0,
            cy: 480.0,
        },
    };
    vec![camera(1000.0, -0.08), camera(1100.0, -0.05)]
}

fn board() -> Vec<Pt3> {
    planar::grid_points(9, 7, 0.03)
}

/// The hand-eye and target pose of each mode: eye-in-hand `gripper_se3_rig`
/// and `base_se3_target`, eye-to-hand `rig_se3_base` and
/// `gripper_se3_target`.
fn truth(mode: HandeyeMode) -> (Iso3, Iso3) {
    match mode {
        HandeyeMode::EyeInHand => (
            poses::make_iso((0.10, -0.05, 0.02), (0.05, -0.03, 0.08)),
            poses::make_iso((0.05, -0.08, 0.0), (-0.10, -0.08, 1.0)),
        ),
        HandeyeMode::EyeToHand => (
            poses::make_iso((0.04, 0.03, -0.02), (0.06, 0.05, 1.0)),
            poses::make_iso((-0.05, 0.08, 0.1), (-0.10, -0.06, 0.02)),
        ),
    }
}

/// Seeded Gaussian noise (xorshift and Box–Muller).
struct Noise {
    state: u64,
}

impl Noise {
    fn new(seed: u64) -> Self {
        Self {
            state: 0x9E37_79B9_7F4A_7C15 ^ seed.wrapping_mul(0x2545_F491_4F6C_DD1D),
        }
    }

    fn uniform(&mut self) -> f64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        ((self.state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn sample(&mut self, sigma: f64) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        sigma * (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

/// A calibration from base robot poses, and the views and noise behind it.
struct Scene {
    mode: HandeyeMode,
    refine: bool,
    sigma_px: f64,
    noise: Noise,
    views: Vec<RigView<RobotPoseMeta>>,
    export: RigHandeyeExport,
}

impl Scene {
    fn new(mode: HandeyeMode, refine: bool, seed: u64, sigma_px: f64) -> Self {
        let mut scene = Self {
            mode,
            refine,
            sigma_px,
            noise: Noise::new(seed),
            views: Vec::new(),
            export: empty_export(),
        };
        for g in &stations()[..BASE] {
            let view = scene.observe(g);
            scene.views.push(view);
        }
        scene.export = scene.calibrate(scene.views.clone());
        scene
    }

    /// The view at the true robot pose `g`: the board points each camera
    /// sees in its image, with pixel noise, and the robot pose as the robot
    /// reports it (with its error when the solve corrects it).
    fn observe(&mut self, g: &Iso3) -> RigView<RobotPoseMeta> {
        let cameras = cameras();
        let layout = poses::rig_layout(2, 0.12, 0.05);
        let (h, t) = truth(self.mode);
        let rig_se3_target = match self.mode {
            HandeyeMode::EyeInHand => h.inverse() * g.inverse() * t,
            HandeyeMode::EyeToHand => h * g * t,
        };
        let [w, hgt] = IMAGE.map(f64::from);
        let observations = cameras
            .iter()
            .zip(&layout)
            .map(|(camera, cam_se3_rig)| {
                let model = camera.build().unwrap();
                let pose = cam_se3_rig * rig_se3_target;
                let (points_3d, points_2d): (Vec<Pt3>, Vec<Pt2>) = board()
                    .into_iter()
                    .filter_map(|p| {
                        let q = model.project_point_c(&pose.transform_point(&p).coords)?;
                        ((0.0..=w).contains(&q.x) && (0.0..=hgt).contains(&q.y)).then_some((p, q))
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
                    .map(|(p, q)| {
                        let noisy = Pt2::new(
                            q.x + self.noise.sample(self.sigma_px),
                            q.y + self.noise.sample(self.sigma_px),
                        );
                        (p, noisy)
                    })
                    .unzip();
                assert!(points_3d.len() >= 16, "the board is in view");
                Some(CorrespondenceView::new(points_3d, points_2d).unwrap())
            })
            .collect();
        let reported = if self.refine {
            let mut e = |s: f64| self.noise.sample(s);
            let error = Iso3::from_parts(
                Translation3::new(e(ROBOT_TRANS), e(ROBOT_TRANS), e(ROBOT_TRANS)),
                UnitQuaternion::from_scaled_axis(Vector3::new(
                    e(ROBOT_ROT),
                    e(ROBOT_ROT),
                    e(ROBOT_ROT),
                )),
            );
            g * error
        } else {
            *g
        };
        RigView {
            meta: RobotPoseMeta {
                base_se3_gripper: reported,
            },
            obs: RigViewObs {
                cameras: observations,
            },
        }
    }

    fn calibrate(&self, views: Vec<RigView<RobotPoseMeta>>) -> RigHandeyeExport {
        let mut config = RigHandeyeConfig::default();
        config.handeye_init.handeye_mode = self.mode;
        config.handeye_optimize.refine_cam_se3_rig = true;
        config.handeye_optimize.robot_poses = if self.refine {
            let mut prior = RobotPosePrior::new(ROBOT_ROT, ROBOT_TRANS);
            prior.estimate_sigma = false;
            RobotPoseConfig::Refine(prior)
        } else {
            RobotPoseConfig::Exact
        };
        let mut session = CalibrationSession::<RigHandeyeProblem>::new();
        session.set_config(config).unwrap();
        session
            .set_input(RigDataset::new(views, 2).unwrap())
            .unwrap();
        run_calibration(&mut session).unwrap();
        session.export().unwrap()
    }

    /// The calibration re-solved with views at the robot poses `added`.
    fn resolve(&mut self, added: &[Iso3]) -> RigHandeyeExport {
        let mut views = self.views.clone();
        for g in added {
            views.push(self.observe(g));
        }
        self.calibrate(views)
    }
}

/// Base views; the rest of [`stations`] are the candidates.
const BASE: usize = 6;

/// Diverse robot stations: the first are the base views, the rest the
/// candidates.
fn stations() -> Vec<Iso3> {
    poses::robot_stations(18, 5)
}

fn empty_export() -> RigHandeyeExport {
    serde_json::from_value(serde_json::json!({
        "kind": "rig_handeye",
        "cameras": [],
        "cam_se3_rig": [],
        "handeye": {
            "mode": "eye_in_hand",
            "gripper_se3_rig": Iso3::identity(),
            "base_se3_target": Iso3::identity(),
        },
        "robot_refinement": null,
        "mean_reproj_error": 0.0,
        "per_cam_reproj_errors": [],
    }))
    .unwrap()
}

static BOARD: std::sync::OnceLock<Vec<Pt3>> = std::sync::OnceLock::new();

fn planner(export: &RigHandeyeExport) -> RigPlanner<'_> {
    RigPlanner::new(export, BOARD.get_or_init(board), &SIZES).expect("planner")
}

/// The prediction for `candidates` against the re-solve with them, at the
/// current result's pixel σ (the re-solve's covariance rescaled from its own
/// σ̂, which fluctuates with the noise of the added views).
struct Agreement {
    labels: Vec<String>,
    /// Predicted σ / re-solved σ, per label.
    sigma_ratio: Vec<f64>,
    /// Predicted / re-solved `ln det` reduction.
    predicted_log_det_reduction: f64,
    real_log_det_reduction: f64,
}

fn agreement(scene: &mut Scene, candidates: &[Iso3]) -> Agreement {
    let predicted = predict_with_robot_poses(&scene.export, candidates, &board(), &SIZES).unwrap();
    let base = scene.export.clone();
    let current = planner(&base);
    let resolved = scene.resolve(candidates);
    let after = planner(&resolved);
    let scale = current.sigma2 / after.sigma2;
    let cov = &after.covariance;
    assert_eq!(cov.labels, predicted.covariance.labels);
    let n = cov.labels.len();
    let sigma_ratio = (0..n)
        .map(|i| (predicted.covariance.covariance[i][i] / (cov.covariance[i][i] * scale)).sqrt())
        .collect();
    let current_log_det = log_det(&current.covariance_matrix).unwrap();
    let predicted_matrix = DMatrix::from_fn(n, n, |i, j| predicted.covariance.covariance[i][j]);
    let resolved_matrix = &after.covariance_matrix * scale;
    Agreement {
        labels: cov.labels.clone(),
        sigma_ratio,
        predicted_log_det_reduction: current_log_det - log_det(&predicted_matrix).unwrap(),
        real_log_det_reduction: current_log_det - log_det(&resolved_matrix).unwrap(),
    }
}

fn mean_abs_deviation(ratios: &[f64]) -> f64 {
    ratios.iter().map(|r| (r - 1.0).abs()).sum::<f64>() / ratios.len() as f64
}

// ─────────────────────────────────────────────────────────────────────────────
// Scale and identity
// ─────────────────────────────────────────────────────────────────────────────

/// `Σ = σ² S_A⁻¹`: the rebuilt information is the one the covariance
/// inverted, so one scalar relates them. To the precision of the solve's
/// own inversion with exact robot poses; with corrected ones, up to the
/// solve's median-based weighting of the robot pose priors (0.050 and
/// 0.049 in correlation here). Compared in the
/// covariance's unit-diagonal frame, where every entry is a correlation.
#[test]
fn the_covariance_is_its_rebuilt_information_inverse_on_one_scale() {
    for (mode, refine, tolerance) in [
        (HandeyeMode::EyeInHand, false, 1e-6),
        (HandeyeMode::EyeToHand, false, 1e-6),
        (HandeyeMode::EyeInHand, true, 0.06),
        (HandeyeMode::EyeToHand, true, 0.06),
    ] {
        let scene = Scene::new(mode, refine, 1, 0.2);
        let planner = planner(&scene.export);
        let existing = planner.existing_information().unwrap();
        let rebuilt = invert_spd(&existing).unwrap() * planner.sigma2;
        let sigma = &planner.covariance_matrix;
        let n = sigma.nrows();
        let d: Vec<f64> = (0..n).map(|i| 1.0 / sigma[(i, i)].sqrt()).collect();
        let deviation = (0..n)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .map(|(i, j)| ((sigma[(i, j)] - rebuilt[(i, j)]) * d[i] * d[j]).abs())
            .fold(0.0, f64::max);
        assert!(
            deviation < tolerance,
            "{mode:?} refine={refine}: Σ − σ² S_A⁻¹ up to {deviation:e}"
        );
        // The implied pixel σ is the noise (0.2 px), scaled by the view
        // scatter where the result found one.
        let scatter = planner.covariance.view_scatter.unwrap_or(1.0).sqrt();
        let sigma_px = planner.sigma2.sqrt() / scatter;
        assert!((0.17..0.23).contains(&sigma_px), "σ = {sigma_px}");
    }
}

#[test]
fn a_pose_out_of_view_changes_nothing() {
    let scene = Scene::new(HandeyeMode::EyeInHand, true, 2, 0.2);
    // The gripper turned half a turn: the rig looks away from the target.
    let away = poses::make_iso((std::f64::consts::PI, 0.0, 0.0), (0.0, 0.0, 0.0));
    let predicted = predict_with_robot_poses(&scene.export, &[away], &board(), &SIZES).unwrap();
    assert_eq!(predicted.visible_points, [vec![0, 0]]);
    let current = scene.export.uncertainty.covariance().unwrap();
    let c = &current.covariance;
    let p = &predicted.covariance.covariance;
    for i in 0..c.len() {
        for j in 0..c.len() {
            // Relative to the entry's scale, `σᵢ σⱼ`: the covariance went
            // through two inversions.
            let scale = (c[i][i] * c[j][j]).sqrt();
            assert!((p[i][j] - c[i][j]).abs() <= 1e-7 * scale, "({i}, {j})");
        }
    }
    assert_eq!(predicted.covariance.dof, current.dof);
}

// ─────────────────────────────────────────────────────────────────────────────
// Prediction against a re-solve (fast)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn prediction_matches_a_resolve() {
    let mut scene = Scene::new(HandeyeMode::EyeInHand, true, 3, 0.2);
    let candidate = stations()[BASE + 2];
    let a = agreement(&mut scene, &[candidate]);
    let deviation = mean_abs_deviation(&a.sigma_ratio);
    assert!(
        deviation < 0.05,
        "σ ratios {:?} for {:?}",
        a.sigma_ratio,
        a.labels
    );
    assert!(a.real_log_det_reduction > 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Ranking and inputs
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn ranking_is_ordered_and_deterministic() {
    let scene = Scene::new(HandeyeMode::EyeToHand, true, 4, 0.2);
    let candidates = &stations()[BASE..];
    let ranked = rank_robot_poses(&scene.export, candidates, &board(), &SIZES).unwrap();
    assert_eq!(ranked.len(), candidates.len());
    for pair in ranked.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        assert!(
            a.log_det_reduction > b.log_det_reduction
                || (a.log_det_reduction == b.log_det_reduction && a.index < b.index)
        );
    }
    for r in &ranked {
        assert!(r.log_det_reduction > 0.0, "{r:?}");
        assert!(r.projection_sigma_reduction_px >= -1e-9, "{r:?}");
        assert_eq!(r.visible_points.len(), 2);
    }
    let again = rank_robot_poses(&scene.export, candidates, &board(), &SIZES).unwrap();
    assert_eq!(ranked, again);
}

#[test]
fn inputs_are_checked() {
    let scene = Scene::new(HandeyeMode::EyeInHand, false, 5, 0.2);
    let pose = stations()[BASE];
    let err = predict_with_robot_poses(&scene.export, &[], &board(), &SIZES).unwrap_err();
    assert!(
        matches!(err, Error::InsufficientData { need: 1, got: 0 }),
        "{err}"
    );
    let err = predict_with_robot_poses(&scene.export, &[pose], &[], &SIZES).unwrap_err();
    assert!(matches!(err, Error::InsufficientData { .. }), "{err}");
    let err = predict_with_robot_poses(&scene.export, &[pose], &board(), &[IMAGE]).unwrap_err();
    assert!(matches!(err, Error::InvalidInput { .. }), "{err}");
    let err =
        predict_with_robot_poses(&scene.export, &[pose], &board(), &[IMAGE, [1, 960]]).unwrap_err();
    assert!(matches!(err, Error::InvalidInput { .. }), "{err}");

    let mut no_covariance = scene.export.clone();
    no_covariance.uncertainty = Uncertainty::Unavailable {
        reason: vision_calibration_core::UncertaintyUnavailable::NotComputed,
    };
    let err = predict_with_robot_poses(&no_covariance, &[pose], &board(), &SIZES).unwrap_err();
    assert!(matches!(err, Error::CovarianceUnavailable { .. }), "{err}");
}

// ─────────────────────────────────────────────────────────────────────────────
// Gate (release, ignored by default)
// ─────────────────────────────────────────────────────────────────────────────

/// Predicted σ per parameter against the re-solve, for the best-, median-
/// and worst-ranked candidate and the best three together, over both modes,
/// with and without robot pose corrections, three seeds each: within 10 %
/// on average, and the best candidate's real `ln det` reduction above the
/// worst one's in every scene.
#[test]
#[ignore = "solves 96 calibrations; run in release"]
fn gate_prediction_against_resolves() {
    let mut deviations = Vec::new();
    let mut worst_case: f64 = 0.0;
    let mut ordered = 0;
    let mut scenes = 0;
    for mode in [HandeyeMode::EyeInHand, HandeyeMode::EyeToHand] {
        for refine in [false, true] {
            let mut per_case = vec![Vec::new(); 4];
            for seed in 10..13 {
                let mut scene = Scene::new(mode, refine, seed, 0.2);
                let candidates = &stations()[BASE..];
                let ranked = rank_robot_poses(&scene.export, candidates, &board(), &SIZES).unwrap();
                let pick = |k: usize| candidates[ranked[k].index];
                let cases: [Vec<Iso3>; 4] = [
                    vec![pick(0)],
                    vec![pick(ranked.len() / 2)],
                    vec![pick(ranked.len() - 1)],
                    (0..3).map(pick).collect(),
                ];
                let mut real = Vec::new();
                for (case, poses) in cases.iter().enumerate() {
                    let a = agreement(&mut scene, poses);
                    let d = mean_abs_deviation(&a.sigma_ratio);
                    worst_case = worst_case.max(
                        a.sigma_ratio
                            .iter()
                            .map(|r| (r - 1.0).abs())
                            .fold(0.0, f64::max),
                    );
                    per_case[case].push(d);
                    real.push(a.real_log_det_reduction);
                    eprintln!(
                        "{mode:?} refine={refine} seed {seed} case {case}: σ dev {:.2} %, \
                         ln det reduction predicted {:.3}, real {:.3}",
                        100.0 * d,
                        a.predicted_log_det_reduction,
                        a.real_log_det_reduction
                    );
                }
                scenes += 1;
                if real[0] > real[2] {
                    ordered += 1;
                }
            }
            let means: Vec<f64> = per_case
                .iter()
                .map(|d| d.iter().sum::<f64>() / d.len() as f64)
                .collect();
            eprintln!(
                "{mode:?} refine={refine}: mean σ deviation best {:.1} %, median {:.1} %, \
                 worst {:.1} %, best three {:.1} %",
                100.0 * means[0],
                100.0 * means[1],
                100.0 * means[2],
                100.0 * means[3]
            );
            deviations.extend(means);
        }
    }
    eprintln!(
        "largest single-parameter deviation {:.1} %; best above worst in {ordered}/{scenes}",
        100.0 * worst_case
    );
    for d in &deviations {
        assert!(*d < 0.10, "mean σ deviation {d}");
    }
    assert_eq!(ordered, scenes);
}
