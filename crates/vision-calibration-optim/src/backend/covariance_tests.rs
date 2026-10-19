//! The covariance against Monte-Carlo scatter, across backends, and on the
//! problems it must refuse.

use std::collections::HashMap;

use nalgebra::{DMatrix, DVector, UnitQuaternion, Vector3};
use vision_calibration_core::synthetic::noise::UniformPixelNoise;
use vision_calibration_core::synthetic::planar::{grid_points, project_views_noisy};
use vision_calibration_core::synthetic::poses::{centered_board_poses, seeded_board_pose_specs};
use vision_calibration_core::{
    BrownConrady5, FxFyCxCySkew, Iso3, PlanarDataset, View, make_pinhole_camera,
};

use crate::backend::covariance::{
    Coordinate, CovarianceEstimate, ParamUncertainty, ResidualFamily, UnavailableReason,
};
use crate::backend::{BackendSolveOptions, SolverBackend, solve};
use crate::ir::{
    Bound, CameraModelDesc, FactorKind, FixedMask, ManifoldKind, ProblemIR, ReprojChain,
    ResidualBlock, RobustLoss,
};
use crate::problems::planar_intrinsics::{
    PlanarIntrinsicsParams, PlanarIntrinsicsSolveOptions, build_planar_intrinsics_ir,
};

/// Ground truth of the planar scene: a moderate Brown–Conrady lens, an 8×6
/// board of 30 mm cells, eight tilted views at 0.6 m.
struct PlanarScene {
    camera: vision_calibration_core::PinholeCamera,
    points: Vec<vision_calibration_core::Pt3>,
    poses: Vec<Iso3>,
}

fn planar_scene() -> PlanarScene {
    let camera = make_pinhole_camera(
        FxFyCxCySkew {
            fx: 800.0,
            fy: 780.0,
            cx: 640.0,
            cy: 360.0,
            skew: 0.0,
        },
        BrownConrady5 {
            k1: -0.12,
            k2: 0.05,
            k3: 0.0,
            p1: 0.001,
            p2: -0.0005,
            iters: 8,
        },
    );
    let points = grid_points(8, 6, 0.03);
    let specs = seeded_board_pose_specs(8, 0.45, 17);
    let poses = centered_board_poses(&specs, (0.105, 0.075), 0.6);
    PlanarScene {
        camera,
        points,
        poses,
    }
}

/// Deterministic values in `[0, 1)`.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A standard normal sample (Box–Muller).
    fn gaussian(&mut self) -> f64 {
        let u = self.next().max(f64::MIN_POSITIVE);
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * self.next()).cos()
    }
}

/// Displace a fraction of the points by 10–30 px in a random direction.
fn add_outliers(ir: &mut ProblemIR, fraction: f64, seed: u64) {
    let mut rng = Lcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xABCD);
    for block in &mut ir.residuals {
        if let FactorKind::ReprojPoint { uv, .. } = &mut block.factor
            && rng.next() < fraction
        {
            let angle = rng.next() * std::f64::consts::TAU;
            let magnitude = 10.0 + 20.0 * rng.next();
            uv[0] += magnitude * angle.cos();
            uv[1] += magnitude * angle.sin();
        }
    }
}

/// The planar IR for one noise realization, started at the truth.
fn planar_ir(
    scene: &PlanarScene,
    noise: UniformPixelNoise,
    loss: RobustLoss,
) -> (ProblemIR, HashMap<String, DVector<f64>>) {
    let views = project_views_noisy(&scene.camera, &scene.points, &scene.poses, &noise).unwrap();
    let dataset = PlanarDataset::new(views.into_iter().map(View::without_meta).collect()).unwrap();
    let initial =
        PlanarIntrinsicsParams::from_pinhole(scene.camera.clone(), scene.poses.clone()).unwrap();
    let opts = PlanarIntrinsicsSolveOptions {
        robust_loss: loss,
        ..PlanarIntrinsicsSolveOptions::default()
    };
    build_planar_intrinsics_ir(&dataset, &initial, &opts).unwrap()
}

fn options(backend: SolverBackend) -> BackendSolveOptions {
    BackendSolveOptions {
        backend,
        max_iters: 50,
        ..BackendSolveOptions::default()
    }
}

fn available(uncertainty: ParamUncertainty) -> CovarianceEstimate {
    match uncertainty {
        ParamUncertainty::Available(estimate) => *estimate,
        ParamUncertainty::Unavailable(reason) => panic!("no covariance: {reason:?}"),
    }
}

/// Indices (in the covariance) of the coordinates of the blocks named.
fn coordinate_indices(ir: &ProblemIR, est: &CovarianceEstimate, names: &[&str]) -> Vec<usize> {
    est.coordinates
        .iter()
        .enumerate()
        .filter(|(_, c)| names.contains(&ir.params[c.block].name.as_str()))
        .map(|(i, _)| i)
        .collect()
}

/// Predicted against empirical σ of the camera coordinates over `runs`
/// noise realizations, and the mean NEES.
fn monte_carlo(loss: RobustLoss, outliers: f64, runs: u64) -> (Vec<f64>, f64, usize) {
    let scene = planar_scene();
    let truth = planar_ir(&scene, UniformPixelNoise::default(), RobustLoss::None).1;
    let mut errors: Vec<DVector<f64>> = Vec::new();
    let mut predicted_var: Option<DVector<f64>> = None;
    let mut nees_sum = 0.0;
    for seed in 0..runs {
        let noise = UniformPixelNoise {
            seed: 1000 + seed,
            max_abs_px: 0.5,
        };
        let (mut ir, initial) = planar_ir(&scene, noise, loss);
        add_outliers(&mut ir, outliers, seed);
        let solution = solve(&ir, &initial, &options(SolverBackend::TinySolver)).unwrap();
        let est = available(solution.uncertainty);
        let idx = coordinate_indices(&ir, &est, &["cam", "dist"]);
        let error = DVector::from_iterator(
            idx.len(),
            idx.iter().map(|&i| {
                let c = est.coordinates[i];
                let name = &ir.params[c.block].name;
                solution.params[name][c.index] - truth[name][c.index]
            }),
        );
        let cov = DMatrix::from_fn(idx.len(), idx.len(), |a, b| est.matrix[(idx[a], idx[b])]);
        nees_sum += (error.transpose() * cov.clone().try_inverse().unwrap() * &error)[(0, 0)];
        let var = cov.diagonal();
        predicted_var = Some(predicted_var.map_or(var.clone(), |acc| acc + var));
        errors.push(error);
    }
    let k = errors[0].len();
    let n = runs as f64;
    let predicted = (predicted_var.unwrap() / n).map(f64::sqrt);
    let mean = errors.iter().fold(DVector::zeros(k), |a, e| a + e) / n;
    let empirical = (errors
        .iter()
        .fold(DVector::zeros(k), |a, e| a + (e - &mean).map(|v| v * v))
        / (n - 1.0))
        .map(f64::sqrt);
    let ratios = (0..k).map(|i| predicted[i] / empirical[i]).collect();
    (ratios, nees_sum / n, k)
}

/// On clean data the camera coordinates' predicted σ match their
/// Monte-Carlo scatter within 10 %, and the NEES matches χ².
#[test]
fn planar_covariance_matches_monte_carlo() {
    const RUNS: u64 = 300;
    let (ratios, nees, k) = monte_carlo(RobustLoss::None, 0.0, RUNS);
    for (i, ratio) in ratios.iter().enumerate() {
        assert!(
            (ratio - 1.0).abs() < 0.10,
            "coordinate {i}: σ ratio {ratio:.3}"
        );
    }
    // NEES ~ χ²(k): its mean lies within three standard errors of k.
    let k = k as f64;
    let band = 3.0 * (2.0 * k / RUNS as f64).sqrt();
    assert!(
        (nees - k).abs() < band,
        "NEES mean {nees:.3}, expected {k} ± {band:.3}"
    );
}

/// With 5 % of the points displaced by 10–30 px and a robust loss, the
/// predicted σ stay within 20 % of the scatter, for a convex (Huber) and a
/// redescending (Cauchy) loss.
#[test]
fn planar_covariance_holds_under_outliers() {
    const RUNS: u64 = 200;
    for loss in [
        RobustLoss::Huber { scale: 1.0 },
        RobustLoss::Cauchy { scale: 1.0 },
    ] {
        let (ratios, nees, k) = monte_carlo(loss, 0.05, RUNS);
        for (i, ratio) in ratios.iter().enumerate() {
            assert!(
                (ratio - 1.0).abs() < 0.20,
                "{loss:?} coordinate {i}: σ ratio {ratio:.3}"
            );
        }
        let k = k as f64;
        let band = 3.0 * (2.0 * k / RUNS as f64).sqrt();
        assert!(
            (nees - k).abs() < band,
            "{loss:?}: NEES mean {nees:.3}, expected {k} ± {band:.3}"
        );
    }
}

/// Both backends report the same covariance and noise level, with and
/// without a robust loss. Solved to tight tolerances, so both stop at the
/// same minimum: under a robust loss their steps differ (Triggs vs IRLS),
/// and the default tolerances leave the optima 1e-5 apart.
#[test]
fn covariance_agrees_across_backends() {
    let scene = planar_scene();
    let noise = UniformPixelNoise {
        seed: 5,
        max_abs_px: 0.5,
    };
    let tight = |backend| BackendSolveOptions {
        min_abs_decrease: Some(0.0),
        min_rel_decrease: Some(1e-14),
        max_iters: 200,
        ..options(backend)
    };
    for loss in [RobustLoss::None, RobustLoss::Huber { scale: 0.3 }] {
        let (ir, initial) = planar_ir(&scene, noise, loss);
        let tiny = solve(&ir, &initial, &tight(SolverBackend::TinySolver)).unwrap();
        let factrs = solve(&ir, &initial, &tight(SolverBackend::Factrs)).unwrap();
        let (tiny, factrs) = (available(tiny.uncertainty), available(factrs.uncertainty));
        assert_eq!(tiny.coordinates, factrs.coordinates, "{loss:?}");
        let diff = (&tiny.matrix - &factrs.matrix).norm() / tiny.matrix.norm();
        assert!(diff < 1e-6, "{loss:?}: relative difference {diff:.2e}");
        let (st, sf) = (tiny.families[0].sigma, factrs.families[0].sigma);
        assert!((st / sf - 1.0).abs() < 1e-6, "{loss:?}: σ̂ {st} vs {sf}");
    }
}

/// `σ̂` of the reprojection family estimates Gaussian pixel noise, and
/// outliers do not inflate it.
#[test]
fn reprojection_sigma_estimates_the_pixel_noise() {
    let scene = planar_scene();
    let sigma = 0.3;
    let (mut ir, initial) = planar_ir(&scene, UniformPixelNoise::default(), RobustLoss::None);
    let mut rng = Lcg(42);
    for block in &mut ir.residuals {
        if let FactorKind::ReprojPoint { uv, .. } = &mut block.factor {
            uv[0] += sigma * rng.gaussian();
            uv[1] += sigma * rng.gaussian();
        }
    }
    for (loss, outliers) in [
        (RobustLoss::None, 0.0),
        (RobustLoss::Cauchy { scale: 1.0 }, 0.05),
    ] {
        let mut ir = ir.clone();
        for block in &mut ir.residuals {
            block.loss = loss;
        }
        add_outliers(&mut ir, outliers, 7);
        let solution = solve(&ir, &initial, &options(SolverBackend::TinySolver)).unwrap();
        let est = available(solution.uncertainty);
        assert_eq!(est.families.len(), 1);
        let family = &est.families[0];
        assert_eq!(family.family, ResidualFamily::Reprojection);
        assert_eq!(family.num_residuals, est.num_residuals);
        assert!(
            (family.sigma / sigma - 1.0).abs() < 0.10,
            "{loss:?}: σ̂ {} vs {sigma}",
            family.sigma
        );
    }
}

/// One camera seen through `cam_se3_rig · rig_se3_target` with both poses
/// free: only their product is observable, a six-dimensional gauge.
#[test]
fn free_gauge_is_reported_rank_deficient() {
    let scene = planar_scene();
    let truth = planar_ir(&scene, UniformPixelNoise::default(), RobustLoss::None).1;
    let mut ir = ProblemIR::new();
    let mut initial = HashMap::new();
    let cam = ir.add_param_block(
        "cam",
        4,
        ManifoldKind::Euclidean,
        FixedMask::all_fixed(4),
        None,
    );
    let dist = ir.add_param_block(
        "dist",
        5,
        ManifoldKind::Euclidean,
        FixedMask::all_fixed(5),
        None,
    );
    initial.insert("cam".to_string(), truth["cam"].clone());
    initial.insert("dist".to_string(), truth["dist"].clone());
    let extrinsics = ir.add_param_block(
        "extrinsics",
        7,
        ManifoldKind::SE3,
        FixedMask::all_free(),
        None,
    );
    initial.insert("extrinsics".to_string(), se3_vec(&Iso3::identity()));
    let views = project_views_noisy(
        &scene.camera,
        &scene.points,
        &scene.poses[..3],
        &UniformPixelNoise {
            seed: 3,
            max_abs_px: 0.3,
        },
    )
    .unwrap();
    for (v, view) in views.iter().enumerate() {
        let name = format!("rig_pose/{v}");
        let pose = ir.add_param_block(&name, 7, ManifoldKind::SE3, FixedMask::all_free(), None);
        initial.insert(name, se3_vec(&scene.poses[v]));
        for (pw, uv) in view.points_3d.iter().zip(&view.points_2d) {
            ir.add_residual_block(ResidualBlock {
                params: vec![cam, dist, extrinsics, pose],
                loss: RobustLoss::None,
                factor: FactorKind::ReprojPoint {
                    model: CameraModelDesc::PINHOLE4_DIST5,
                    chain: ReprojChain::TwoSe3,
                    pw: [pw.x, pw.y, pw.z],
                    uv: [uv.x, uv.y],
                    w: 1.0,
                },
                residual_dim: 2,
            });
        }
    }
    for backend in [SolverBackend::TinySolver, SolverBackend::Factrs] {
        let solution = solve(&ir, &initial, &options(backend)).unwrap();
        match solution.uncertainty {
            ParamUncertainty::Unavailable(UnavailableReason::RankDeficient { unobservable }) => {
                assert!(
                    unobservable.iter().any(|c| c.block == extrinsics.0),
                    "{backend:?}: the gauge involves the extrinsics: {unobservable:?}"
                );
            }
            other => panic!("{backend:?}: expected RankDeficient, got {other:?}"),
        }
    }
}

/// A coordinate held on a bound has no Gaussian covariance.
#[test]
fn coordinate_on_a_bound_is_reported() {
    let scene = planar_scene();
    let (mut ir, initial) = planar_ir(
        &scene,
        UniformPixelNoise {
            seed: 4,
            max_abs_px: 0.3,
        },
        RobustLoss::None,
    );
    // fx is 800: a lower bound above it holds it there.
    let cam = ir.params.iter().position(|p| p.name == "cam").unwrap();
    ir.params[cam].bounds = Some(vec![Bound {
        idx: 0,
        lower: 805.0,
        upper: 900.0,
    }]);
    let mut initial = initial;
    initial.get_mut("cam").unwrap()[0] = 810.0;
    for backend in [SolverBackend::TinySolver, SolverBackend::Factrs] {
        let solution = solve(&ir, &initial, &options(backend)).unwrap();
        match solution.uncertainty {
            ParamUncertainty::Unavailable(UnavailableReason::AtBound { coordinates }) => {
                assert_eq!(
                    coordinates,
                    [Coordinate {
                        block: cam,
                        index: 0
                    }]
                );
            }
            other => panic!("{backend:?}: expected AtBound, got {other:?}"),
        }
    }
}

/// A solve cut short by its iteration budget reports no covariance.
#[test]
fn iteration_budget_is_not_convergence() {
    let scene = planar_scene();
    let (ir, mut initial) = planar_ir(
        &scene,
        UniformPixelNoise {
            seed: 6,
            max_abs_px: 0.3,
        },
        RobustLoss::None,
    );
    initial.get_mut("cam").unwrap()[0] = 900.0;
    let opts = BackendSolveOptions {
        max_iters: 1,
        ..options(SolverBackend::TinySolver)
    };
    let solution = solve(&ir, &initial, &opts).unwrap();
    assert!(matches!(
        solution.uncertainty,
        ParamUncertainty::Unavailable(UnavailableReason::NotConverged)
    ));
}

/// With priors only, `Σ` is their inverse information, unscaled.
#[test]
fn prior_information_is_known() {
    let sqrt_info = [1.0, 2.0, 0.5, 4.0, 1.5, 0.25];
    let mut ir = ProblemIR::new();
    let delta = ir.add_param_block(
        "robot_delta",
        6,
        ManifoldKind::Euclidean,
        FixedMask::all_free(),
        None,
    );
    ir.add_residual_block(ResidualBlock {
        params: vec![delta],
        loss: RobustLoss::None,
        factor: FactorKind::Se3TangentPrior { sqrt_info },
        residual_dim: 6,
    });
    let initial = HashMap::from([(
        "robot_delta".to_string(),
        DVector::from_row_slice(&[0.01, -0.02, 0.03, 0.001, 0.002, -0.003]),
    )]);
    for backend in [SolverBackend::TinySolver, SolverBackend::Factrs] {
        let est = available(solve(&ir, &initial, &options(backend)).unwrap().uncertainty);
        assert!(est.families.is_empty());
        for (i, s) in sqrt_info.iter().enumerate() {
            let expected = 1.0 / (s * s);
            assert!(
                (est.matrix[(i, i)] / expected - 1.0).abs() < 1e-9,
                "{backend:?} [{i}]: {} vs {expected}",
                est.matrix[(i, i)]
            );
        }
    }
}

fn se3_vec(pose: &Iso3) -> DVector<f64> {
    let q: UnitQuaternion<f64> = pose.rotation;
    let t: Vector3<f64> = pose.translation.vector;
    DVector::from_row_slice(&[q.i, q.j, q.k, q.w, t.x, t.y, t.z])
}
