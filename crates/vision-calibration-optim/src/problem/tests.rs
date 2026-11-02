//! Tests of the public problem builder.

use nalgebra::{DMatrix, DVector, RealField, Vector3};
use vision_calibration_core::synthetic::planar::grid_points;
use vision_calibration_core::synthetic::poses::{seeded_tilts, tilted_board_poses};
use vision_calibration_core::{
    BrownConrady5, CameraFix, CameraParams, CorrespondenceView, DistortionParams, FxFyCxCy, Iso3,
    PlanarDataset, ProjectionParams, Pt2, Pt3, ResidualKind, SensorParams, Uncertainty,
    UncertaintyUnavailable, View,
};

use super::*;
use crate::problems::planar_intrinsics::{
    PlanarIntrinsicsParams, PlanarIntrinsicsSolveOptions, optimize_planar_intrinsics,
};
use crate::projection::projection_uncertainty;

/// A deterministic standard normal draw for `key`.
fn gauss(key: u64) -> f64 {
    let mix = |mut z: u64| {
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };
    let uniform = |z: u64| ((mix(z) >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
    let (u1, u2) = (uniform(2 * key), uniform(2 * key + 1));
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// `y = a·x + b` at one `x`.
struct LinePoint {
    x: f64,
    y: f64,
}

impl Residual for LinePoint {
    fn dim(&self) -> usize {
        1
    }
    fn block_dims(&self) -> Vec<usize> {
        vec![2]
    }
    fn eval<T: RealField>(&self, blocks: &[DVector<T>]) -> DVector<T> {
        let (a, b) = (blocks[0][0].clone(), blocks[0][1].clone());
        let x = T::from_f64(self.x).unwrap();
        DVector::from_element(1, a * x + b - T::from_f64(self.y).unwrap())
    }
}

/// `(XᵀX)⁻¹` of the line fit at `xs`, with the residual sum of squares.
fn line_fit(xs: &[f64], ys: &[f64]) -> (DMatrix<f64>, [f64; 2], f64) {
    let x = DMatrix::from_fn(xs.len(), 2, |i, j| if j == 0 { xs[i] } else { 1.0 });
    let y = DVector::from_column_slice(ys);
    let inverse = (x.transpose() * &x).try_inverse().unwrap();
    let beta = &inverse * x.transpose() * &y;
    let rss = (&x * &beta - y).norm_squared();
    (inverse, [beta[0], beta[1]], rss)
}

fn line_problem(xs: &[f64], ys: &[f64], noise: Noise) -> (Problem, BlockId) {
    let mut problem = Problem::new();
    let line = problem
        .add_block("line", &[0.0, 0.0], Manifold::Euclidean)
        .unwrap();
    problem.set_labels(line, &["a", "b"]).unwrap();
    for (&x, &y) in xs.iter().zip(ys) {
        problem
            .add_residual(LinePoint { x, y }, &[line], RobustLoss::None, noise.clone())
            .unwrap();
    }
    (problem, line)
}

fn line_data() -> (Vec<f64>, Vec<f64>) {
    let xs: Vec<f64> = (0..12).map(|i| i as f64 * 0.5).collect();
    let ys = xs
        .iter()
        .enumerate()
        .map(|(i, x)| 2.0 * x + 1.0 + 0.05 * gauss(i as u64))
        .collect();
    (xs, ys)
}

fn covariance(solution: &Solution) -> &vision_calibration_core::ParameterCovariance {
    solution
        .uncertainty
        .covariance()
        .expect("covariance available")
}

fn assert_close(a: f64, b: f64, rel: f64, what: &str) {
    assert!(
        (a - b).abs() <= rel * b.abs().max(1e-300),
        "{what}: {a} vs {b}"
    );
}

#[test]
fn an_estimated_family_gives_the_classical_covariance() {
    let (xs, ys) = line_data();
    let (problem, line) = line_problem(&xs, &ys, Noise::Estimated("y".into()));
    let solution = solve(&problem, &SolveOptions::default()).unwrap();
    let (inverse, beta, rss) = line_fit(&xs, &ys);
    let fit = solution.value(line).unwrap();
    assert_close(fit[0], beta[0], 1e-8, "slope");
    assert_close(fit[1], beta[1], 1e-8, "intercept");

    // σ̂² = RSS / (m − n), Σ = σ̂² (XᵀX)⁻¹.
    let sigma2 = rss / (xs.len() - 2) as f64;
    let cov = covariance(&solution);
    assert_eq!(cov.labels, ["line.a", "line.b"]);
    for i in 0..2 {
        for j in 0..2 {
            assert_close(cov.covariance[i][j], sigma2 * inverse[(i, j)], 1e-6, "Σ");
        }
    }
    assert_eq!(cov.dof, xs.len() - 2);
    assert_eq!(solution.noise.len(), 1);
    assert_eq!(solution.noise[0].kind, ResidualKind::Custom("y".into()));
    assert_eq!(solution.noise[0].count, xs.len());
}

#[test]
fn a_known_sigma_counts_as_known_information() {
    let (xs, ys) = line_data();
    let sigma = 0.05;
    let (problem, _) = line_problem(&xs, &ys, Noise::Known(sigma));
    let solution = solve(&problem, &SolveOptions::default()).unwrap();
    let (inverse, _, _) = line_fit(&xs, &ys);
    let cov = covariance(&solution);
    for i in 0..2 {
        for j in 0..2 {
            assert_close(
                cov.covariance[i][j],
                sigma * sigma * inverse[(i, j)],
                1e-6,
                "Σ",
            );
        }
    }
    assert!(solution.noise.is_empty(), "no family is estimated");
}

#[test]
fn fixed_components_and_bounds_are_held() {
    let (xs, ys) = line_data();
    let (mut problem, line) = line_problem(&xs, &ys, Noise::Estimated("y".into()));
    problem.fix_components(line, &[1]).unwrap();
    let solution = solve(&problem, &SolveOptions::default()).unwrap();
    assert_eq!(solution.value(line).unwrap()[1], 0.0);
    let cov = covariance(&solution);
    assert_eq!(cov.labels, ["line.a"]);
    assert_eq!(cov.fixed, ["line.b"]);

    let (mut problem, line) = line_problem(&xs, &ys, Noise::Estimated("y".into()));
    problem.set_bounds(line, 0, -1.0, 1.5).unwrap();
    let solution = solve(&problem, &SolveOptions::default()).unwrap();
    assert!((solution.value(line).unwrap()[0] - 1.5).abs() < 1e-9);
    let Uncertainty::Unavailable {
        reason: UncertaintyUnavailable::AtBound { labels },
    } = &solution.uncertainty
    else {
        panic!("a solution on a bound has no covariance");
    };
    assert_eq!(labels, &["line.a"]);
}

#[test]
fn a_block_no_residual_reads_is_reported_unobservable() {
    let (xs, ys) = line_data();
    let (mut problem, _) = line_problem(&xs, &ys, Noise::Estimated("y".into()));
    let spare = problem
        .add_block("spare", &[3.0], Manifold::Euclidean)
        .unwrap();
    let solution = solve(&problem, &SolveOptions::default()).unwrap();
    assert_eq!(solution.value(spare).unwrap(), [3.0]);
    let Uncertainty::Unavailable {
        reason: UncertaintyUnavailable::RankDeficient { unobservable },
    } = &solution.uncertainty
    else {
        panic!("an unread block is not observable");
    };
    assert_eq!(unobservable, &["spare.0"]);
}

/// The distance from the origin of a unit vector scaled by a length: reads
/// an S² block and a Euclidean one.
struct Scaled {
    target: [f64; 3],
}

impl Residual for Scaled {
    fn dim(&self) -> usize {
        3
    }
    fn block_dims(&self) -> Vec<usize> {
        vec![3, 1]
    }
    fn eval<T: RealField>(&self, blocks: &[DVector<T>]) -> DVector<T> {
        let n = Vector3::new(
            blocks[0][0].clone(),
            blocks[0][1].clone(),
            blocks[0][2].clone(),
        );
        let p = n * blocks[1][0].clone();
        DVector::from_fn(3, |i, _| {
            p[i].clone() - T::from_f64(self.target[i]).unwrap()
        })
    }
}

#[test]
fn unit_vectors_stay_on_the_sphere() {
    let mut problem = Problem::new();
    let n = problem
        .add_block("normal", &[0.0, 0.0, 1.0], Manifold::UnitVector)
        .unwrap();
    let length = problem
        .add_block("length", &[1.0], Manifold::Euclidean)
        .unwrap();
    problem
        .add_residual(
            Scaled {
                target: [1.0, 2.0, 2.0],
            },
            &[n, length],
            RobustLoss::None,
            Noise::Known(1.0),
        )
        .unwrap();
    let solution = solve(&problem, &SolveOptions::default()).unwrap();
    let n = solution.value(n).unwrap();
    assert!((Vector3::new(n[0], n[1], n[2]).norm() - 1.0).abs() < 1e-12);
    // The default stopping rules end a zero-residual solve near, not at,
    // its minimum.
    assert!((n[0] - 1.0 / 3.0).abs() < 1e-5, "{n:?}");
    assert!((solution.value(length).unwrap()[0] - 3.0).abs() < 1e-4);
    let cov = covariance(&solution);
    assert_eq!(
        cov.labels,
        ["normal.nx", "normal.ny", "normal.nz", "length.0"]
    );
}

/// A residual that misreports its length once its block passes 1.
struct Misbehaving;

impl Residual for Misbehaving {
    fn dim(&self) -> usize {
        1
    }
    fn block_dims(&self) -> Vec<usize> {
        vec![1]
    }
    fn eval<T: RealField>(&self, blocks: &[DVector<T>]) -> DVector<T> {
        let x = blocks[0][0].clone();
        let len = if x.clone() > T::one() { 2 } else { 1 };
        DVector::from_element(len, x - T::from_f64(3.0).unwrap())
    }
}

#[test]
fn a_residual_of_the_wrong_length_is_a_rejected_step() {
    let mut problem = Problem::new();
    let x = problem.add_block("x", &[0.0], Manifold::Euclidean).unwrap();
    problem
        .add_residual(Misbehaving, &[x], RobustLoss::None, Noise::Known(1.0))
        .unwrap();
    let solution = solve(&problem, &SolveOptions::default()).unwrap();
    assert!(solution.value(x).unwrap()[0] <= 1.0);
}

#[test]
fn inputs_are_checked_as_they_are_added() {
    let mut problem = Problem::new();
    let err = |r: Result<BlockId, Error>| r.unwrap_err().to_string();
    assert!(err(problem.add_block("", &[1.0], Manifold::Euclidean)).contains("empty"));
    assert!(err(problem.add_block("cam0", &[1.0], Manifold::Euclidean)).contains("cameras"));
    assert!(err(problem.add_block("x", &[], Manifold::Euclidean)).contains("at least one"));
    assert!(err(problem.add_block("x", &[f64::NAN], Manifold::Euclidean)).contains("finite"));
    assert!(
        err(problem.add_block("q", &[0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0], Manifold::Se3))
            .contains("unit quaternion")
    );
    assert!(
        err(problem.add_block("n", &[0.0, 0.0, 2.0], Manifold::UnitVector)).contains("unit norm")
    );
    let x = problem
        .add_block("x", &[0.0, 0.0], Manifold::Euclidean)
        .unwrap();
    assert!(err(problem.add_block("x", &[1.0], Manifold::Euclidean)).contains("already"));
    let pose = problem.add_pose("pose", &Iso3::identity()).unwrap();

    let unit = || Noise::Known(1.0);
    let line = || LinePoint { x: 1.0, y: 2.0 };
    let add = |p: &mut Problem, blocks: &[BlockId], loss, noise| {
        p.add_residual(line(), blocks, loss, noise)
            .unwrap_err()
            .to_string()
    };
    assert!(add(&mut problem, &[x, x], RobustLoss::None, unit()).contains("declares 1 blocks"));
    assert!(add(&mut problem, &[pose], RobustLoss::None, unit()).contains("7 values"));
    let huber = RobustLoss::Huber { scale: -1.0 };
    assert!(add(&mut problem, &[x], huber, unit()).contains("positive"));
    assert!(add(&mut problem, &[x], RobustLoss::None, Noise::Known(0.0)).contains("known σ"));
    let empty = Noise::Estimated(String::new());
    assert!(add(&mut problem, &[x], RobustLoss::None, empty).contains("name"));
    struct Pair;
    impl Residual for Pair {
        fn dim(&self) -> usize {
            1
        }
        fn block_dims(&self) -> Vec<usize> {
            vec![2, 2]
        }
        fn eval<T: RealField>(&self, blocks: &[DVector<T>]) -> DVector<T> {
            DVector::from_element(1, blocks[0][0].clone() - blocks[1][0].clone())
        }
    }
    let message = problem
        .add_residual(Pair, &[x, x], RobustLoss::None, unit())
        .unwrap_err()
        .to_string();
    assert!(message.contains("given twice"), "{message}");
    let other = Problem::new()
        .add_block("y", &[0.0, 0.0], Manifold::Euclidean)
        .unwrap();
    assert!(add(&mut problem, &[other], RobustLoss::None, unit()).contains("another problem"));

    struct NotFinite;
    impl Residual for NotFinite {
        fn dim(&self) -> usize {
            1
        }
        fn block_dims(&self) -> Vec<usize> {
            vec![2]
        }
        fn eval<T: RealField>(&self, blocks: &[DVector<T>]) -> DVector<T> {
            DVector::from_element(1, blocks[0][0].clone() / blocks[0][1].clone())
        }
    }
    let message = problem
        .add_residual(NotFinite, &[x], RobustLoss::None, unit())
        .unwrap_err()
        .to_string();
    assert!(
        message.contains("residual 0") && message.contains("not finite"),
        "{message}"
    );

    assert!(problem.fix_components(pose, &[0]).is_err());
    assert!(problem.fix_components(x, &[2]).is_err());
    assert!(
        problem.set_bounds(x, 0, 1.0, 2.0).is_err(),
        "initial outside"
    );
    assert!(problem.set_bounds(x, 0, 1.0, -1.0).is_err());
    assert!(problem.set_labels(x, &["a"]).is_err());
    assert!(problem.set_labels(x, &["a", "a"]).is_err());
    assert!(problem.set_labels(x, &["a", "b.c"]).is_err());
    assert!(
        problem
            .set_labels(pose, &["a", "b", "c", "d", "e", "f"])
            .is_ok()
    );
}

#[test]
fn a_user_residual_on_factrs_is_a_typed_error() {
    let (xs, ys) = line_data();
    let (problem, _) = line_problem(&xs, &ys, Noise::Estimated("y".into()));
    let options = SolveOptions {
        backend: SolverBackend::Factrs,
        ..SolveOptions::default()
    };
    let err = solve(&problem, &options).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput { .. }) && err.to_string().contains("tiny-solver"),
        "{err}"
    );
}

/// Board corners, eight views of them and a camera with radial distortion.
struct Scene {
    camera: CameraParams,
    board: Vec<Pt3>,
    poses: Vec<Iso3>,
    pixels: Vec<Vec<Pt2>>,
}

fn scene(noise: f64) -> Scene {
    let camera = CameraParams {
        projection: ProjectionParams::Pinhole,
        distortion: DistortionParams::BrownConrady5 {
            params: BrownConrady5 {
                k1: -0.12,
                k2: 0.03,
                k3: 0.0,
                p1: 0.0005,
                p2: -0.0004,
                iters: 8,
            },
        },
        sensor: SensorParams::Identity,
        intrinsics: FxFyCxCy {
            fx: 820.0,
            fy: 810.0,
            cx: 640.0,
            cy: 360.0,
        },
    };
    let board = grid_points(9, 6, 0.03);
    let poses = tilted_board_poses(&seeded_tilts(8, 0.4, 7), (0.12, 0.075), 0.45, 0.02);
    let model = camera.build().unwrap();
    let pixels = poses
        .iter()
        .enumerate()
        .map(|(v, pose)| {
            board
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let q = model.project_point(&pose.transform_point(p)).unwrap();
                    let key = 1000 * v as u64 + i as u64;
                    Pt2::new(
                        q.x + noise * gauss(2 * key),
                        q.y + noise * gauss(2 * key + 1),
                    )
                })
                .collect()
        })
        .collect();
    Scene {
        camera,
        board,
        poses,
        pixels,
    }
}

/// The scene's camera moved off its true values, as a start.
fn perturbed(camera: &CameraParams) -> CameraParams {
    let mut start = camera.clone();
    start.intrinsics.fx *= 1.03;
    start.intrinsics.fy *= 0.98;
    start.intrinsics.cx += 6.0;
    start
}

#[test]
fn a_planar_problem_matches_the_library_planar_solve() {
    let scene = scene(0.3);
    let start = perturbed(&scene.camera);

    let mut problem = Problem::new();
    let camera = problem.add_camera(&start, &CameraFix::default()).unwrap();
    for (v, (pose, pixels)) in scene.poses.iter().zip(&scene.pixels).enumerate() {
        let pose = problem.add_pose(format!("pose{v}"), pose).unwrap();
        for (point, pixel) in scene.board.iter().zip(pixels) {
            problem
                .add_reprojection(camera, pose, *point, *pixel, RobustLoss::None)
                .unwrap();
        }
    }
    let solution = solve(&problem, &SolveOptions::default()).unwrap();
    let ours = solution.camera(camera).unwrap();

    let dataset = PlanarDataset::new(
        scene
            .pixels
            .iter()
            .map(|pixels| {
                View::without_meta(
                    CorrespondenceView::new(scene.board.clone(), pixels.clone()).unwrap(),
                )
            })
            .collect(),
    )
    .unwrap();
    let library = optimize_planar_intrinsics(
        &dataset,
        &PlanarIntrinsicsParams::new(start.clone(), scene.poses.clone()).unwrap(),
        PlanarIntrinsicsSolveOptions::default(),
        BackendSolveOptions::default(),
    )
    .unwrap();

    let (a, b) = (ours.blocks(), library.params.camera.blocks());
    for (x, y) in a.iter().flatten().zip(b.iter().flatten()) {
        assert!((x - y).abs() <= 1e-6 * y.abs().max(1.0), "{a:?} vs {b:?}");
    }
    let theirs = library.uncertainty.covariance().unwrap();
    assert!(
        theirs.view_scatter.is_none(),
        "the scene fits its noise model"
    );
    let ours_cov = covariance(&solution);
    // The camera's parameters lead both covariances, with the same labels.
    let n = theirs.labels.len();
    assert_eq!(ours_cov.labels[..n], theirs.labels[..]);
    for i in 0..n {
        for j in 0..n {
            assert_close(
                ours_cov.covariance[i][j],
                theirs.covariance[i][j],
                1e-5,
                "Σ",
            );
        }
    }
    assert_eq!(solution.noise[0].kind, ResidualKind::Reprojection);

    // The covariance is one the analysis functions read.
    // At the image centre σ is about that of the principal point.
    let map = projection_uncertainty(&ours, ours_cov, 0, [1280, 720]).unwrap();
    let sigma_cx = ours_cov.covariance[2][2].sqrt();
    assert!(
        (map.center / sigma_cx - 1.0).abs() < 0.5,
        "{} vs σ(cx) {sigma_cx}",
        map.center
    );
}

/// The distance between the camera centre and the target origin, from a
/// `camera_se3_target` pose: `‖t‖`.
struct Range {
    measured: f64,
}

impl Residual for Range {
    fn dim(&self) -> usize {
        1
    }
    fn block_dims(&self) -> Vec<usize> {
        vec![7]
    }
    fn eval<T: RealField>(&self, blocks: &[DVector<T>]) -> DVector<T> {
        let pose = se3(&blocks[0]);
        let range = pose.translation.vector.norm();
        DVector::from_element(1, range - T::from_f64(self.measured).unwrap())
    }
}

#[test]
fn a_user_residual_joins_the_reprojection_residuals() {
    let scene = scene(0.3);
    let start = perturbed(&scene.camera);
    let truth = scene.poses[0].translation.vector.norm();

    let build = |with_range: bool| {
        let mut problem = Problem::new();
        let camera = problem.add_camera(&start, &CameraFix::default()).unwrap();
        let mut first = None;
        for (v, (pose, pixels)) in scene.poses.iter().zip(&scene.pixels).enumerate() {
            let pose = problem.add_pose(format!("pose{v}"), pose).unwrap();
            first.get_or_insert(pose);
            for (point, pixel) in scene.board.iter().zip(pixels) {
                problem
                    .add_reprojection(camera, pose, *point, *pixel, RobustLoss::None)
                    .unwrap();
            }
        }
        let first = first.unwrap();
        if with_range {
            problem
                .add_residual(
                    Range { measured: truth },
                    &[first],
                    RobustLoss::None,
                    Noise::Known(1e-5),
                )
                .unwrap();
        }
        (problem, first)
    };
    let tz = |solution: &Solution| {
        let cov = covariance(solution);
        let i = cov.labels.iter().position(|l| l == "pose0.tz").unwrap();
        cov.covariance[i][i].sqrt()
    };

    let (without, _) = build(false);
    let (with, first) = build(true);
    let without = solve(&without, &SolveOptions::default()).unwrap();
    let with = solve(&with, &SolveOptions::default()).unwrap();
    let range = with.pose(first).unwrap().translation.vector.norm();
    assert!((range - truth).abs() < 3e-5, "{range} vs {truth}");
    // A 10 µm range pins the board's depth far below what the pixels do.
    assert!(
        tz(&with) < 0.5 * tz(&without),
        "{} vs {}",
        tz(&with),
        tz(&without)
    );
    assert_eq!(
        with.noise.len(),
        1,
        "only the reprojection family is estimated"
    );

    let factrs = SolveOptions {
        backend: SolverBackend::Factrs,
        ..SolveOptions::default()
    };
    let (problem, _) = build(false);
    solve(&problem, &factrs).expect("the library's residuals run on factrs");
}

#[test]
fn several_cameras_are_labelled_by_index() {
    let scene = scene(0.0);
    let mut problem = Problem::new();
    let a = problem
        .add_camera(&scene.camera, &CameraFix::default())
        .unwrap();
    let b = problem
        .add_camera(&scene.camera, &CameraFix::all(scene.camera.spec().unwrap()))
        .unwrap();
    let pose = problem.add_pose("pose", &scene.poses[0]).unwrap();
    // A point behind a pinhole camera has no image.
    let behind = Pt3::new(0.0, 0.0, -2.0);
    assert!(
        problem
            .add_reprojection(b, pose, behind, Pt2::new(0.0, 0.0), RobustLoss::None)
            .is_err_and(|e| e.to_string().contains("does not project"))
    );
    for camera in [a, b] {
        for (point, pixel) in scene.board.iter().zip(&scene.pixels[0]) {
            problem
                .add_reprojection(camera, pose, *point, *pixel, RobustLoss::None)
                .unwrap();
        }
    }
    let labels = problem.labels();
    let names: Vec<&str> = labels.iter().map(|l| l.coordinates[0].as_str()).collect();
    assert_eq!(
        names,
        ["cam0.fx", "cam0.k1", "cam1.fx", "cam1.k1", "pose.rx"]
    );
    let other = Problem::new()
        .add_camera(&scene.camera, &CameraFix::default())
        .unwrap();
    assert!(
        problem
            .add_reprojection(
                other,
                pose,
                scene.board[0],
                scene.pixels[0][0],
                RobustLoss::None
            )
            .is_err()
    );
}
