//! A problem of your own, built from the facade alone, as a downstream
//! crate builds it: closed-form initial values, the library's reprojection
//! residual, a residual of your own with a known σ, and a covariance that
//! matches the scatter of the estimates over repeated noise.

use vision_calibration::core::{
    BrownConrady5, CameraFix, CameraParams, CorrespondenceView, FxFyCxCy, Iso3,
    ParameterCovariance, PlanarDataset, Pt2, Pt3, View, make_pinhole_camera, pinhole_camera_params,
};
use vision_calibration::init::{
    IterativeIntrinsicsOptions, estimate_intrinsics_iterative, estimate_planar_pose_from_h,
};
use vision_calibration::mvg::dlt_homography;
use vision_calibration::nalgebra::{DVector, RealField};
use vision_calibration::optim::RobustLoss;
use vision_calibration::optim::problem::{
    BlockId, CameraId, Noise, Problem, Residual, Solution, SolveOptions, se3, solve,
};
use vision_calibration::synthetic::{planar, poses};

/// A measured distance from the camera centre to the board's origin: the
/// length of the translation of `camera_se3_target`.
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
        let range = se3(&blocks[0]).translation.vector.norm();
        DVector::from_element(1, range - T::from_f64(self.measured).unwrap())
    }
}

const PIXEL_SIGMA: f64 = 0.3;
const RANGE_SIGMA: f64 = 2e-4;

/// A deterministic standard normal draw for `key`.
fn gauss(key: u64) -> f64 {
    let mix = |mut z: u64| {
        z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };
    let uniform = |z: u64| ((mix(z) >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
    let (u1, u2) = (uniform(2 * key), uniform(2 * key + 1));
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// The true camera, board and board poses.
struct Truth {
    camera: CameraParams,
    board: Vec<Pt3>,
    poses: Vec<Iso3>,
}

fn truth() -> Truth {
    let k = FxFyCxCy {
        fx: 820.0,
        fy: 810.0,
        cx: 640.0,
        cy: 360.0,
    };
    let distortion = BrownConrady5 {
        k1: -0.12,
        k2: 0.03,
        k3: 0.0,
        p1: 0.0,
        p2: 0.0,
        iters: 8,
    };
    Truth {
        camera: pinhole_camera_params(&make_pinhole_camera(k, distortion)),
        board: planar::grid_points(9, 6, 0.03),
        poses: poses::tilted_board_poses(
            &poses::seeded_tilts(8, 0.4, 7),
            (0.12, 0.075),
            0.45,
            0.02,
        ),
    }
}

/// Noisy pixels of every view and a noisy range of view 0, from `seed`.
fn observe(truth: &Truth, seed: u64) -> (Vec<Vec<Pt2>>, f64) {
    let model = truth.camera.build().unwrap();
    let key = |v: usize, i: usize, axis: u64| (seed << 32) + 2 * (1000 * v + i) as u64 + axis;
    let pixels = truth
        .poses
        .iter()
        .enumerate()
        .map(|(v, pose)| {
            truth
                .board
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let q = model.project_point(&pose.transform_point(p)).unwrap();
                    Pt2::new(
                        q.x + PIXEL_SIGMA * gauss(key(v, i, 0)),
                        q.y + PIXEL_SIGMA * gauss(key(v, i, 1)),
                    )
                })
                .collect()
        })
        .collect();
    let range = truth.poses[0].translation.vector.norm() + RANGE_SIGMA * gauss((seed << 32) - 1);
    (pixels, range)
}

/// A solve, with its camera and the pose block of each view.
struct Refined {
    solution: Solution,
    camera: CameraId,
    poses: Vec<BlockId>,
}

impl Refined {
    fn camera(&self) -> CameraParams {
        self.solution.camera(self.camera).unwrap()
    }

    fn pose(&self, view: usize) -> Iso3 {
        self.solution.pose(self.poses[view]).unwrap()
    }
}

/// The camera and a pose per view, refined from `camera` and `poses` with
/// the reprojection residuals and, on view 0, the range.
fn refine(
    board: &[Pt3],
    pixels: &[Vec<Pt2>],
    range: f64,
    camera: &CameraParams,
    poses: &[Iso3],
) -> Refined {
    let mut problem = Problem::new();
    let cam = problem.add_camera(camera, &CameraFix::default()).unwrap();
    let mut blocks = Vec::new();
    for (v, (pose, pixels)) in poses.iter().zip(pixels).enumerate() {
        let pose = problem.add_pose(format!("view{v}"), pose).unwrap();
        for (point, pixel) in board.iter().zip(pixels) {
            problem
                .add_reprojection(cam, pose, *point, *pixel, RobustLoss::None)
                .unwrap();
        }
        blocks.push(pose);
    }
    problem
        .add_residual(
            Range { measured: range },
            &[blocks[0]],
            RobustLoss::None,
            Noise::Known(RANGE_SIGMA),
        )
        .unwrap();
    Refined {
        solution: solve(&problem, &SolveOptions::default()).unwrap(),
        camera: cam,
        poses: blocks,
    }
}

#[test]
fn a_problem_starts_from_the_closed_form_initializers() -> anyhow::Result<()> {
    let truth = truth();
    let (pixels, range) = observe(&truth, 1);

    // Zhang's intrinsics with distortion, then each view's pose from the
    // homography of its pixels.
    let views = pixels
        .iter()
        .map(|p| {
            Ok(View::without_meta(CorrespondenceView::new(
                truth.board.clone(),
                p.clone(),
            )?))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let dataset = PlanarDataset::new(views)?;
    let start = estimate_intrinsics_iterative(&dataset, IterativeIntrinsicsOptions::default())?;
    let k = start.k.k_matrix();
    let board_2d: Vec<Pt2> = truth.board.iter().map(|p| Pt2::new(p.x, p.y)).collect();
    let poses = pixels
        .iter()
        .map(|p| {
            let h = dlt_homography(&board_2d, p)?;
            Ok(estimate_planar_pose_from_h(&k, &h)?)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let refined = refine(
        &truth.board,
        &pixels,
        range,
        &pinhole_camera_params(&start),
        &poses,
    );
    let camera = refined.camera();
    assert!(
        (camera.intrinsics.fx / 820.0 - 1.0).abs() < 0.01,
        "{camera:?}"
    );
    let solved = refined.pose(0).translation.vector.norm();
    assert!(
        (solved - range).abs() < 3.0 * RANGE_SIGMA,
        "{solved} vs {range}"
    );
    Ok(())
}

/// The stated variance of a pose's translation along the camera's z axis.
///
/// A pose's covariance is over its right-perturbation tangent: `T·Exp(δ)`
/// moves the translation by `R·δt`, a step in the board's frame. Rotated
/// into the camera frame, the depth's variance is `r₃ᵀ Σ_t r₃` with `r₃`
/// the third row of `R`.
fn depth_variance(cov: &ParameterCovariance, view: usize, pose: &Iso3) -> f64 {
    let at = |c: &str| {
        let label = format!("view{view}.{c}");
        cov.labels.iter().position(|l| *l == label).unwrap()
    };
    let t = [at("tx"), at("ty"), at("tz")];
    let r = pose.rotation.to_rotation_matrix();
    let row = [r[(2, 0)], r[(2, 1)], r[(2, 2)]];
    (0..3)
        .flat_map(|a| (0..3).map(move |b| (a, b)))
        .map(|(a, b)| row[a] * cov.covariance[t[a]][t[b]] * row[b])
        .sum()
}

#[test]
fn the_stated_sigma_matches_the_scatter_of_the_estimates() {
    let truth = truth();
    let runs = 120;
    let names = ["fx", "cx", "view 0 depth", "view 3 depth"];
    let mut estimates = vec![Vec::new(); names.len()];
    let mut variances = vec![0.0; names.len()];
    for seed in 0..runs {
        let (pixels, range) = observe(&truth, seed + 100);
        let refined = refine(&truth.board, &pixels, range, &truth.camera, &truth.poses);
        let cov = refined
            .solution
            .uncertainty
            .covariance()
            .expect("covariance");
        let label = |l: &str| cov.labels.iter().position(|x| x == l).unwrap();
        let camera = refined.camera();
        let (fx, cx) = (label("fx"), label("cx"));
        let stated = [
            cov.covariance[fx][fx],
            cov.covariance[cx][cx],
            depth_variance(cov, 0, &truth.poses[0]),
            depth_variance(cov, 3, &truth.poses[3]),
        ];
        let values = [
            camera.intrinsics.fx,
            camera.intrinsics.cx,
            refined.pose(0).translation.z,
            refined.pose(3).translation.z,
        ];
        for k in 0..names.len() {
            variances[k] += stated[k] / runs as f64;
            estimates[k].push(values[k]);
        }
    }
    for (k, name) in names.iter().enumerate() {
        let mean = estimates[k].iter().sum::<f64>() / runs as f64;
        let scatter = (estimates[k].iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / (runs - 1) as f64)
            .sqrt();
        let stated = variances[k].sqrt();
        // The sample standard deviation of 120 draws is within ±20 % of the
        // true one at about 3σ.
        assert!(
            (scatter / stated - 1.0).abs() < 0.2,
            "{name}: stated σ {stated:.3e}, scatter {scatter:.3e}"
        );
    }
}
