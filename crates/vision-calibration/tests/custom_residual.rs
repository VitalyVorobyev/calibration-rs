//! A problem of your own, built from the facade alone, as a downstream
//! crate builds it: closed-form initial values, the library's reprojection
//! residual, a residual of your own with a known σ, and a covariance that
//! matches the scatter of the estimates over repeated noise.

mod common;

use common::{Truth, depth_variance, gauss, observe_board, scatter_over_stated, truth};
use vision_calibration::core::{
    CameraFix, CameraParams, CorrespondenceView, Iso3, PlanarDataset, Pt2, Pt3, View,
    pinhole_camera_params,
};
use vision_calibration::init::{
    IterativeIntrinsicsOptions, estimate_intrinsics_iterative, estimate_planar_pose_from_h,
};
use vision_calibration::mvg::solvers::dlt_homography;
use vision_calibration::nalgebra::{DVector, RealField};
use vision_calibration::optim::RobustLoss;
use vision_calibration::optim::problem::{
    BlockId, CameraId, Noise, PixelNoise, PoseChain, Problem, Residual, Solution, SolveOptions,
    TargetPoint, se3, solve,
};

/// A measured distance from the camera centre to the board's origin: the
/// length of the translation of `cam_se3_target`.
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

const RANGE_SIGMA: f64 = 2e-4;

/// Noisy pixels of every view and a noisy range of view 0, from `seed`.
fn observe(truth: &Truth, seed: u64) -> (Vec<Vec<Pt2>>, f64) {
    let range = truth.poses[0].translation.vector.norm() + RANGE_SIGMA * gauss((seed << 32) - 1);
    (observe_board(truth, seed), range)
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
                .add_reprojection(
                    cam,
                    PoseChain::Pose(pose),
                    TargetPoint::Known(*point),
                    *pixel,
                    PixelNoise::Estimated,
                    RobustLoss::None,
                )
                .unwrap();
        }
        blocks.push(pose);
    }
    problem
        .add_residual(
            Range { measured: range },
            &[blocks[0]],
            RobustLoss::None,
            Noise::Stated(RANGE_SIGMA),
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

#[test]
fn the_stated_sigma_matches_the_scatter_of_the_estimates() {
    let truth = truth();
    let runs = 120;
    let names = ["fx", "cx", "view 0 depth", "view 3 depth"];
    let mut estimates = vec![Vec::new(); names.len()];
    let mut variances = vec![Vec::new(); names.len()];
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
            depth_variance(cov, "view0", &truth.poses[0]),
            depth_variance(cov, "view3", &truth.poses[3]),
        ];
        let values = [
            camera.intrinsics.fx,
            camera.intrinsics.cx,
            refined.pose(0).translation.z,
            refined.pose(3).translation.z,
        ];
        for k in 0..names.len() {
            variances[k].push(stated[k]);
            estimates[k].push(values[k]);
        }
    }
    for (k, name) in names.iter().enumerate() {
        let ratio = scatter_over_stated(&estimates[k], &variances[k]);
        // The sample standard deviation of 120 draws is within ±20 % of the
        // true one at about 3σ.
        assert!(
            (ratio - 1.0).abs() < 0.2,
            "{name}: scatter / stated σ {ratio:.3}"
        );
    }
}
