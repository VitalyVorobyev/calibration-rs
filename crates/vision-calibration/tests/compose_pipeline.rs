//! A pipeline composed from the facade's public pieces, as the book's
//! "Compose Your Own Pipeline" chapter builds it: closed-form initial
//! values, one problem holding a calibration and a measurement, and the
//! measurement's σ with the calibration's uncertainty in it.

mod common;

use common::{Truth, depth_variance, observe_board, project, scatter_over_stated, truth};
use vision_calibration::core::{
    CalibratedRig, CameraFix, CameraParams, CorrespondenceView, Iso3, PlanarDataset, Pt2, Pt3,
    View, pinhole_camera_params,
};
use vision_calibration::init::{
    IterativeIntrinsicsOptions, estimate_intrinsics_iterative, estimate_planar_pose_from_h, pnp,
};
use vision_calibration::mvg::dlt_homography;
use vision_calibration::nalgebra::{Translation3, UnitQuaternion, Vector3};
use vision_calibration::optim::RobustLoss;
use vision_calibration::optim::problem::{
    BlockId, CameraId, PixelNoise, PoseChain, Problem, Solution, SolveOptions, TargetPoint, Use,
    solve,
};

/// The view index the part's pixels are drawn under, apart from the board's.
const PART_VIEW: usize = 100;

/// Eight fiducials on a machined part, in its own frame: the corners of an
/// 80 × 60 × 30 mm block.
fn part() -> Vec<Pt3> {
    let mut points = Vec::new();
    for z in [0.0, 0.03] {
        for (x, y) in [(0.0, 0.0), (0.08, 0.0), (0.08, 0.06), (0.0, 0.06)] {
            points.push(Pt3::new(x, y, z));
        }
    }
    points
}

/// The part's true `camera_se3_part`: half a metre away, turned a little.
fn part_pose() -> Iso3 {
    Iso3::from_parts(
        Translation3::new(-0.03, -0.02, 0.5),
        UnitQuaternion::from_scaled_axis(Vector3::new(0.25, -0.3, 0.1)),
    )
}

/// The blocks of a solved problem: the camera, the board's pose per
/// calibration view and, when measured, the part's pose.
struct Solved {
    solution: Solution,
    camera: CameraId,
    part: Option<BlockId>,
}

impl Solved {
    fn camera(&self) -> CameraParams {
        self.solution.camera(self.camera).unwrap()
    }

    fn part(&self) -> Iso3 {
        self.solution.pose(self.part.unwrap()).unwrap()
    }
}

/// The calibration views: the board, its pixels and initial pose per view.
#[derive(Clone, Copy)]
struct Calibration<'a> {
    board: &'a [Pt3],
    pixels: &'a [Vec<Pt2>],
    poses: &'a [Iso3],
}

/// The measurement: the part's fiducials, their pixels and its initial pose.
#[derive(Clone, Copy)]
struct Measurement<'a> {
    fiducials: &'a [Pt3],
    pixels: &'a [Pt2],
    pose: &'a Iso3,
}

/// A problem holding `camera` (with `fix`), the calibration views when
/// given, and the measurement when given.
fn build(
    camera: &CameraParams,
    fix: &CameraFix,
    calibration: Option<Calibration>,
    measurement: Option<Measurement>,
) -> Solved {
    let mut problem = Problem::new();
    let cam = problem.add_camera(camera, fix).unwrap();
    if let Some(c) = calibration {
        for (v, (pose, pixels)) in c.poses.iter().zip(c.pixels).enumerate() {
            let pose = problem.add_pose(format!("view{v}"), pose).unwrap();
            for (point, pixel) in c.board.iter().zip(pixels) {
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
        }
    }
    let part = measurement.map(|m| {
        let pose = problem.add_pose("part", m.pose).unwrap();
        for (point, pixel) in m.fiducials.iter().zip(m.pixels) {
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
        pose
    });
    Solved {
        solution: solve(&problem, &SolveOptions::default()).unwrap(),
        camera: cam,
        part,
    }
}

/// The calibration views as a planar dataset.
fn dataset(truth: &Truth, pixels: &[Vec<Pt2>]) -> anyhow::Result<PlanarDataset> {
    let views = pixels
        .iter()
        .map(|p| {
            Ok(View::without_meta(CorrespondenceView::new(
                truth.board.clone(),
                p.clone(),
            )?))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(PlanarDataset::new(views)?)
}

#[test]
fn a_measurement_starts_from_the_closed_form_initializers() -> anyhow::Result<()> {
    let truth = truth();
    let pixels = observe_board(&truth, 1);
    let part = part();
    let part_pixels = project(&truth.camera, &part_pose(), &part, 1, PART_VIEW);

    // The camera from Zhang's method with a distortion fit, each board pose
    // from its homography, and the part's pose by EPnP.
    let start = estimate_intrinsics_iterative(
        &dataset(&truth, &pixels)?,
        IterativeIntrinsicsOptions::default(),
    )?;
    let board_2d: Vec<Pt2> = truth.board.iter().map(|p| Pt2::new(p.x, p.y)).collect();
    let poses = pixels
        .iter()
        .map(|p| {
            let h = dlt_homography(&board_2d, p)?;
            Ok(estimate_planar_pose_from_h(&start.k.k_matrix(), &h)?)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let part_start = pnp::epnp(&part, &part_pixels, &start.k)?;

    let solved = build(
        &pinhole_camera_params(&start),
        &CameraFix::default(),
        Some(Calibration {
            board: &truth.board,
            pixels: &pixels,
            poses: &poses,
        }),
        Some(Measurement {
            fiducials: &part,
            pixels: &part_pixels,
            pose: &part_start,
        }),
    );
    let camera = solved.camera();
    assert!(
        (camera.intrinsics.fx / 820.0 - 1.0).abs() < 0.01,
        "{camera:?}"
    );
    let depth = solved.part().translation.z;
    assert!((depth - 0.5).abs() < 0.005, "depth {depth}");
    Ok(())
}

/// The part measured through a calibration added as consider parameters:
/// its pose and the variance of its depth.
fn measure_with(
    rig: &CalibratedRig,
    fiducials: &[Pt3],
    pixels: &[Pt2],
    start: &Iso3,
) -> (f64, f64) {
    let mut problem = Problem::new();
    let blocks = problem.add_calibrated_rig(rig, Use::Consider).unwrap();
    let pose = problem.add_pose("part", start).unwrap();
    for (point, pixel) in fiducials.iter().zip(pixels) {
        problem
            .add_reprojection(
                blocks.camera(0).unwrap(),
                PoseChain::Pose(pose),
                TargetPoint::Known(*point),
                *pixel,
                PixelNoise::Estimated,
                RobustLoss::None,
            )
            .unwrap();
    }
    let solution = solve(&problem, &SolveOptions::default()).unwrap();
    let cov = solution.uncertainty.covariance().unwrap();
    (
        solution.pose(pose).unwrap().translation.z,
        depth_variance(cov, "part", start),
    )
}

/// Over repeated noise in the calibration images and in the measurement,
/// the σ of the part's depth matches the scatter of the depths when the
/// calibration's uncertainty is in it: solved jointly, or measured with the
/// calibration as consider parameters. A camera held fixed at its
/// calibrated values understates it.
#[test]
fn the_joint_sigma_carries_the_calibration_and_a_fixed_camera_does_not() {
    let truth = truth();
    let part = part();
    let part_pose = part_pose();
    let spec = truth.camera.spec().unwrap();
    let runs = 120;
    let (mut joint, mut joint_var) = (Vec::new(), Vec::new());
    let (mut fixed, mut fixed_var) = (Vec::new(), Vec::new());
    let (mut consider, mut consider_var) = (Vec::new(), Vec::new());
    for seed in 0..runs {
        let seed = seed + 200;
        let pixels = observe_board(&truth, seed);
        let part_pixels = project(&truth.camera, &part_pose, &part, seed, PART_VIEW);
        let board = Some(Calibration {
            board: &truth.board,
            pixels: &pixels,
            poses: &truth.poses,
        });
        let measured = Some(Measurement {
            fiducials: &part,
            pixels: &part_pixels,
            pose: &part_pose,
        });

        // The calibration and the measurement in one problem.
        let solved = build(&truth.camera, &CameraFix::default(), board, measured);
        let cov = solved.solution.uncertainty.covariance().unwrap();
        joint.push(solved.part().translation.z);
        joint_var.push(depth_variance(cov, "part", &part_pose));

        // The calibration first, then the measurement with the camera fixed.
        let calibration = build(&truth.camera, &CameraFix::default(), board, None);
        let calibrated = calibration.camera();
        let solved = build(&calibrated, &CameraFix::all(spec), None, measured);
        let cov = solved.solution.uncertainty.covariance().unwrap();
        fixed.push(solved.part().translation.z);
        fixed_var.push(depth_variance(cov, "part", &part_pose));

        // The calibration first, then the measurement with its uncertainty
        // as consider parameters.
        let rig = CalibratedRig::single(calibrated, calibration.solution.uncertainty).unwrap();
        let (depth, var) = measure_with(&rig, &part, &part_pixels, &part_pose);
        consider.push(depth);
        consider_var.push(var);
    }
    let joint_ratio = scatter_over_stated(&joint, &joint_var);
    let fixed_ratio = scatter_over_stated(&fixed, &fixed_var);
    let consider_ratio = scatter_over_stated(&consider, &consider_var);
    let rms = |v: &[f64]| (v.iter().sum::<f64>() / v.len() as f64).sqrt() * 1e3;
    eprintln!(
        "depth σ: joint {:.3} mm (scatter/stated {joint_ratio:.2}), \
         consider {:.3} mm (scatter/stated {consider_ratio:.2}), \
         fixed camera {:.3} mm (scatter/stated {fixed_ratio:.2})",
        rms(&joint_var),
        rms(&consider_var),
        rms(&fixed_var),
    );
    // The sample standard deviation of 120 draws is within ±20 % of the true
    // one at about 3σ.
    assert!((joint_ratio - 1.0).abs() < 0.2, "joint {joint_ratio:.3}");
    assert!(
        (consider_ratio - 1.0).abs() < 0.2,
        "consider {consider_ratio:.3}"
    );
    // A consider measurement is the fixed camera's estimate.
    for (a, b) in consider.iter().zip(&fixed) {
        assert!((a - b).abs() < 1e-9, "{a} vs {b}");
    }
    assert!(fixed_ratio > 2.0, "fixed camera {fixed_ratio:.3}");
}
