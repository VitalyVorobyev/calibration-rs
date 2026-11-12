//! Tests of measurements with a calibration: [`Use::Consider`] and
//! [`Use::Prior`], free points, rig chains and stated pixel noise.

use nalgebra::{DMatrix, Translation3, UnitQuaternion, Vector3};
use vision_calibration_core::{
    CalibratedRig, CameraFix, CameraParams, Iso3, ParameterCovariance, PointCovariance, Pt2, Pt3,
    Uncertainty, UncertaintyUnavailable,
};

use super::tests::{Scene, assert_close, gauss, scene};
use super::*;

/// Eight fiducials on a part: the corners of an 80 × 60 × 30 mm block.
fn part() -> Vec<Pt3> {
    let mut points = Vec::new();
    for z in [0.0, 0.03] {
        for (x, y) in [(0.0, 0.0), (0.08, 0.0), (0.08, 0.06), (0.0, 0.06)] {
            points.push(Pt3::new(x, y, z));
        }
    }
    points
}

/// The part's `cam_se3_part`, half a metre away.
fn part_pose() -> Iso3 {
    Iso3::from_parts(
        Translation3::new(-0.03, -0.02, 0.5),
        UnitQuaternion::from_scaled_axis(Vector3::new(0.25, -0.3, 0.1)),
    )
}

/// The part's pixels through the true camera, with noise σ `sigma`.
fn part_pixels(scene: &Scene, sigma: f64) -> Vec<Pt2> {
    let model = scene.camera.build().unwrap();
    part()
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let q = model
                .project_point(&part_pose().transform_point(p))
                .unwrap();
            let key = 900_000 + i as u64;
            Pt2::new(
                q.x + sigma * gauss(2 * key),
                q.y + sigma * gauss(2 * key + 1),
            )
        })
        .collect()
}

/// Add the board views of `scene` through `camera`, with pixel noise
/// `noise`.
fn add_board_with(problem: &mut Problem, camera: CameraId, scene: &Scene, noise: PixelNoise) {
    for (v, (pose, pixels)) in scene.poses.iter().zip(&scene.pixels).enumerate() {
        let pose = problem.add_pose(format!("view{v}"), pose).unwrap();
        for (point, pixel) in scene.board.iter().zip(pixels) {
            problem
                .add_reprojection(
                    camera,
                    PoseChain::Pose(pose),
                    TargetPoint::Known(*point),
                    *pixel,
                    noise,
                    RobustLoss::None,
                )
                .unwrap();
        }
    }
}

/// [`add_board_with`] with the noise estimated.
fn add_board(problem: &mut Problem, camera: CameraId, scene: &Scene) {
    add_board_with(problem, camera, scene, PixelNoise::Estimated);
}

/// The scene's pixel noise, 0.3 px, stated.
fn stated() -> PixelNoise {
    PixelNoise::Stated(PointCovariance::new(0.09, 0.0, 0.09).unwrap())
}

/// Add the part, seen through `camera` with pixel noise `noise`; returns
/// its pose block.
fn add_part_with(
    problem: &mut Problem,
    camera: CameraId,
    pixels: &[Pt2],
    noise: PixelNoise,
) -> BlockId {
    let pose = problem.add_pose("part", &part_pose()).unwrap();
    for (point, pixel) in part().iter().zip(pixels) {
        problem
            .add_reprojection(
                camera,
                PoseChain::Pose(pose),
                TargetPoint::Known(*point),
                *pixel,
                noise,
                RobustLoss::None,
            )
            .unwrap();
    }
    pose
}

/// [`add_part_with`] with the noise estimated.
fn add_part(problem: &mut Problem, camera: CameraId, pixels: &[Pt2]) -> BlockId {
    add_part_with(problem, camera, pixels, PixelNoise::Estimated)
}

/// The camera calibrated from the scene's board views with pixel noise
/// `noise`, with its uncertainty.
fn calibrate_with(scene: &Scene, noise: PixelNoise) -> CalibratedRig {
    let mut problem = Problem::new();
    let camera = problem
        .add_camera(&scene.camera, &CameraFix::default())
        .unwrap();
    add_board_with(&mut problem, camera, scene, noise);
    let solution = solve(&problem, &SolveOptions::default()).unwrap();
    CalibratedRig::single(solution.camera(camera).unwrap(), solution.uncertainty).unwrap()
}

/// [`calibrate_with`] with the noise estimated.
fn calibrate(scene: &Scene) -> CalibratedRig {
    calibrate_with(scene, PixelNoise::Estimated)
}

/// The covariance block of `labels` from `covariance`.
fn block(covariance: &ParameterCovariance, labels: &[String]) -> DMatrix<f64> {
    let at: Vec<usize> = labels
        .iter()
        .map(|l| covariance.index(l).unwrap_or_else(|| panic!("{l}")))
        .collect();
    DMatrix::from_fn(at.len(), at.len(), |a, b| {
        covariance.covariance[at[a]][at[b]]
    })
}

/// The tangent step from `from` to `to` on the SE3 manifold:
/// `[Log(R₀ᵀR), R₀ᵀ(t − t₀)]`.
fn se3_step(from: &Iso3, to: &Iso3) -> [f64; 6] {
    let w = (from.rotation.inverse() * to.rotation).scaled_axis();
    let v = from
        .rotation
        .inverse_transform_vector(&(to.translation.vector - from.translation.vector));
    [w.x, w.y, w.z, v.x, v.y, v.z]
}

/// `camera` with the parameter `label` moved by `step`.
fn moved(camera: &CameraParams, label: &str, step: f64) -> CameraParams {
    let mut c = camera.clone();
    let k = &mut c.intrinsics;
    match label {
        "fx" => k.fx += step,
        "fy" => k.fy += step,
        "cx" => k.cx += step,
        "cy" => k.cy += step,
        _ => {
            let DistortionParams::BrownConrady5 { params } = &mut c.distortion else {
                panic!("a Brown–Conrady camera");
            };
            match label {
                "k1" => params.k1 += step,
                "k2" => params.k2 += step,
                "k3" => params.k3 += step,
                "p1" => params.p1 += step,
                "p2" => params.p2 += step,
                _ => panic!("{label}"),
            }
        }
    }
    c
}

/// The part's pose and covariance measured through `camera` held fixed.
fn measure_fixed(camera: &CameraParams, pixels: &[Pt2]) -> (Iso3, Option<DMatrix<f64>>) {
    let mut problem = Problem::new();
    let spec = camera.spec().unwrap();
    let cam = problem.add_camera(camera, &CameraFix::all(spec)).unwrap();
    let part = add_part(&mut problem, cam, pixels);
    let solution = solve(&problem, &SolveOptions::default()).unwrap();
    let cov = solution
        .uncertainty
        .covariance()
        .map(|c| block(c, &uncertainty::pose_labels("part")));
    (solution.pose(part).unwrap(), cov)
}

/// With the calibration as consider parameters, the part's covariance is
/// the fixed camera's plus `S Σ_c Sᵀ`, with `S = ∂x̂/∂c` the central
/// difference of re-solved measurements; the estimate is the fixed
/// camera's.
#[test]
fn consider_adds_the_calibration_through_the_measurements_sensitivity() {
    let scene = scene(0.3);
    let rig = calibrate(&scene);
    let pixels = part_pixels(&scene, 0.3);

    let mut problem = Problem::new();
    let blocks = problem.add_calibrated_rig(&rig, Use::Consider).unwrap();
    let camera = blocks.camera(0).unwrap();
    let part = add_part(&mut problem, camera, &pixels);
    let solution = solve(&problem, &SolveOptions::default()).unwrap();
    let pose_labels = uncertainty::pose_labels("part");
    let consider = block(covariance_of(&solution), &pose_labels);
    assert_eq!(covariance_of(&solution).labels, pose_labels);

    let calibrated = &rig.cameras()[0];
    let (fixed_pose, fixed_cov) = measure_fixed(calibrated, &pixels);
    let step = se3_step(&fixed_pose, &solution.pose(part).unwrap());
    assert!(step.iter().all(|s| s.abs() < 1e-10), "{step:?}");

    let calibration = rig.uncertainty().covariance().unwrap();
    let camera_labels: Vec<String> = calibration
        .labels
        .iter()
        .filter(|l| !l.starts_with("view"))
        .cloned()
        .collect();
    let sigma_c = block(calibration, &camera_labels);
    let mut s = DMatrix::zeros(6, camera_labels.len());
    for (j, label) in camera_labels.iter().enumerate() {
        let h = 1e-2 * sigma_c[(j, j)].sqrt();
        let plus = measure_fixed(&moved(calibrated, label, h), &pixels).0;
        let minus = measure_fixed(&moved(calibrated, label, -h), &pixels).0;
        let d = se3_step(&minus, &plus);
        for i in 0..6 {
            s[(i, j)] = d[i] / (2.0 * h);
        }
    }
    let expected = fixed_cov.unwrap() + &s * sigma_c * s.transpose();
    for i in 0..6 {
        for j in 0..6 {
            let scale = (expected[(i, i)] * expected[(j, j)]).sqrt();
            assert!(
                (consider[(i, j)] - expected[(i, j)]).abs() < 1e-3 * scale,
                "Σ[{i},{j}]: {} vs {}",
                consider[(i, j)],
                expected[(i, j)]
            );
        }
    }
    // The calibration's share is most of the depth's variance.
    let fixed_only = measure_fixed(calibrated, &pixels).1.unwrap();
    assert!(consider[(5, 5)] > 2.0 * fixed_only[(5, 5)]);
}

fn covariance_of(solution: &Solution) -> &ParameterCovariance {
    solution.uncertainty.covariance().expect("a covariance")
}

/// With the calibration as a prior, the part's covariance is the joint
/// solve's: the prior holds the information the board views gave the
/// camera. Every pixel's noise is stated, so each solve weighs its
/// residuals by their information.
#[test]
fn a_prior_holds_what_the_calibration_images_gave() {
    let scene = scene(0.3);
    let rig = calibrate_with(&scene, stated());
    let pixels = part_pixels(&scene, 0.3);
    let pose_labels = uncertainty::pose_labels("part");

    let mut joint = Problem::new();
    let camera = joint
        .add_camera(&scene.camera, &CameraFix::default())
        .unwrap();
    add_board_with(&mut joint, camera, &scene, stated());
    let joint_part = add_part_with(&mut joint, camera, &pixels, stated());
    let joint = solve(&joint, &SolveOptions::default()).unwrap();

    let mut problem = Problem::new();
    let blocks = problem.add_calibrated_rig(&rig, Use::Prior).unwrap();
    let part = add_part_with(&mut problem, blocks.camera(0).unwrap(), &pixels, stated());
    let prior = solve(&problem, &SolveOptions::default()).unwrap();

    let a = block(covariance_of(&prior), &pose_labels);
    let b = block(covariance_of(&joint), &pose_labels);
    for i in 0..6 {
        assert_close(a[(i, i)].sqrt(), b[(i, i)].sqrt(), 0.03, &pose_labels[i]);
    }
    let step = se3_step(&joint.pose(joint_part).unwrap(), &prior.pose(part).unwrap());
    assert!(step[5].abs() < 0.1 * b[(5, 5)].sqrt(), "{step:?}");
    // The camera is estimated, labelled as in the calibration.
    // A residual of estimated noise beside the prior is refused.
    add_board(&mut problem, blocks.camera(0).unwrap(), &scene);
    assert!(
        solve(&problem, &SolveOptions::default())
            .is_err_and(|e| e.to_string().contains("noise is stated"))
    );
    let fx = covariance_of(&prior).std_dev("fx").unwrap();
    let calibrated = rig
        .uncertainty()
        .covariance()
        .unwrap()
        .std_dev("fx")
        .unwrap();
    assert!(
        fx < calibrated && fx > 0.9 * calibrated,
        "{fx} vs {calibrated}"
    );
}

#[test]
fn a_calibration_without_covariance_is_named_and_cannot_be_a_prior() {
    let scene = scene(0.3);
    let rig = CalibratedRig::single(scene.camera.clone(), Uncertainty::default()).unwrap();
    let mut problem = Problem::new();
    assert!(
        problem
            .add_calibrated_rig(&rig, Use::Prior)
            .is_err_and(|e| e.to_string().contains("cannot be a prior"))
    );
    let blocks = problem.add_calibrated_rig(&rig, Use::Consider).unwrap();
    assert!(problem.add_calibrated_rig(&rig, Use::Consider).is_err());
    add_part(
        &mut problem,
        blocks.camera(0).unwrap(),
        &part_pixels(&scene, 0.3),
    );
    let solution = solve(&problem, &SolveOptions::default()).unwrap();
    assert_eq!(
        solution.uncertainty,
        Uncertainty::Unavailable {
            reason: UncertaintyUnavailable::CalibrationWithoutCovariance
        }
    );
}

/// A two-camera rig: the scene's camera at the rig origin and a copy
/// 0.12 m to its right, turned in a little.
fn stereo(scene: &Scene, sigma_c: f64) -> CalibratedRig {
    let right = Iso3::from_parts(
        Translation3::new(-0.12, 0.0, 0.004),
        UnitQuaternion::from_scaled_axis(Vector3::new(0.0, -0.05, 0.01)),
    );
    let mut labels = Vec::new();
    for k in 0..2 {
        for p in ["fx", "fy", "cx", "cy", "k1", "k2"] {
            labels.push(format!("cam{k}.{p}"));
        }
    }
    labels.extend(uncertainty::pose_labels("cam1.cam_se3_rig"));
    let variance = |l: &str| {
        // Extrinsics and distortion 1e-3 (rad, m, unitless), the
        // intrinsics 0.5 px, times `sigma_c`.
        let dimensionless = l.contains("cam_se3_rig") || l.ends_with("k1") || l.ends_with("k2");
        let v = if dimensionless { 1e-6 } else { 0.25 };
        v * sigma_c * sigma_c
    };
    let n = labels.len();
    let covariance = ParameterCovariance {
        covariance: (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| if i == j { variance(&labels[i]) } else { 0.0 })
                    .collect()
            })
            .collect(),
        labels,
        views: Vec::new(),
        fixed: Vec::new(),
        noise: Vec::new(),
        num_residuals: 0,
        dof: 0,
        view_scatter: None,
        second_order: None,
    };
    CalibratedRig::new(
        vec![scene.camera.clone(), scene.camera.clone()],
        vec![Iso3::identity(), right],
        Uncertainty::Available(covariance),
    )
    .unwrap()
}

/// Points triangulated through a rig chain from their pixels in two
/// cameras, as free blocks; returns the solution and the point blocks.
fn triangulate(
    rig: &CalibratedRig,
    points: &[Pt3],
    noise: PixelNoise,
    backend: SolverBackend,
) -> (Solution, Vec<BlockId>) {
    let mut problem = Problem::new();
    let blocks = problem.add_calibrated_rig(rig, Use::Consider).unwrap();
    let rig_se3_world = problem.add_pose("capture", &Iso3::identity()).unwrap();
    problem.fix(rig_se3_world).unwrap();
    let mut ids = Vec::new();
    for (i, p) in points.iter().enumerate() {
        // A start off the truth, so the solve has something to do.
        let start = [p.x + 0.01, p.y - 0.01, p.z + 0.02];
        let id = problem
            .add_block(format!("p{i}"), &start, Manifold::Euclidean)
            .unwrap();
        for k in 0..2 {
            let model = rig.model(k).unwrap();
            let pixel = model.project_point(&(rig.cam_se3_rig()[k] * p)).unwrap();
            problem
                .add_reprojection(
                    blocks.camera(k).unwrap(),
                    blocks.chain(k, rig_se3_world).unwrap(),
                    TargetPoint::Block(id),
                    pixel,
                    noise,
                    RobustLoss::None,
                )
                .unwrap();
        }
        ids.push(id);
    }
    let options = SolveOptions {
        backend,
        ..SolveOptions::default()
    };
    (solve(&problem, &options).unwrap(), ids)
}

fn stereo_points() -> Vec<Pt3> {
    vec![
        Pt3::new(0.05, -0.02, 0.6),
        Pt3::new(-0.08, 0.04, 0.8),
        Pt3::new(0.0, 0.0, 1.0),
    ]
}

/// Free points through a rig chain come back where they are, on both
/// backends; with stated pixel noise their covariance scales with it, and
/// the calibration adds its share.
#[test]
fn free_points_triangulate_through_a_rig_with_stated_noise() {
    let scene = scene(0.0);
    let points = stereo_points();
    let stated = |sigma: f64| {
        PixelNoise::Stated(PointCovariance::new(sigma * sigma, 0.0, sigma * sigma).unwrap())
    };
    let variances = |solution: &Solution| -> Vec<f64> {
        let c = covariance_of(solution);
        (0..points.len())
            .flat_map(|i| ["0", "1", "2"].map(|a| c.std_dev(&format!("p{i}.{a}")).unwrap().powi(2)))
            .collect()
    };

    let without = stereo(&scene, 0.0);
    let mut reference: Option<Vec<f64>> = None;
    for backend in [SolverBackend::TinySolver, SolverBackend::Factrs] {
        let (solution, ids) = triangulate(&without, &points, stated(0.5), backend);
        for (id, p) in ids.iter().zip(&points) {
            let v = solution.value(*id).unwrap();
            let error = (Vector3::new(v[0], v[1], v[2]) - p.coords).norm();
            // The solve stops when the cost decreases by less than 1e-5
            // (whitened): microns here, against a depth σ of millimetres.
            assert!(error < 1e-5, "{backend:?}: {error} {:?}", solution.report);
        }
        let v = variances(&solution);
        match &reference {
            None => reference = Some(v),
            Some(r) => {
                for (a, b) in v.iter().zip(r) {
                    assert_close(*a, *b, 1e-4, "variance across backends");
                }
            }
        }
    }
    // Stated noise is known: σ = 1 px is four times the variance of 0.5 px,
    // although the pixels fit exactly.
    let half = reference.unwrap();
    let (one, _) = triangulate(&without, &points, stated(1.0), SolverBackend::TinySolver);
    for (a, b) in variances(&one).iter().zip(&half) {
        assert_close(*a, 4.0 * b, 1e-6, "stated variance");
    }
    // The calibration's uncertainty adds to every point's.
    let (with, _) = triangulate(
        &stereo(&scene, 1.0),
        &points,
        stated(0.5),
        SolverBackend::TinySolver,
    );
    for (a, b) in variances(&with).iter().zip(&half) {
        assert!(a > b, "{a} vs {b}");
    }
}

/// A rig pose on the robot needs a considered calibration with a hand-eye,
/// and the robot's pose exactly where the rig moves with it.
#[test]
fn rig_poses_on_a_robot_are_checked() {
    use vision_calibration_core::{Handeye, RobotFrame, RobotPose};
    let scene = scene(0.3);
    let handeye = Handeye::EyeInHand {
        gripper_se3_rig: part_pose(),
    };
    let robot = RobotPose::isotropic(part_pose().inverse(), 1e-3, 1e-3).unwrap();

    let mut problem = Problem::new();
    let err = problem
        .add_rig_pose("rig", RobotFrame::Base, Some(&robot))
        .unwrap_err();
    assert!(err.to_string().contains("before its poses"), "{err}");

    let bare = calibrate(&scene);
    let mut problem = Problem::new();
    problem.add_calibrated_rig(&bare, Use::Consider).unwrap();
    let err = problem
        .add_rig_pose("rig", RobotFrame::Gripper, None)
        .unwrap_err();
    assert!(err.to_string().contains("no hand-eye"), "{err}");

    let rig = bare.clone().with_handeye(handeye).unwrap();
    let mut problem = Problem::new();
    problem.add_calibrated_rig(&rig, Use::Consider).unwrap();
    let err = problem
        .add_rig_pose("a", RobotFrame::Gripper, Some(&robot))
        .unwrap_err();
    assert!(err.to_string().contains("does not enter"), "{err}");
    let err = problem
        .add_rig_pose("a", RobotFrame::Base, None)
        .unwrap_err();
    assert!(err.to_string().contains("state the robot's pose"), "{err}");
    let gripper = problem
        .add_rig_pose("in_gripper", RobotFrame::Gripper, None)
        .unwrap();
    let base = problem
        .add_rig_pose("in_base", RobotFrame::Base, Some(&robot))
        .unwrap();
    assert_ne!(gripper, base);
    assert!(
        problem
            .add_rig_pose("in_base", RobotFrame::Base, Some(&robot))
            .is_err()
    );

    let mut problem = Problem::new();
    problem.add_calibrated_rig(&rig, Use::Prior).unwrap();
    let err = problem
        .add_rig_pose("rig", RobotFrame::Gripper, None)
        .unwrap_err();
    assert!(err.to_string().contains("Use::Consider"), "{err}");
}
