//! Tests of the laser plane and laser residual of the problem builder:
//! what [`Problem::add_plane`] and [`Problem::add_laser`] check, and how a
//! plane is held, labelled and read back.

use nalgebra::{Translation3, UnitQuaternion, Vector3};
use vision_calibration_core::synthetic::laser::{BoardExtent, laser_stripe_pixels};
use vision_calibration_core::synthetic::noise::UniformPixelNoise;
use vision_calibration_core::{
    CameraFix, CameraParams, DistortionParams, FxFyCxCy, Iso3, LaserPlane, ProjectionParams, Pt2,
    ResidualKind, SensorParams,
};

use super::tests::{Scene, scene};
use super::*;
use crate::LaserlineResidualType;

/// A laser plane crossing the scene's boards, in the camera frame.
fn truth_plane() -> LaserPlane {
    let normal = Vector3::new(0.1, 1.0, 0.05).normalize();
    // Through the board centre region, half a metre ahead.
    let through = Vector3::new(0.0, 0.0, 0.45);
    LaserPlane::new(normal, -normal.dot(&through)).unwrap()
}

/// The laser pixels on each of the scene's boards, with noise.
fn stripes(scene: &Scene, plane: &LaserPlane) -> Vec<Vec<Pt2>> {
    let model = scene.camera.build().unwrap();
    let board = BoardExtent::from_grid(9, 6, 0.03);
    let noise = UniformPixelNoise {
        seed: 41,
        max_abs_px: 0.2,
    };
    scene
        .poses
        .iter()
        .enumerate()
        .map(|(v, pose)| {
            laser_stripe_pixels(
                &model,
                pose,
                &plane.normal().into_inner(),
                plane.distance(),
                &board,
                v,
                &noise,
            )
        })
        .collect()
}

/// A camera, its board views and a laser plane seen on them: the problem,
/// the camera, the view poses and the plane.
fn laser_problem(
    scene: &Scene,
    residual: LaserlineResidualType,
) -> (Problem, CameraId, Vec<BlockId>, PlaneId) {
    let mut problem = Problem::new();
    let camera = problem
        .add_camera(&scene.camera, &CameraFix::default())
        .unwrap();
    let truth = truth_plane();
    let mut poses = Vec::new();
    for (v, (pose, pixels)) in scene.poses.iter().zip(&scene.pixels).enumerate() {
        let pose = problem.add_pose(format!("view{v}"), pose).unwrap();
        for (point, pixel) in scene.board.iter().zip(pixels) {
            problem
                .add_reprojection(
                    camera,
                    PoseChain::Pose(pose),
                    TargetPoint::Known(*point),
                    *pixel,
                    PixelNoise::Estimated,
                    RobustLoss::None,
                )
                .unwrap();
        }
        poses.push(pose);
    }
    let seed = LaserPlane::new(
        UnitQuaternion::from_scaled_axis(Vector3::new(0.03, 0.0, 0.02))
            * truth.normal().into_inner(),
        truth.distance() + 0.004,
    )
    .unwrap();
    let plane = problem.add_plane("plane", &seed).unwrap();
    for (pose, pixels) in poses.iter().zip(stripes(scene, &truth)) {
        for pixel in pixels {
            problem
                .add_laser(
                    camera,
                    PoseChain::Pose(*pose),
                    plane,
                    pixel,
                    residual,
                    LaserNoise::Estimated,
                    RobustLoss::None,
                )
                .unwrap();
        }
    }
    (problem, camera, poses, plane)
}

fn add_one(
    problem: &mut Problem,
    camera: CameraId,
    chain: PoseChain,
    plane: PlaneId,
    pixel: Pt2,
    residual: LaserlineResidualType,
) -> Result<(), Error> {
    problem.add_laser(
        camera,
        chain,
        plane,
        pixel,
        residual,
        LaserNoise::Estimated,
        RobustLoss::None,
    )
}

#[test]
fn a_plane_is_recovered_with_its_labels_and_family() {
    let scene = scene(0.2);
    for residual in [
        LaserlineResidualType::PointToPlane,
        LaserlineResidualType::LineDistNormalized,
    ] {
        let (problem, _, _, plane) = laser_problem(&scene, residual);
        let solution = solve(&problem, &SolveOptions::default()).unwrap();
        let got = solution.plane(plane).unwrap();
        let truth = truth_plane();
        let angle = got.normal().dot(&truth.normal()).clamp(-1.0, 1.0).acos();
        assert!(angle < 2e-3, "{residual:?}: normal off by {angle} rad");
        assert!((got.distance() - truth.distance()).abs() < 1e-3);
        let covariance = solution.uncertainty.covariance().unwrap();
        for label in ["plane.nx", "plane.ny", "plane.nz", "plane.d"] {
            assert!(
                covariance.index(label).is_some(),
                "{label} in the covariance"
            );
        }
        let family = match residual {
            LaserlineResidualType::PointToPlane => ResidualKind::LaserPlane,
            LaserlineResidualType::LineDistNormalized => ResidualKind::LaserLine,
        };
        assert!(solution.noise.iter().any(|n| n.kind == family));
    }
}

#[test]
fn a_fixed_plane_stays_and_leaves_the_covariance() {
    let scene = scene(0.2);
    let (mut problem, _, _, plane) = laser_problem(&scene, LaserlineResidualType::PointToPlane);
    let p = problem.plane_index(plane).unwrap();
    let seed = (
        problem.blocks[problem.planes[p].normal].values.clone(),
        problem.blocks[problem.planes[p].distance].values.clone(),
    );
    problem.fix_plane(plane).unwrap();
    let solution = solve(&problem, &SolveOptions::default()).unwrap();
    let got = solution.plane(plane).unwrap();
    assert_eq!(
        got.normal().into_inner(),
        Vector3::from_column_slice(seed.0.as_slice())
    );
    assert_eq!(got.distance(), seed.1[0]);
    let covariance = solution.uncertainty.covariance().unwrap();
    assert!(covariance.index("plane.d").is_none());
}

#[test]
fn a_plane_of_another_problem_is_rejected() {
    let scene = scene(0.0);
    let (mut problem, camera, poses, _) =
        laser_problem(&scene, LaserlineResidualType::PointToPlane);
    let other = Problem::new().add_plane("plane", &truth_plane()).unwrap();
    let pixel = stripes(&scene, &truth_plane())[0][0];
    let err = add_one(
        &mut problem,
        camera,
        PoseChain::Pose(poses[0]),
        other,
        pixel,
        LaserlineResidualType::PointToPlane,
    )
    .unwrap_err();
    assert!(err.to_string().contains("another problem"), "{err}");
    assert!(problem.fix_plane(other).is_err());
    let solution = solve(&problem, &SolveOptions::default()).unwrap();
    assert!(solution.plane(other).is_none());
}

#[test]
fn chains_are_checked() {
    let scene = scene(0.0);
    let (mut problem, camera, poses, plane) =
        laser_problem(&scene, LaserlineResidualType::PointToPlane);
    let pixel = stripes(&scene, &truth_plane())[0][0];
    let vector = problem
        .add_block("v", &[0.0, 0.0, 1.0], Manifold::Euclidean)
        .unwrap();
    let not_se3 = add_one(
        &mut problem,
        camera,
        PoseChain::Pose(vector),
        plane,
        pixel,
        LaserlineResidualType::PointToPlane,
    )
    .unwrap_err();
    assert!(not_se3.to_string().contains("not SE3"), "{not_se3}");
    let twice = add_one(
        &mut problem,
        camera,
        PoseChain::Rig {
            cam_se3_rig: poses[0],
            rig_se3_target: poses[0],
        },
        plane,
        pixel,
        LaserlineResidualType::PointToPlane,
    )
    .unwrap_err();
    assert!(twice.to_string().contains("twice"), "{twice}");
    let nan = add_one(
        &mut problem,
        camera,
        PoseChain::Pose(poses[0]),
        plane,
        Pt2::new(f64::NAN, 1.0),
        LaserlineResidualType::PointToPlane,
    )
    .unwrap_err();
    assert!(nan.to_string().contains("finite"), "{nan}");
    let bad_loss = problem
        .add_laser(
            camera,
            PoseChain::Pose(poses[0]),
            plane,
            pixel,
            LaserlineResidualType::PointToPlane,
            LaserNoise::Estimated,
            RobustLoss::Huber { scale: -1.0 },
        )
        .unwrap_err();
    assert!(bad_loss.to_string().contains("residual"), "{bad_loss}");
}

#[test]
fn a_ray_that_misses_the_target_is_rejected() {
    let scene = scene(0.0);
    let (mut problem, camera, _, plane) =
        laser_problem(&scene, LaserlineResidualType::PointToPlane);
    // The board behind the camera.
    let behind = problem
        .add_pose(
            "behind",
            &Iso3::from_parts(
                Translation3::new(0.0, 0.0, -0.5),
                UnitQuaternion::identity(),
            ),
        )
        .unwrap();
    // The board edge-on: its plane contains the optical axis.
    let edge_on = problem
        .add_pose(
            "edge_on",
            &Iso3::from_parts(
                Translation3::new(0.0, 0.0, 0.5),
                UnitQuaternion::from_scaled_axis(Vector3::new(
                    std::f64::consts::FRAC_PI_2,
                    0.0,
                    0.0,
                )),
            ),
        )
        .unwrap();
    for pose in [behind, edge_on] {
        let err = add_one(
            &mut problem,
            camera,
            PoseChain::Pose(pose),
            plane,
            Pt2::new(640.0, 360.0),
            LaserlineResidualType::PointToPlane,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("does not meet the target"),
            "{err}"
        );
    }
}

#[test]
fn the_line_distance_needs_a_projection_that_keeps_lines() {
    let fisheye = CameraParams::new(
        ProjectionParams::Equidistant,
        DistortionParams::None,
        SensorParams::Identity,
        FxFyCxCy {
            fx: 600.0,
            fy: 600.0,
            cx: 640.0,
            cy: 360.0,
        },
    );
    let mut problem = Problem::new();
    let camera = problem.add_camera(&fisheye, &CameraFix::default()).unwrap();
    let pose = problem
        .add_pose(
            "board",
            &Iso3::from_parts(
                Translation3::new(-0.1, -0.1, 0.5),
                UnitQuaternion::identity(),
            ),
        )
        .unwrap();
    let plane = problem.add_plane("plane", &truth_plane()).unwrap();
    let pixel = Pt2::new(640.0, 360.0);
    let err = add_one(
        &mut problem,
        camera,
        PoseChain::Pose(pose),
        plane,
        pixel,
        LaserlineResidualType::LineDistNormalized,
    )
    .unwrap_err();
    assert!(err.to_string().contains("point-to-plane"), "{err}");
    add_one(
        &mut problem,
        camera,
        PoseChain::Pose(pose),
        plane,
        pixel,
        LaserlineResidualType::PointToPlane,
    )
    .unwrap();
}

#[test]
fn plane_names_are_checked() {
    let mut problem = Problem::new();
    problem
        .add_block("used", &[0.0], Manifold::Euclidean)
        .unwrap();
    assert!(problem.add_plane("", &truth_plane()).is_err());
    assert!(problem.add_plane("cam0", &truth_plane()).is_err());
    assert!(problem.add_plane("used", &truth_plane()).is_err());
    problem.add_plane("cam0.plane", &truth_plane()).unwrap();
    assert!(problem.add_plane("cam0.plane", &truth_plane()).is_err());
    // A block cannot take a plane's name or one of its blocks' names.
    for name in ["cam0.plane", "cam0.plane.normal", "cam0.plane.d"] {
        assert!(
            problem
                .add_block(name, &[0.0], Manifold::Euclidean)
                .is_err(),
            "{name}"
        );
    }
}

#[test]
fn a_laser_residual_cannot_join_a_prior() {
    let scene = scene(0.2);
    let rig = {
        let mut problem = Problem::new();
        let camera = problem
            .add_camera(&scene.camera, &CameraFix::default())
            .unwrap();
        for (v, (pose, pixels)) in scene.poses.iter().zip(&scene.pixels).enumerate() {
            let pose = problem.add_pose(format!("view{v}"), pose).unwrap();
            for (point, pixel) in scene.board.iter().zip(pixels) {
                problem
                    .add_reprojection(
                        camera,
                        PoseChain::Pose(pose),
                        TargetPoint::Known(*point),
                        *pixel,
                        PixelNoise::Estimated,
                        RobustLoss::None,
                    )
                    .unwrap();
            }
        }
        let solution = solve(&problem, &SolveOptions::default()).unwrap();
        vision_calibration_core::CalibratedRig::single(
            solution.camera(camera).unwrap(),
            solution.uncertainty,
        )
        .unwrap()
    };
    let mut problem = Problem::new();
    let blocks = problem.add_calibrated_rig(&rig, Use::Prior).unwrap();
    let camera = blocks.camera(0).unwrap();
    let pose = problem.add_pose("board", &scene.poses[0]).unwrap();
    let stated =
        PixelNoise::Stated(vision_calibration_core::PointCovariance::new(0.04, 0.0, 0.04).unwrap());
    for (point, pixel) in scene.board.iter().zip(&scene.pixels[0]) {
        problem
            .add_reprojection(
                camera,
                PoseChain::Pose(pose),
                TargetPoint::Known(*point),
                *pixel,
                stated,
                RobustLoss::None,
            )
            .unwrap();
    }
    // Without a laser residual the prior solves.
    solve(&problem, &SolveOptions::default()).unwrap();
    let plane = problem.add_plane("plane", &truth_plane()).unwrap();
    for pixel in &stripes(&scene, &truth_plane())[0] {
        add_one(
            &mut problem,
            camera,
            PoseChain::Pose(pose),
            plane,
            *pixel,
            LaserlineResidualType::PointToPlane,
        )
        .unwrap();
    }
    let err = solve(&problem, &SolveOptions::default()).unwrap_err();
    assert!(err.to_string().contains("laser"), "{err}");
}
