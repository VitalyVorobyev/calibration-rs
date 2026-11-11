//! The rig view information against central differences of the projection,
//! and against the covariance of a hand-eye solve rebuilt from it.

use nalgebra::{DMatrix, Vector3};
use vision_calibration_core::synthetic::{noise::UniformPixelNoise, planar, poses};
use vision_calibration_core::{
    BrownConrady5, CorrespondenceView, DistortionParams, FxFyCxCy, NoiseLevel, ProjectionParams,
    ResidualKind, RigView, RigViewObs, ScheimpflugParams, SensorParams, Uncertainty,
};

use super::*;
use crate::backend::BackendSolveOptions;
use crate::problems::handeye::{
    HandeyeDataset, HandeyeParams, HandeyeSolveOptions, RobotPoseMeta, optimize_handeye,
};

const IMAGE: [f64; 2] = [1280.0, 960.0];

fn cameras() -> Vec<CameraParams> {
    let brown_conrady = |k1: f64| DistortionParams::BrownConrady5 {
        params: BrownConrady5 {
            k1,
            k2: 0.02,
            k3: 0.0,
            p1: 0.0005,
            p2: -0.0003,
            iters: 8,
        },
    };
    vec![
        CameraParams {
            projection: ProjectionParams::Pinhole,
            distortion: brown_conrady(-0.08),
            sensor: SensorParams::Identity,
            intrinsics: FxFyCxCy {
                fx: 1000.0,
                fy: 1000.0,
                cx: 640.0,
                cy: 480.0,
            },
        },
        CameraParams {
            projection: ProjectionParams::Pinhole,
            distortion: brown_conrady(-0.05),
            sensor: SensorParams::Scheimpflug {
                params: ScheimpflugParams {
                    tilt_x: 0.02,
                    tilt_y: -0.015,
                },
            },
            intrinsics: FxFyCxCy {
                fx: 1100.0,
                fy: 1095.0,
                cx: 630.0,
                cy: 490.0,
            },
        },
    ]
}

fn board() -> Vec<Pt3> {
    planar::grid_points(9, 7, 0.03)
}

/// Eye-in-hand: the bench's rig hand-eye truth. Eye-to-hand: the rig in the
/// base, looking up at a target on the gripper.
fn truth(mode: HandeyeMode) -> (Iso3, Iso3) {
    match mode {
        HandeyeMode::EyeInHand => (
            poses::make_iso((0.10, -0.05, 0.02), (0.05, -0.03, 0.08)),
            poses::make_iso((0.05, -0.08, 0.0), (-0.10, -0.08, 1.0)),
        ),
        HandeyeMode::EyeToHand => (
            // rig_se3_base, gripper_se3_target.
            poses::make_iso((0.04, 0.03, -0.02), (0.06, 0.05, 1.0)),
            poses::make_iso((-0.05, 0.08, 0.1), (-0.10, -0.06, 0.02)),
        ),
    }
}

fn geometry<'a>(
    cameras: &'a [CameraParams],
    layout: &'a [Iso3],
    mode: HandeyeMode,
) -> RigHandeyeGeometry<'a> {
    let (handeye, target) = truth(mode);
    RigHandeyeGeometry {
        cameras,
        cam_se3_rig: layout,
        mode,
        handeye,
        target,
    }
}

/// The board points each camera sees in the image at the robot pose `g`.
fn visible(geometry: &RigHandeyeGeometry<'_>, g: &Iso3) -> Vec<Vec<Pt3>> {
    (0..geometry.cameras.len())
        .map(|i| {
            let model = geometry.cameras[i].build().unwrap();
            let chain = Chain::new(geometry, i, g);
            board()
                .into_iter()
                .filter(|x| {
                    model
                        .project_point_c(&chain.camera_point(x))
                        .is_some_and(|q| {
                            (0.0..=IMAGE[0]).contains(&q.x) && (0.0..=IMAGE[1]).contains(&q.y)
                        })
                })
                .collect()
        })
        .collect()
}

fn camera_from(theta: &[f64], like: &CameraParams) -> CameraParams {
    let spec = like.spec().unwrap();
    let (proj, rest) = theta[4..].split_at(spec.projection.dim());
    let (dist, sensor) = rest.split_at(spec.distortion.dim());
    CameraParams {
        intrinsics: FxFyCxCy {
            fx: theta[0],
            fy: theta[1],
            cx: theta[2],
            cy: theta[3],
        },
        projection: ProjectionParams::from_values(spec.projection, proj).unwrap(),
        distortion: DistortionParams::from_values(spec.distortion, dist).unwrap(),
        sensor: SensorParams::from_values(spec.sensor, sensor).unwrap(),
    }
}

/// `T · Exp(h eₖ)`: one tangent coordinate at a time is exactly the SE(3)
/// exponential.
fn right(t: &Iso3, k: usize, h: f64) -> Iso3 {
    let mut d = [0.0; 6];
    d[k] = h;
    t * Iso3::new(
        Vector3::new(d[3], d[4], d[5]),
        Vector3::new(d[0], d[1], d[2]),
    )
}

/// Central-difference information over the labels and the robot
/// correction, the poses perturbed on the right.
fn numeric(geometry: &RigHandeyeGeometry<'_>, g: &Iso3, points: &[Vec<Pt3>]) -> DMatrix<f64> {
    let layout = Layout::new(geometry.cameras, geometry.mode).unwrap();
    let n = layout.labels.len();
    let mut a = DMatrix::<f64>::zeros(n + 6, n + 6);
    for (i, camera_points) in points.iter().enumerate() {
        let theta: Vec<f64> = layout.vectors[i].theta.iter().copied().collect();
        for x in camera_points {
            let pixel = |column: usize, h: f64| {
                let mut cameras = geometry.cameras.to_vec();
                let mut layout_poses = geometry.cam_se3_rig.to_vec();
                let (mut handeye, mut target, mut robot) = (geometry.handeye, geometry.target, *g);
                let m = theta.len();
                if (layout.camera[i]..layout.camera[i] + m).contains(&column) {
                    let mut t = theta.clone();
                    let k = column - layout.camera[i];
                    t[k] += h * t[k].abs().max(1.0);
                    cameras[i] = camera_from(&t, &geometry.cameras[i]);
                } else if let Some(c) =
                    layout.extrinsics[i].filter(|c| (*c..c + 6).contains(&column))
                {
                    layout_poses[i] = right(&layout_poses[i], column - c, h);
                } else if (layout.handeye..layout.handeye + 6).contains(&column) {
                    handeye = right(&handeye, column - layout.handeye, h);
                } else if (layout.target..layout.target + 6).contains(&column) {
                    target = right(&target, column - layout.target, h);
                } else if column >= n {
                    robot = right(&robot, column - n, h);
                }
                let perturbed = RigHandeyeGeometry {
                    cameras: &cameras,
                    cam_se3_rig: &layout_poses,
                    handeye,
                    target,
                    ..*geometry
                };
                let p = Chain::new(&perturbed, i, &robot).camera_point(x);
                let q = cameras[i].build().unwrap().project_point_c(&p).unwrap();
                Vector3::new(q.x, q.y, 0.0)
            };
            let mut j = DMatrix::<f64>::zeros(2, n + 6);
            for column in 0..n + 6 {
                let h = 1e-6;
                let scale = if (layout.camera[i]..layout.camera[i] + theta.len()).contains(&column)
                {
                    h * theta[column - layout.camera[i]].abs().max(1.0)
                } else {
                    h
                };
                let d = (pixel(column, h) - pixel(column, -h)) / (2.0 * scale);
                j[(0, column)] = d.x;
                j[(1, column)] = d.y;
            }
            a += j.transpose() * &j;
        }
    }
    a
}

/// The two matrices agree in the first one's unit-diagonal frame, where
/// every entry is O(1).
fn assert_close(exact: &DMatrix<f64>, other: &DMatrix<f64>, tolerance: f64) {
    let n = exact.nrows();
    let d: Vec<f64> = (0..n).map(|i| 1.0 / exact[(i, i)].sqrt()).collect();
    let mut worst = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let e = (exact[(i, j)] - other[(i, j)]) * d[i] * d[j];
            worst = worst.max(e.abs());
        }
    }
    assert!(
        worst < tolerance,
        "largest difference {worst} ≥ {tolerance}"
    );
}

#[test]
fn the_information_matches_central_differences_in_both_modes() {
    let cameras = cameras();
    let layout = poses::rig_layout(2, 0.12, 0.05);
    for mode in [HandeyeMode::EyeInHand, HandeyeMode::EyeToHand] {
        let geometry = geometry(&cameras, &layout, mode);
        let g = poses::robot_stations(5, 1)[4];
        let points = visible(&geometry, &g);
        assert!(
            points.iter().all(|p| p.len() > 20),
            "{mode:?}: the board is in view"
        );

        let a = numeric(&geometry, &g, &points);
        let n = a.nrows() - 6;
        let exact = rig_handeye_view_information(&geometry, &g, &points, RobotPose::Exact).unwrap();
        assert_eq!(exact.labels.len(), n);
        assert_close(&exact.matrix, &a.view((0, 0), (n, n)).into_owned(), 1e-5);

        // The correction's prior per unit pixel variance, then its Schur
        // complement.
        let sigma = RobotPoseSigma {
            rot_sigma: 1e-3,
            trans_sigma: 5e-4,
        };
        let pixel_sigma = 0.3;
        let corrected = rig_handeye_view_information(
            &geometry,
            &g,
            &points,
            RobotPose::Corrected { sigma, pixel_sigma },
        )
        .unwrap();
        let mut a_dd = a.view((n, n), (6, 6)).into_owned();
        for k in 0..6 {
            let s = if k < 3 {
                sigma.rot_sigma
            } else {
                sigma.trans_sigma
            };
            a_dd[(k, k)] += (pixel_sigma / s).powi(2);
        }
        let a_td = a.view((0, n), (n, 6)).into_owned();
        let schur = a.view((0, 0), (n, n)).into_owned()
            - &a_td * a_dd.try_inverse().unwrap() * a_td.transpose();
        assert_close(&corrected.matrix, &schur, 1e-5);
    }
}

#[test]
fn labels_name_the_covariance_parameters() {
    let cameras = cameras();
    let layout = poses::rig_layout(2, 0.12, 0.05);
    let geometry = geometry(&cameras, &layout, HandeyeMode::EyeToHand);
    let g = poses::robot_stations(1, 1)[0];
    let info =
        rig_handeye_view_information(&geometry, &g, &visible(&geometry, &g), RobotPose::Exact)
            .unwrap();
    let has = |l: &str| info.labels.iter().any(|x| x == l);
    for label in [
        "cam0.fx",
        "cam1.tilt_x",
        "cam1.cam_se3_rig.rx",
        "rig_se3_base.tz",
        "gripper_se3_target.rx",
    ] {
        assert!(has(label), "{label} in {:?}", info.labels);
    }

    // One camera is the rig: no prefix and no camera-to-rig pose.
    let one = &cameras[..1];
    let identity = [Iso3::identity()];
    let single = RigHandeyeGeometry {
        cameras: one,
        cam_se3_rig: &identity,
        ..geometry
    };
    let points = visible(&single, &g);
    let info = rig_handeye_view_information(&single, &g, &points, RobotPose::Exact).unwrap();
    assert_eq!(info.labels[0], "fx");
    assert!(info.labels.iter().any(|l| l == "cam_se3_base.rx"));
    assert!(!info.labels.iter().any(|l| l.contains("cam_se3_rig")));
}

#[test]
fn inconsistent_inputs_are_rejected() {
    let cameras = cameras();
    let layout = poses::rig_layout(2, 0.12, 0.05);
    let geometry = geometry(&cameras, &layout, HandeyeMode::EyeInHand);
    let g = Iso3::identity();
    let err =
        rig_handeye_view_information(&geometry, &g, &[board()], RobotPose::Exact).unwrap_err();
    assert!(matches!(err, Error::InvalidInput { .. }), "{err}");
    let bad = RobotPose::Corrected {
        sigma: RobotPoseSigma {
            rot_sigma: 1e-3,
            trans_sigma: 1e-3,
        },
        pixel_sigma: 0.0,
    };
    let points = visible(&geometry, &g);
    let err = rig_handeye_view_information(&geometry, &g, &points, bad).unwrap_err();
    assert!(matches!(err, Error::InvalidInput { .. }), "{err}");
}

/// Gaussian pixel noise of standard deviation `noise.max_abs_px` per axis
/// (Box-Muller over the deterministic uniform samples).
fn gaussian(noise: &UniformPixelNoise, view: usize, point: usize) -> nalgebra::Vector2<f64> {
    let unit = |k: usize| {
        let s = noise.sample(view, k);
        let a = noise.max_abs_px;
        (s.x / a * 0.5 + 0.5, s.y / a * 0.5 + 0.5)
    };
    let (u1, u2) = unit(2 * point);
    let (u3, u4) = unit(2 * point + 1);
    let r = |u: f64| (-2.0 * (1.0 - u).ln()).sqrt() * noise.max_abs_px;
    let tau = std::f64::consts::TAU;
    nalgebra::Vector2::new(r(u1) * (tau * u2).cos(), r(u3) * (tau * u4).cos())
}

/// A hand-eye solve of noisy synthetic views, and the information of those
/// views at its estimate, restricted to the covariance's parameters.
fn solve_and_rebuild(mode: HandeyeMode, refine_robot_poses: bool) -> (DMatrix<f64>, DMatrix<f64>) {
    let cameras = cameras();
    let layout = poses::rig_layout(2, 0.12, 0.05);
    let truth_geometry = geometry(&cameras, &layout, mode);
    let stations = poses::robot_stations(14, 7);
    let noise = UniformPixelNoise {
        seed: 11,
        max_abs_px: 0.3,
    };
    let views: Vec<RigView<RobotPoseMeta>> = stations
        .iter()
        .enumerate()
        .map(|(v, g)| {
            let points = visible(&truth_geometry, g);
            let cameras = points
                .iter()
                .enumerate()
                .map(|(i, pts)| {
                    let model = cameras[i].build().unwrap();
                    let chain = Chain::new(&truth_geometry, i, g);
                    let pixels = pts
                        .iter()
                        .enumerate()
                        .map(|(k, x)| {
                            let q = model.project_point_c(&chain.camera_point(x)).unwrap();
                            (q.coords + gaussian(&noise, v * 2 + i, k)).into()
                        })
                        .collect();
                    Some(CorrespondenceView::new(pts.clone(), pixels).unwrap())
                })
                .collect();
            RigView {
                meta: RobotPoseMeta {
                    base_se3_gripper: *g,
                },
                obs: RigViewObs { cameras },
            }
        })
        .collect();
    let dataset = HandeyeDataset::new(views, 2, mode).unwrap();
    let (handeye, target) = truth(mode);
    let sigma = RobotPoseSigma {
        rot_sigma: 2e-4,
        trans_sigma: 2e-4,
    };
    let estimate = optimize_handeye(
        dataset,
        HandeyeParams {
            cameras: cameras.clone(),
            rig_se3_cam: layout.iter().map(Iso3::inverse).collect(),
            handeye,
            target_poses: vec![target],
        },
        HandeyeSolveOptions {
            fix_extrinsics: vec![true, false],
            refine_robot_poses,
            robot_rot_sigma: sigma.rot_sigma,
            robot_trans_sigma: sigma.trans_sigma,
            ..HandeyeSolveOptions::default()
        },
        BackendSolveOptions::default(),
    )
    .unwrap();
    let Uncertainty::Available(covariance) = &estimate.uncertainty else {
        panic!("no covariance: {:?}", estimate.uncertainty);
    };
    let pixel_sigma = covariance
        .noise
        .iter()
        .find(|n| n.kind == ResidualKind::Reprojection)
        .map(|n| match n.level {
            NoiseLevel::Sigma(s) | NoiseLevel::Sigma0(s) => s,
        })
        .unwrap();
    let robot = if refine_robot_poses {
        RobotPose::Corrected { sigma, pixel_sigma }
    } else {
        RobotPose::Exact
    };

    let params = &estimate.params;
    let cam_se3_rig: Vec<Iso3> = params.rig_se3_cam.iter().map(Iso3::inverse).collect();
    let solved = RigHandeyeGeometry {
        cameras: &params.cameras,
        cam_se3_rig: &cam_se3_rig,
        mode,
        handeye: params.handeye,
        target: params.target_poses[0],
    };
    let index: Vec<usize> = covariance
        .labels
        .iter()
        .map(|label| {
            Layout::new(&params.cameras, mode)
                .unwrap()
                .labels
                .iter()
                .position(|l| l == label)
                .unwrap_or_else(|| panic!("{label} is not a view parameter"))
        })
        .collect();
    let n = index.len();
    let mut information = DMatrix::<f64>::zeros(n, n);
    for (v, g) in stations.iter().enumerate() {
        let g = match estimate.robot_refinement.as_ref().map(|r| &r.deltas) {
            Some(deltas) => crate::problems::handeye::apply_robot_delta(g, &deltas[v]),
            None => *g,
        };
        let points = visible(&truth_geometry, &stations[v]);
        let info = rig_handeye_view_information(&solved, &g, &points, robot).unwrap();
        information += DMatrix::from_fn(n, n, |a, b| info.matrix[(index[a], index[b])]);
    }
    let sigma_matrix = DMatrix::from_fn(n, n, |a, b| covariance.covariance[a][b]);
    (sigma_matrix, information)
}

/// The solver's covariance is `σ²` times the inverse of the views'
/// information: the labels, the pose conventions and the prior all match
/// the solve's.
#[test]
fn the_views_information_reproduces_the_hand_eye_covariance() {
    for mode in [HandeyeMode::EyeInHand, HandeyeMode::EyeToHand] {
        for refine in [false, true] {
            let (covariance, information) = solve_and_rebuild(mode, refine);
            let n = covariance.nrows();
            let sigma2 = (&covariance * &information).trace() / n as f64;
            let predicted = information.try_inverse().unwrap() * sigma2;
            // Within 3 % in σ and 0.03 in correlation: a refined solve weighs
            // its priors at its first solve's noise level.
            assert_close(&covariance, &predicted, 0.03);
        }
    }
}
