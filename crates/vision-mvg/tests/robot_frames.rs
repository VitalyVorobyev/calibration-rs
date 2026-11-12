//! Triangulated points in a robot's frames, with the hand-eye's and the
//! robot poses' uncertainty, against Monte-Carlo in parameter space.
//!
//! A stereo rig on a robot (eye-in-hand) sees points fixed in the base from
//! two robot stations; a fixed stereo rig (eye-to-hand) sees points held by
//! the gripper at two stations. Each trial draws the calibration (cameras,
//! extrinsic and hand-eye, correlated as one solve correlates them), the
//! robot's pose error at each station and the pixel noise, and
//! triangulates in the base (eye-in-hand) or gripper (eye-to-hand) frame:
//! the scatter of the points is the covariance `triangulate` states at the
//! nominal calibration, the cross terms between the cameras and the
//! hand-eye included. Without those cross terms, or without the robot's
//! share, the stated covariance is measurably off.

#![allow(missing_docs)]

use nalgebra::{DMatrix, DVector, Translation3, UnitQuaternion, Vector3};
use vision_calibration_core::{
    BrownConrady5, CalibratedRig, CameraParams, FxFyCxCy, Handeye, Iso3, ParameterCovariance,
    PointCovariance, Pt2, Pt3, RobotFrame, RobotPose, RunControl, Uncertainty, make_pinhole_camera,
    pinhole_camera_params,
};
use vision_mvg::Error;
use vision_mvg::triangulation::{RigObservation, triangulate};
use vision_mvg::types::Captures;

/// A deterministic generator (SplitMix64) with normal draws.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

fn iso(r: [f64; 3], t: [f64; 3]) -> Iso3 {
    Iso3::from_parts(
        Translation3::new(t[0], t[1], t[2]),
        UnitQuaternion::from_scaled_axis(Vector3::new(r[0], r[1], r[2])),
    )
}

/// `pose · Exp(ξ)`.
fn exp(pose: &Iso3, xi: &[f64]) -> Iso3 {
    pose * iso([xi[0], xi[1], xi[2]], [xi[3], xi[4], xi[5]])
}

fn camera(fx: f64, fy: f64, cx: f64, cy: f64) -> CameraParams {
    let k = FxFyCxCy { fx, fy, cx, cy };
    pinhole_camera_params(&make_pinhole_camera(k, BrownConrady5::default()))
}

const INTRINSICS: [&str; 4] = ["fx", "fy", "cx", "cy"];
const TANGENT: [&str; 6] = ["rx", "ry", "rz", "tx", "ty", "tz"];
const PIXEL_SIGMA: f64 = 0.3;
const ROBOT_SIGMA: (f64, f64) = (2e-4, 0.3e-3);
const TRIALS: usize = 3000;

/// The scene of one mode: the nominal calibration, its covariance over
/// `z` (cam0 and cam1 intrinsics, cam1's extrinsic, the hand-eye), the
/// robot's two stations and the points, in the frame they are fixed in.
struct Scene {
    intrinsics: [[f64; 4]; 2],
    cam1_se3_rig: Iso3,
    handeye: Handeye,
    labels: Vec<String>,
    sigma: DMatrix<f64>,
    robot: [Iso3; 2],
    points: Vec<Pt3>,
    /// The frame the points are measured in.
    frame: RobotFrame,
}

impl Scene {
    fn new(eye_in_hand: bool) -> Self {
        let handeye = if eye_in_hand {
            Handeye::EyeInHand {
                gripper_se3_rig: iso([0.05, -0.1, 0.2], [0.03, -0.05, 0.1]),
            }
        } else {
            // The rig 1.2 m from the base, looking back at the workspace.
            Handeye::EyeToHand {
                rig_se3_base: iso([0.1, std::f64::consts::PI, 0.05], [0.1, 0.0, 1.2]),
            }
        };
        let name = if eye_in_hand {
            "gripper_se3_rig"
        } else {
            "rig_se3_base"
        };
        let mut labels = Vec::new();
        for k in 0..2 {
            labels.extend(INTRINSICS.map(|p| format!("cam{k}.{p}")));
        }
        labels.extend(TANGENT.map(|c| format!("cam1.cam_se3_rig.{c}")));
        labels.extend(TANGENT.map(|c| format!("{name}.{c}")));
        let mut std = vec![2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0];
        std.extend([5e-4, 5e-4, 5e-4, 2e-4, 2e-4, 2e-4]);
        std.extend([1e-3, 1e-3, 1e-3, 0.6e-3, 0.6e-3, 0.6e-3]);
        let n = std.len();
        let mut corr = DMatrix::<f64>::identity(n, n);
        let mut tie = |a: usize, b: usize, r: f64| {
            corr[(a, b)] = r;
            corr[(b, a)] = r;
        };
        tie(0, 4, 0.5); // cam0.fx - cam1.fx
        tie(0, 19, 0.6); // cam0.fx - hand-eye tz
        tie(4, 18, -0.5); // cam1.fx - hand-eye ty
        tie(11, 17, 0.4); // extrinsic tx - hand-eye tx
        tie(2, 14, 0.5); // cam0.cx - hand-eye ry
        let sigma = DMatrix::from_fn(n, n, |a, b| std[a] * std[b] * corr[(a, b)]);
        assert!(sigma.clone().cholesky().is_some());

        // The robot's stations, and the points in the frame they are fixed
        // in: the base eye-in-hand (0.5-0.8 m ahead of the rig at station
        // 0), the gripper eye-to-hand.
        let robot = if eye_in_hand {
            [
                iso([0.0, 0.1, 0.0], [0.4, 0.0, 0.3]),
                iso([0.05, 0.0, 0.15], [0.55, 0.1, 0.32]),
            ]
        } else {
            [
                iso([0.0, 0.2, 0.0], [0.1, 0.05, 0.5]),
                iso([0.15, 0.0, -0.2], [0.15, -0.05, 0.55]),
            ]
        };
        let frame = if eye_in_hand {
            RobotFrame::Base
        } else {
            RobotFrame::Gripper
        };
        let scene = Self {
            intrinsics: [[800.0, 790.0, 320.0, 240.0], [810.0, 805.0, 330.0, 235.0]],
            cam1_se3_rig: iso([0.0, -0.06, 0.01], [-0.12, 0.0, 0.004]),
            handeye,
            labels,
            sigma,
            robot,
            points: Vec::new(),
            frame,
        };
        let station0 = scene.rig_se3_world(&scene.handeye, &robot[0]);
        let points = [(0.05, -0.02, 0.6), (-0.1, 0.06, 0.75), (0.12, 0.08, 0.5)]
            .iter()
            .map(|&(x, y, z)| station0.inverse() * Pt3::new(x, y, z))
            .collect();
        Self { points, ..scene }
    }

    fn rig_se3_world(&self, handeye: &Handeye, robot: &Iso3) -> Iso3 {
        handeye.rig_in_frame(self.frame, robot).rig_se3_frame
    }

    fn uncertainty(&self, keep: impl Fn(usize, usize) -> bool) -> Uncertainty {
        let n = self.labels.len();
        Uncertainty::Available(ParameterCovariance {
            labels: self.labels.clone(),
            covariance: (0..n)
                .map(|r| {
                    (0..n)
                        .map(|c| if keep(r, c) { self.sigma[(r, c)] } else { 0.0 })
                        .collect()
                })
                .collect(),
            views: Vec::new(),
            fixed: Vec::new(),
            noise: Vec::new(),
            num_residuals: 100,
            dof: 50,
            view_scatter: None,
            second_order: None,
        })
    }

    /// The rig at `z = z₀ + δ`, labelled with `uncertainty`.
    fn rig(&self, delta: &[f64], uncertainty: Uncertainty) -> CalibratedRig {
        let cam = |k: usize| {
            let i = &self.intrinsics[k];
            let d = &delta[4 * k..4 * k + 4];
            camera(i[0] + d[0], i[1] + d[1], i[2] + d[2], i[3] + d[3])
        };
        let handeye = match self.handeye {
            Handeye::EyeInHand { gripper_se3_rig } => Handeye::EyeInHand {
                gripper_se3_rig: exp(&gripper_se3_rig, &delta[14..20]),
            },
            Handeye::EyeToHand { rig_se3_base } => Handeye::EyeToHand {
                rig_se3_base: exp(&rig_se3_base, &delta[14..20]),
            },
        };
        CalibratedRig::new(
            vec![cam(0), cam(1)],
            vec![Iso3::identity(), exp(&self.cam1_se3_rig, &delta[8..14])],
            uncertainty,
        )
        .unwrap()
        .with_handeye(handeye)
        .unwrap()
    }

    /// Every point seen by both cameras at both stations, from the true
    /// calibration, with pixel noise.
    fn tracks(&self, rng: &mut Rng, noisy: bool) -> Vec<Vec<RigObservation>> {
        let truth = self.rig(&[0.0; 20], Uncertainty::default());
        let noise = PointCovariance::isotropic(PIXEL_SIGMA).unwrap();
        self.points
            .iter()
            .map(|p| {
                let mut track = Vec::new();
                for (k, robot) in self.robot.iter().enumerate() {
                    let rig_se3_world = self.rig_se3_world(&self.handeye, robot);
                    for c in 0..2 {
                        let px = truth
                            .model(c)
                            .unwrap()
                            .project_point(&(truth.cam_se3_rig()[c] * rig_se3_world * p))
                            .unwrap();
                        let (dx, dy) = if noisy {
                            (PIXEL_SIGMA * rng.normal(), PIXEL_SIGMA * rng.normal())
                        } else {
                            (0.0, 0.0)
                        };
                        track.push(RigObservation::new(
                            k,
                            c,
                            Pt2::new(px.x + dx, px.y + dy),
                            noise,
                        ));
                    }
                }
                track
            })
            .collect()
    }

    fn robot_poses(&self, robot: &[Iso3; 2], with_sigma: bool) -> Vec<RobotPose> {
        robot
            .iter()
            .map(|p| {
                if with_sigma {
                    RobotPose::isotropic(*p, ROBOT_SIGMA.0, ROBOT_SIGMA.1).unwrap()
                } else {
                    RobotPose::exact(*p)
                }
            })
            .collect()
    }

    fn triangulate(
        &self,
        rig: &CalibratedRig,
        robot: &[RobotPose],
        tracks: &[Vec<RigObservation>],
    ) -> vision_mvg::triangulation::Triangulation {
        triangulate(
            rig,
            Captures::Robot {
                frame: self.frame,
                robot,
            },
            tracks,
            &RunControl::default(),
        )
        .unwrap()
    }

    fn stated(&self, uncertainty: Uncertainty, robot_sigma: bool) -> DMatrix<f64> {
        let rig = self.rig(&[0.0; 20], uncertainty);
        let tracks = self.tracks(&mut Rng(0), false);
        let found = self.triangulate(&rig, &self.robot_poses(&self.robot, robot_sigma), &tracks);
        let c = found.uncertainty.covariance().unwrap();
        DMatrix::from_fn(c.labels.len(), c.labels.len(), |r, s| c.covariance[r][s])
    }
}

fn frobenius_error(sample: &DMatrix<f64>, stated: &DMatrix<f64>) -> f64 {
    (sample - stated).norm() / stated.norm()
}

/// The stated covariance of the noise-free points is the sample covariance
/// of the points over draws of the calibration, the robot poses and the
/// pixels, in both modes.
#[test]
fn robot_frame_covariance_matches_the_sampled_one() {
    for (eye_in_hand, seed) in [(true, 11_u64), (false, 12)] {
        let scene = Scene::new(eye_in_hand);
        let stated = scene.stated(scene.uncertainty(|_, _| true), true);
        let chol = scene.sigma.clone().cholesky().unwrap().l();
        let mut rng = Rng(seed);
        let n = 3 * scene.points.len();
        let mut draws = Vec::with_capacity(TRIALS);
        for _ in 0..TRIALS {
            let g = DVector::from_fn(chol.nrows(), |_, _| rng.normal());
            let delta = &chol * g;
            let rig = scene.rig(delta.as_slice(), Uncertainty::default());
            let robot = scene.robot.map(|p| {
                let xi: Vec<f64> = (0..6)
                    .map(|i| if i < 3 { ROBOT_SIGMA.0 } else { ROBOT_SIGMA.1 } * rng.normal())
                    .collect();
                exp(&p, &xi)
            });
            let tracks = scene.tracks(&mut rng, true);
            let found = scene.triangulate(&rig, &scene.robot_poses(&robot, false), &tracks);
            draws.push(DVector::from_iterator(
                n,
                found
                    .points
                    .iter()
                    .flat_map(|p| p.point.coords.iter().copied()),
            ));
        }
        let mean = draws.iter().fold(DVector::zeros(n), |a, d| a + d) / TRIALS as f64;
        let sample = draws.iter().fold(DMatrix::zeros(n, n), |a, d| {
            let e = d - &mean;
            a + &e * e.transpose()
        }) / (TRIALS as f64 - 1.0);

        let block =
            |m: &DMatrix<f64>, i: usize, j: usize| m.view((3 * i, 3 * j), (3, 3)).into_owned();
        let mut worst: f64 = 0.0;
        for i in 0..scene.points.len() {
            let error = frobenius_error(&block(&sample, i, i), &block(&stated, i, i));
            worst = worst.max(error);
            assert!(
                error <= 0.08,
                "eye-in-hand {eye_in_hand}, point {i}: {error:.3}"
            );
        }
        let cross = frobenius_error(&block(&sample, 0, 2), &block(&stated, 0, 2));
        assert!(
            cross <= 0.1,
            "eye-in-hand {eye_in_hand}, cross block: {cross:.3}"
        );

        // The cross terms between the cameras and the hand-eye, and the
        // robot's share, each move the stated covariance beyond the gate.
        let he = |i: usize| i >= 14;
        let split = scene.stated(scene.uncertainty(|r, c| he(r) == he(c)), true);
        let without_robot = scene.stated(scene.uncertainty(|_, _| true), false);
        let moved = |other: &DMatrix<f64>| {
            (0..scene.points.len())
                .map(|i| frobenius_error(&block(other, i, i), &block(&stated, i, i)))
                .fold(0.0_f64, f64::max)
        };
        let (by_cross, by_robot) = (moved(&split), moved(&without_robot));
        eprintln!(
            "eye-in-hand {eye_in_hand}: worst point block {worst:.3}, cross block {cross:.3}; \
             without the camera x hand-eye terms the σ moves by {by_cross:.3}, without the \
             robot's share by {by_robot:.3}"
        );
        assert!(by_cross > 0.15, "cross terms move it by only {by_cross:.3}");
        assert!(by_robot > 0.15, "the robot moves it by only {by_robot:.3}");
    }
}

/// The frames a rig serves, and the typed errors for the others.
#[test]
fn robot_frames_are_checked() {
    let scene = Scene::new(true);
    let rig = scene.rig(&[0.0; 20], scene.uncertainty(|_, _| true));
    let tracks = scene.tracks(&mut Rng(0), false);
    let robot = scene.robot_poses(&scene.robot, true);
    let control = RunControl::default();

    // In the gripper frame (one capture) the points are the rig frame's
    // through the hand-eye.
    let one: Vec<Vec<RigObservation>> = tracks
        .iter()
        .map(|t| t.iter().filter(|o| o.capture == 0).copied().collect())
        .collect();
    let in_rig = triangulate(&rig, Captures::Rig, &one, &control).unwrap();
    let in_gripper =
        triangulate(&rig, Captures::Handeye(RobotFrame::Gripper), &one, &control).unwrap();
    let Handeye::EyeInHand { gripper_se3_rig } = scene.handeye else {
        unreachable!()
    };
    for (g, r) in in_gripper.points.iter().zip(&in_rig.points) {
        assert!((g.point - gripper_se3_rig * r.point).norm() < 1e-9);
    }
    // The base frame recovers the points.
    let found = triangulate(
        &rig,
        Captures::Robot {
            frame: RobotFrame::Base,
            robot: &robot,
        },
        &tracks,
        &control,
    )
    .unwrap();
    for (f, p) in found.points.iter().zip(&scene.points) {
        assert!((f.point - p).norm() < 1e-9);
    }

    let invalid = |captures: Captures<'_>, rig: &CalibratedRig| {
        let err = triangulate(rig, captures, &tracks, &control).unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "{err}");
        err.to_string()
    };
    let m = invalid(Captures::Handeye(RobotFrame::Base), &rig);
    assert!(m.contains("moves in the base frame"), "{m}");
    let m = invalid(
        Captures::Robot {
            frame: RobotFrame::Gripper,
            robot: &robot,
        },
        &rig,
    );
    assert!(m.contains("fixed in the gripper frame"), "{m}");
    let bare = CalibratedRig::new(
        rig.cameras().to_vec(),
        rig.cam_se3_rig().to_vec(),
        Uncertainty::default(),
    )
    .unwrap();
    let m = invalid(Captures::Handeye(RobotFrame::Gripper), &bare);
    assert!(m.contains("hand-eye"), "{m}");
    // A capture without a robot pose.
    let m = invalid(
        Captures::Robot {
            frame: RobotFrame::Base,
            robot: &robot[..1],
        },
        &rig,
    );
    assert!(m.contains("capture 1 out of range"), "{m}");
}
