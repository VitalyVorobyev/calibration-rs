//! Robot-pose refinement against exact robot poses on a synthetic
//! eye-in-hand scene shaped like `kuka_1`: accuracy against the truth and
//! whether the reported σ is honest (normalized errors `|err|/σ` with an RMS
//! near 1 and about 95 % of them within 2σ).
//!
//! A study, not a gate (it prints a table; the Monte-Carlo gates are in
//! `uncertainty.rs`):
//! `cargo test --release -p vision-calibration-optim --test robot_pose_priors -- --ignored --nocapture`
//!
//! `STUDY_RUNS` (default 30) sets the seeds per cell, `STUDY_PX` the pixel
//! noise levels (comma-separated, default `0.1,0.3,1.0`).

#![allow(missing_docs)]

use nalgebra::{Translation3, UnitQuaternion};
use vision_calibration_core::{
    BrownConrady5, CorrespondenceView, FxFyCxCy, Iso3, PinholeCamera, Pt2, Pt3, RigView,
    RigViewObs, Uncertainty, make_pinhole_camera, pinhole_camera_params,
};
use vision_calibration_optim::{
    BackendSolveOptions, HandeyeDataset, HandeyeEstimate, HandeyeMode, HandeyeParams,
    HandeyeSolveOptions, RobotPoseMeta, apply_robot_delta, optimize_handeye,
};

/// SplitMix64 with Box–Muller Gaussians.
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

    fn gauss(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

/// `kuka_1`'s geometry: a 1928 × 1208 camera (fx 2060 px), a 17 × 28
/// chessboard of 20 mm 1–1.3 m away, the board 2.8 m from the robot base,
/// 30 robot stations.
struct Scene {
    camera: PinholeCamera,
    gripper_se3_camera: Iso3,
    base_se3_target: Iso3,
    camera_se3_target: Vec<Iso3>,
    base_se3_gripper: Vec<Iso3>,
    board: Vec<Pt3>,
}

impl Scene {
    fn new() -> Self {
        let camera = make_pinhole_camera(
            FxFyCxCy {
                fx: 2060.0,
                fy: 2060.0,
                cx: 962.0,
                cy: 610.0,
            },
            BrownConrady5 {
                k1: -0.11,
                k2: 0.14,
                k3: 0.0,
                p1: 0.0003,
                p2: 0.0002,
                iters: 8,
            },
        );
        let gripper_se3_camera = Iso3::from_parts(
            Translation3::new(0.26, 0.034, -0.10),
            UnitQuaternion::from_euler_angles(1.55, 0.02, 1.6),
        );
        let base_se3_target = Iso3::from_parts(
            Translation3::new(2.75, -0.27, 0.69),
            UnitQuaternion::from_euler_angles(-1.6, 0.03, -1.55),
        );
        let mut rng = Rng(0xC0FFEE);
        let camera_se3_target: Vec<Iso3> = (0..30)
            .map(|_| {
                let rx = 0.45 * (rng.uniform() - 0.5);
                let ry = 0.45 * (rng.uniform() - 0.5);
                let rz = 0.6 * (rng.uniform() - 0.5);
                let tx = 0.10 * (rng.uniform() - 0.5);
                let ty = 0.08 * (rng.uniform() - 0.5);
                let tz = 0.9 + 0.4 * rng.uniform();
                Iso3::from_parts(
                    Translation3::new(tx, ty, tz),
                    UnitQuaternion::from_euler_angles(rx, ry, rz),
                )
            })
            .collect();
        let base_se3_gripper = camera_se3_target
            .iter()
            .map(|c| base_se3_target * c.inverse() * gripper_se3_camera.inverse())
            .collect();
        let board = (0..17 * 28)
            .map(|i| {
                Pt3::new(
                    (i % 28) as f64 * 0.02 - 0.27,
                    (i / 28) as f64 * 0.02 - 0.16,
                    0.0,
                )
            })
            .collect();
        Self {
            camera,
            gripper_se3_camera,
            base_se3_target,
            camera_se3_target,
            base_se3_gripper,
            board,
        }
    }

    fn initial(&self) -> HandeyeParams {
        HandeyeParams {
            cameras: vec![pinhole_camera_params(&self.camera)],
            cam_to_rig: vec![Iso3::identity()],
            handeye: self.gripper_se3_camera,
            target_poses: vec![self.base_se3_target],
        }
    }

    /// Gaussian pixel noise `px` and robot poses perturbed on every view by
    /// Gaussian `(rotation rad, translation m)` noise.
    fn dataset(&self, seed: u64, px: f64, robot: (f64, f64)) -> HandeyeDataset {
        let mut rng = Rng(seed.wrapping_mul(0x2545_F491_4F6C_DD1D) ^ 0xABCD);
        let views = self
            .camera_se3_target
            .iter()
            .zip(&self.base_se3_gripper)
            .map(|(c, b)| {
                let (p3, p2): (Vec<Pt3>, Vec<Pt2>) = self
                    .board
                    .iter()
                    .filter_map(|p| {
                        let uv = self.camera.project_point(&c.transform_point(p))?;
                        Some((
                            *p,
                            Pt2::new(uv.x + px * rng.gauss(), uv.y + px * rng.gauss()),
                        ))
                    })
                    .unzip();
                let eps: [f64; 6] =
                    std::array::from_fn(|i| if i < 3 { robot.0 } else { robot.1 } * rng.gauss());
                RigView {
                    meta: RobotPoseMeta {
                        base_se3_gripper: apply_robot_delta(b, &eps),
                    },
                    obs: RigViewObs {
                        cameras: vec![Some(CorrespondenceView::new(p3, p2).unwrap())],
                    },
                }
            })
            .collect();
        HandeyeDataset::new(views, 1, HandeyeMode::EyeInHand).unwrap()
    }
}

/// The right-perturbation coordinates `[ω, v]` of `estimate = truth·Exp(δ)`.
fn pose_error(truth: &Iso3, estimate: &Iso3) -> [f64; 6] {
    let rel = truth.inverse() * estimate;
    let w = rel.rotation.scaled_axis();
    let v = rel.translation.vector;
    [w.x, w.y, w.z, v.x, v.y, v.z]
}

/// Errors, predicted variances and normalized errors over the seeds of a
/// cell.
#[derive(Default)]
struct Cell {
    runs: f64,
    /// Σ‖e‖² and Σ tr(σ²) of the hand-eye rotation, hand-eye translation,
    /// target translation (base frame) and `fx`.
    err2: [f64; 4],
    var: [f64; 4],
    /// Σ z² and the count of |z| < 2 over the 10 coordinates (hand-eye,
    /// target translation, `fx`).
    z2: f64,
    within_2: f64,
    sigma_px: f64,
}

impl Cell {
    fn push(&mut self, scene: &Scene, est: &HandeyeEstimate) {
        let Uncertainty::Available(cov) = &est.uncertainty else {
            panic!("no covariance: {:?}", est.uncertainty);
        };
        let he = pose_error(&scene.gripper_se3_camera, &est.params.handeye);
        let target = pose_error(&scene.base_se3_target, &est.params.target_poses[0]);
        let fx = est.params.cameras[0].intrinsics.fx - scene.camera.k.fx;
        let coords = ["rx", "ry", "rz", "tx", "ty", "tz"];
        let mut add = |group: usize, label: &str, e: f64| {
            let s = cov.std_dev(label).unwrap();
            self.err2[group] += e * e;
            self.var[group] += s * s;
            let z = e / s;
            self.z2 += z * z;
            self.within_2 += f64::from(u8::from(z.abs() < 2.0));
        };
        for (i, c) in coords.iter().enumerate() {
            add(i / 3, &format!("gripper_se3_camera.{c}"), he[i]);
        }
        for (i, c) in coords.iter().enumerate().skip(3) {
            add(2, &format!("base_se3_target.{c}"), target[i]);
        }
        add(3, "fx", fx);
        self.sigma_px += cov.noise[0].level.value();
        self.runs += 1.0;
    }

    fn row(&self, label: &str) -> String {
        let rms = |g: usize| (self.err2[g] / self.runs).sqrt();
        let sd = |g: usize| (self.var[g] / self.runs).sqrt();
        format!(
            "| {label} | {:.4} / {:.4} | {:.3} / {:.3} | {:.2} / {:.2} | {:.3} / {:.3} | {:.2} | {:.0} % | {:.3} |",
            rms(0).to_degrees(),
            sd(0).to_degrees(),
            rms(1) * 1e3,
            sd(1) * 1e3,
            rms(2) * 1e3,
            sd(2) * 1e3,
            rms(3),
            sd(3),
            (self.z2 / (10.0 * self.runs)).sqrt(),
            100.0 * self.within_2 / (10.0 * self.runs),
            self.sigma_px / self.runs,
        )
    }
}

#[test]
#[ignore = "study; run in release with --ignored --nocapture"]
fn robot_pose_prior_study() {
    let env = |key: &str| std::env::var(key).ok();
    let runs: u64 = env("STUDY_RUNS").map_or(30, |v| v.parse().unwrap());
    let pixel_levels: Vec<f64> = env("STUDY_PX").map_or(vec![0.1, 0.3, 1.0], |v| {
        v.split(',').map(|x| x.parse().unwrap()).collect()
    });
    let deg = std::f64::consts::PI / 180.0;
    // (label, the robot's pose noise, the priors' σ)
    let robot_cases = [
        (
            "0.01°/0.1 mm, priors matched",
            (0.01 * deg, 1e-4),
            (0.01 * deg, 1e-4),
        ),
        (
            "0.1°/1 mm, priors matched",
            (0.1 * deg, 1e-3),
            (0.1 * deg, 1e-3),
        ),
        (
            "0.05°/0.2 mm, default priors 0.5°/1 mm",
            (0.05 * deg, 2e-4),
            (0.5 * deg, 1e-3),
        ),
        (
            "0.1°/1 mm, priors 0.01°/0.1 mm",
            (0.1 * deg, 1e-3),
            (0.01 * deg, 1e-4),
        ),
    ];
    let scene = Scene::new();
    println!(
        "| σ_px | robot | scheme | hand-eye rot err / σ (°) | hand-eye trans err / σ (mm) | target trans err / σ (mm) | fx err / σ (px) | z RMS | within 2σ | σ̂ (px) |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|");
    for &px in &pixel_levels {
        for (label, robot, prior) in &robot_cases {
            let refine = HandeyeSolveOptions {
                fix_extrinsics: vec![true],
                refine_robot_poses: true,
                robot_rot_sigma: prior.0,
                robot_trans_sigma: prior.1,
                ..Default::default()
            };
            let exact = HandeyeSolveOptions {
                refine_robot_poses: false,
                ..refine.clone()
            };
            let mut cells = [
                ("refined", Cell::default()),
                ("exact poses", Cell::default()),
            ];
            for seed in 0..runs {
                for ((_, cell), opts) in cells.iter_mut().zip([&refine, &exact]) {
                    let est = optimize_handeye(
                        scene.dataset(seed, px, *robot),
                        scene.initial(),
                        opts.clone(),
                        BackendSolveOptions::default(),
                    )
                    .unwrap();
                    cell.push(&scene, &est);
                }
            }
            for (scheme, cell) in &cells {
                println!("{}", cell.row(&format!("{px} | {label} | {scheme}")));
            }
        }
    }
}
