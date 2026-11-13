//! The closed-form hand-eye methods compared under rotation-poor motion
//! (math note: `docs/notes/hand-eye.md`).
//!
//! Every trial builds an eye-in-hand scene — a known camera, a random
//! hand-eye transform, ten robot stations whose camera looks at the board
//! from directions spread by `±spread` about each axis — projects the board
//! with pixel noise and perturbs the robot poses. The intrinsics are fixed at
//! the truth, so the per-view target poses carry only the pixel noise. On
//! the same data each method then initializes the hand-eye stage and the
//! hand-eye bundle adjustment refines it; a reference refinement starts at
//! the ground truth. A method's init is "in the basin" when its refinement
//! ends within one σ (the reference's own hand-eye standard deviation) of
//! the reference.
//!
//! `handeye_methods_gate` runs a small grid on every `cargo test`;
//! `handeye_methods_comparison_table` (ignored) prints the full table:
//!
//! ```text
//! cargo test --release -p vision-calibration --test handeye_init_comparison \
//!     -- --ignored --nocapture
//! ```

#![allow(missing_docs)]

use vision_calibration::common::IntrinsicsManualInit;
use vision_calibration::common::config::RobotPoseConfig;
use vision_calibration::common::{HandeyeInitOptions, HandeyeOptimizeOptions, RobotPoseMeta};
use vision_calibration::core::{
    BrownConrady5, CameraFix, DistortionParams, FxFyCxCy, Iso3, PinholeCamera, Pt3, Real,
    Uncertainty, make_pinhole_camera,
};
use vision_calibration::init::HandeyeMethod;
use vision_calibration::nalgebra::{Translation3, UnitQuaternion, Vector3};
use vision_calibration::optim::SolverBackend;
use vision_calibration::session::CalibrationSession;
use vision_calibration::single_cam_handeye::{
    HandeyeManualInit, SingleCamHandeyeConfig, SingleCamHandeyeInput, SingleCamHandeyeProblem,
    SingleCamHandeyeSolution, SingleCamHandeyeView, step_handeye_init, step_handeye_init_with_seed,
    step_handeye_optimize, step_intrinsics_init_with_seed, step_intrinsics_optimize,
};
use vision_calibration::synthetic::{noise::UniformPixelNoise, planar, poses};

const METHODS: [HandeyeMethod; 4] = [
    HandeyeMethod::TsaiLenz,
    HandeyeMethod::ParkMartin,
    HandeyeMethod::Daniilidis,
    HandeyeMethod::Shah,
];

/// SplitMix64: a deterministic stream of uniforms for the scene generator.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in `[-a, a]`.
    fn sym(&mut self, a: Real) -> Real {
        let u = (self.next_u64() >> 11) as Real / (1u64 << 53) as Real;
        a * (2.0 * u - 1.0)
    }
    fn vec3(&mut self, a: Real) -> Vector3<Real> {
        Vector3::new(self.sym(a), self.sym(a), self.sym(a))
    }
}

fn iso(rotvec: Vector3<Real>, t: Vector3<Real>) -> Iso3 {
    Iso3::from_parts(
        Translation3::from(t),
        UnitQuaternion::from_scaled_axis(rotvec),
    )
}

/// The motion and noise regime of a table cell.
#[derive(Debug, Clone, Copy)]
struct Regime {
    /// Half-range of the per-axis viewing-direction spread, degrees.
    spread_deg: Real,
    /// Pixel noise half-range, pixels.
    pixel_noise: Real,
    /// Robot pose noise: rotation half-range (degrees), translation
    /// half-range (metres), per axis.
    robot_noise: (Real, Real),
    /// Four extra stations that translate the camera without rotating it.
    translation_only: bool,
    /// Refine the robot poses in the bundle adjustment (the workflow's
    /// default). Off, the refinement is one plain solve, ~20× faster.
    refine_robot_poses: bool,
}

struct Scene {
    camera: PinholeCamera,
    input: SingleCamHandeyeInput,
    gripper_se3_cam: Iso3,
    base_se3_target: Iso3,
}

const FX: Real = 800.0;
const FY: Real = 790.0;
const CX: Real = 512.0;
const CY: Real = 384.0;

fn board() -> Vec<Pt3> {
    planar::grid_points(9, 7, 0.03)
}

fn scene(regime: &Regime, seed: u64) -> Scene {
    let mut rng = Rng(seed.wrapping_mul(0x2545_F491_4F6C_DD1D) ^ 0xC0FF_EE00);
    let camera = make_pinhole_camera(
        FxFyCxCy {
            fx: FX,
            fy: FY,
            cx: CX,
            cy: CY,
        },
        BrownConrady5::default(),
    );
    // A hand-eye transform up to ~29° and 10 cm, and a target in the base.
    let gripper_se3_cam = iso(rng.vec3(0.3), rng.vec3(0.1));
    let base_se3_target = iso(rng.vec3(0.3), Vector3::new(0.6, 0.1, 0.0) + rng.vec3(0.1));

    // Camera stations: looking at the board centre from ~0.6 m, the viewing
    // direction rotated about the board centre by up to `spread` per axis.
    let centre = Vector3::new(0.12, 0.09, 0.0);
    let spread = regime.spread_deg.to_radians();
    let mut target_se3_cam: Vec<Iso3> = (0..10)
        .map(|_| {
            let rot = UnitQuaternion::from_scaled_axis(rng.vec3(spread));
            let dist = 0.6 + rng.sym(0.08);
            let offset = Vector3::new(rng.sym(0.03), rng.sym(0.03), 0.0);
            let position = centre + offset + rot * Vector3::new(0.0, 0.0, -dist);
            Iso3::from_parts(Translation3::from(position), rot)
        })
        .collect();
    if regime.translation_only {
        let base = target_se3_cam[0];
        for _ in 0..4 {
            let shift = Vector3::new(rng.sym(0.05), rng.sym(0.05), rng.sym(0.05));
            target_se3_cam.push(Iso3::from_parts(
                Translation3::from(base.translation.vector + shift),
                base.rotation,
            ));
        }
    }
    let cam_se3_target: Vec<Iso3> = target_se3_cam.iter().map(|p| p.inverse()).collect();
    // T_B_G = T_B_T · T_T_C · T_C_G.
    let base_se3_gripper: Vec<Iso3> = target_se3_cam
        .iter()
        .map(|t_c| base_se3_target * t_c * gripper_se3_cam.inverse())
        .collect();

    let noise = UniformPixelNoise {
        seed: seed ^ 0x5EED,
        max_abs_px: regime.pixel_noise,
    };
    let obs = planar::project_views_noisy(&camera, &board(), &cam_se3_target, &noise)
        .expect("the board projects in every view");
    let (rot_noise, trans_noise) = (regime.robot_noise.0.to_radians(), regime.robot_noise.1);
    let views = obs
        .into_iter()
        .zip(&base_se3_gripper)
        .map(|(obs, t_b_g)| {
            // The measured robot pose is off by a gripper-frame error.
            let measured = t_b_g * iso(rng.vec3(rot_noise), rng.vec3(trans_noise));
            SingleCamHandeyeView {
                obs,
                meta: RobotPoseMeta {
                    base_se3_gripper: measured,
                },
            }
        })
        .collect();
    Scene {
        camera,
        input: SingleCamHandeyeInput::new(views).expect("valid input"),
        gripper_se3_cam,
        base_se3_target,
    }
}

/// One method's result on one trial.
#[derive(Debug, Clone, Copy)]
enum Outcome {
    /// A typed error from the linear solve.
    Rejected,
    /// The init's errors and whether the refinement reached the reference.
    Init {
        rot_deg: Real,
        trans_m: Real,
        target_trans_m: Real,
        /// The refined hand-eye's distance from the reference, in units of
        /// its σ (the larger of translation and rotation); infinite when the
        /// refinement failed.
        offset_sigma: Real,
    },
}

/// The session after the (fixed-intrinsics) camera stage.
fn camera_stage(
    scene: &Scene,
    regime: &Regime,
    backend: SolverBackend,
) -> CalibrationSession<SingleCamHandeyeProblem> {
    let mut config = SingleCamHandeyeConfig::default();
    config.camera.fix = CameraFix::all(config.camera.model);
    config.solver.backend = backend;
    if !regime.refine_robot_poses {
        config.handeye_optimize.robot_poses = RobotPoseConfig::Exact;
    }
    let mut session = CalibrationSession::<SingleCamHandeyeProblem>::new();
    session.set_config(config).expect("config");
    session.set_input(scene.input.clone()).expect("input");
    let mut seed = IntrinsicsManualInit::default();
    seed.intrinsics = Some(scene.camera.k);
    seed.distortion = Some(DistortionParams::BrownConrady5 {
        params: BrownConrady5::default(),
    });
    step_intrinsics_init_with_seed(&mut session, seed, None).expect("intrinsics init");
    step_intrinsics_optimize(&mut session, None).expect("intrinsics optimize");
    session
}

/// The refined hand-eye from a session whose init is set.
fn refine(session: &mut CalibrationSession<SingleCamHandeyeProblem>) -> Option<Iso3> {
    let mut opts = HandeyeOptimizeOptions::default();
    opts.max_iters = Some(100);
    step_handeye_optimize(session, Some(opts)).ok()?;
    Some(session.output()?.params.handeye)
}

/// The smallest standard deviation of the refined hand-eye's translation
/// (metres) and rotation (radians) axes.
fn handeye_sigma(session: &CalibrationSession<SingleCamHandeyeProblem>) -> (Real, Real) {
    let out = session.output().expect("refined");
    let Uncertainty::Available(cov) = &out.uncertainty else {
        panic!("no covariance: {:?}", out.uncertainty);
    };
    let min_sd = |axes: [&str; 3]| {
        axes.iter()
            .map(|a| {
                cov.std_dev(&format!("gripper_se3_cam.{a}"))
                    .expect("estimated")
            })
            .fold(Real::INFINITY, Real::min)
    };
    (min_sd(["tx", "ty", "tz"]), min_sd(["rx", "ry", "rz"]))
}

/// Every method on one trial, plus the reference minimum.
fn trial(regime: &Regime, seed: u64, backend: SolverBackend, min_angle_deg: Real) -> [Outcome; 4] {
    let scene = scene(regime, seed);
    // Sessions are rebuilt (deterministically) rather than cloned.
    let mut reference = camera_stage(&scene, regime, backend);
    let gt = HandeyeManualInit::EyeInHand {
        gripper_se3_cam: Some(scene.gripper_se3_cam),
        base_se3_target: Some(scene.base_se3_target),
    };
    step_handeye_init_with_seed(&mut reference, gt, None).expect("seeded init");
    let x_ref = refine(&mut reference).expect("refinement from the truth");
    // Two refinements of the same problem stop on the solver's tolerance,
    // which leaves them up to ~0.8 σ apart at low noise (σ is small there;
    // the costs agree to ~1e-5); another minimum would be many σ away.
    let (sigma_t, sigma_r) = handeye_sigma(&reference);

    METHODS.map(|method| {
        let mut session = camera_stage(&scene, regime, backend);
        let mut opts = HandeyeInitOptions::default();
        opts.method = Some(method);
        opts.min_motion_angle_deg = Some(min_angle_deg);
        let Ok(init) = step_handeye_init(&mut session, Some(opts)) else {
            return Outcome::Rejected;
        };
        let SingleCamHandeyeSolution::EyeInHand {
            gripper_se3_cam: x,
            base_se3_target: z,
        } = init.handeye
        else {
            panic!("eye-in-hand init");
        };
        for p in [x, z] {
            assert!(
                p.translation.vector.iter().all(|v| v.is_finite())
                    && p.rotation.coords.iter().all(|v| v.is_finite()),
                "{method:?} returned a non-finite init"
            );
        }
        let err = poses::pose_error(&x, &scene.gripper_se3_cam);
        let target_err = poses::pose_error(&z, &scene.base_se3_target);
        let offset_sigma = refine(&mut session).map_or(Real::INFINITY, |x_fin| {
            let d = poses::pose_error(&x_fin, &x_ref);
            (d.trans / sigma_t).max(d.rot_rad() / sigma_r)
        });
        Outcome::Init {
            rot_deg: err.rot_deg,
            trans_m: err.trans,
            target_trans_m: target_err.trans,
            offset_sigma,
        }
    })
}

/// A refinement within this many σ of the reference ended at its minimum.
const BASIN_SIGMA: Real = 1.0;

/// Per-method aggregate over a cell's trials.
#[derive(Debug, Default, Clone)]
struct Summary {
    rot_deg: Vec<Real>,
    trans_mm: Vec<Real>,
    target_mm: Vec<Real>,
    in_basin: usize,
    worst_offset_sigma: Real,
    rejected: usize,
    trials: usize,
}

impl Summary {
    fn add(&mut self, o: Outcome) {
        self.trials += 1;
        match o {
            Outcome::Rejected => self.rejected += 1,
            Outcome::Init {
                rot_deg,
                trans_m,
                target_trans_m,
                offset_sigma,
            } => {
                self.rot_deg.push(rot_deg);
                self.trans_mm.push(trans_m * 1e3);
                self.target_mm.push(target_trans_m * 1e3);
                self.in_basin += usize::from(offset_sigma <= BASIN_SIGMA);
                self.worst_offset_sigma = self.worst_offset_sigma.max(offset_sigma);
            }
        }
    }
    fn basin_pct(&self) -> Real {
        100.0 * self.in_basin as Real / self.trials as Real
    }
}

fn median(v: &[Real]) -> Real {
    if v.is_empty() {
        return Real::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    }
}

fn run_cell(
    regime: &Regime,
    trials: u64,
    backend: SolverBackend,
    min_angle_deg: Real,
) -> [Summary; 4] {
    // Trials are independent and seeded, so they run on worker threads and
    // are folded in seed order: the summary does not depend on scheduling.
    let workers = std::thread::available_parallelism().map_or(4, |n| n.get()) as u64;
    let outcomes: Vec<(u64, [Outcome; 4])> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|w| {
                scope.spawn(move || {
                    (w..trials)
                        .step_by(workers as usize)
                        .map(|seed| (seed, trial(regime, seed, backend, min_angle_deg)))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("trial thread"))
            .collect()
    });
    let mut ordered = outcomes;
    ordered.sort_by_key(|(seed, _)| *seed);
    let mut out: [Summary; 4] = Default::default();
    for (_, row) in ordered {
        for (s, o) in out.iter_mut().zip(row) {
            s.add(o);
        }
    }
    out
}

/// Small grid, every run: every method's init is finite or a typed error,
/// everything converges under rich motion, and the rotation-poor cell still
/// returns finite estimates.
#[test]
fn handeye_methods_gate() {
    let rich = Regime {
        spread_deg: 45.0,
        pixel_noise: 0.3,
        robot_noise: (0.02, 2e-4),
        translation_only: false,
        refine_robot_poses: true,
    };
    let poor = Regime {
        spread_deg: 5.0,
        ..rich
    };
    for backend in [SolverBackend::TinySolver, SolverBackend::Factrs] {
        let rich_cell = run_cell(&rich, 3, backend, 5.0);
        for (method, s) in METHODS.iter().zip(&rich_cell) {
            assert_eq!(s.rejected, 0, "{method:?} rejected rich motion");
            assert_eq!(
                s.in_basin, s.trials,
                "{method:?} left the basin, {backend:?}"
            );
            assert!(median(&s.rot_deg) < 0.5, "{method:?}: {:?}", s.rot_deg);
        }
    }
    // Rotation-poor: with a 1° filter every method still answers, finite
    // (checked inside `trial`) if worse, and the refinement still converges.
    let poor_cell = run_cell(&poor, 3, SolverBackend::TinySolver, 1.0);
    for (method, s) in METHODS.iter().zip(&poor_cell) {
        assert_eq!(
            s.rejected, 0,
            "{method:?} rejected 5° motion at a 1° filter"
        );
        assert_eq!(s.in_basin, s.trials, "{method:?} left the basin at 5°");
    }
}

/// Print one table: `spreads × noises × with/without translation-only`.
fn print_table(
    label: &str,
    spreads: &[Real],
    noises: &[(&str, Real, (Real, Real))],
    refine_robot_poses: bool,
    trials: u64,
    backend: SolverBackend,
) {
    println!(
        "\n### {label}: backend {backend:?}, {trials} trials per cell, min_motion_angle_deg 1°\n"
    );
    println!(
        "| spread | noise | trans-only | method | rot err median (°) | trans err median (mm) | target err median (mm) | basin | worst offset (σ) | rejected |"
    );
    println!("|---|---|---|---|---:|---:|---:|---:|---:|---:|");
    for &spread_deg in spreads {
        for &(noise_label, pixel_noise, robot_noise) in noises {
            for translation_only in [false, true] {
                let regime = Regime {
                    spread_deg,
                    pixel_noise,
                    robot_noise,
                    translation_only,
                    refine_robot_poses,
                };
                let cell = run_cell(&regime, trials, backend, 1.0);
                for (method, s) in METHODS.iter().zip(&cell) {
                    println!(
                        "| ±{spread_deg}° | {noise_label} | {} | {method:?} | {:.3} | {:.2} | {:.2} | {:.0} % | {:.2} | {} |",
                        if translation_only { "yes" } else { "no" },
                        median(&s.rot_deg),
                        median(&s.trans_mm),
                        median(&s.target_mm),
                        s.basin_pct(),
                        s.worst_offset_sigma,
                        s.rejected,
                    );
                }
            }
        }
    }
}

/// The full comparison: four spreads × two noise levels × with/without
/// translation-only stations on both backends, the refinement without
/// robot-pose correction; and the rotation-poor cells with it.
#[test]
#[ignore = "the comparison table: run in release with --ignored --nocapture"]
fn handeye_methods_comparison_table() {
    // `HANDEYE_TABLE_TRIALS` shortens a trial run of the table itself.
    let trials: u64 = std::env::var("HANDEYE_TABLE_TRIALS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);
    let noises = [("low", 0.1, (0.0, 0.0)), ("high", 0.5, (0.05, 3e-4))];
    for backend in [SolverBackend::TinySolver, SolverBackend::Factrs] {
        print_table(
            "robot poses fixed",
            &[2.0, 5.0, 15.0, 45.0],
            &noises,
            false,
            trials,
            backend,
        );
    }
    print_table(
        "robot poses refined",
        &[2.0, 5.0],
        &noises[1..],
        true,
        (trials / 4).max(1),
        SolverBackend::TinySolver,
    );
    // The default filter on the rotation-poorest cell.
    let regime = Regime {
        spread_deg: 2.0,
        pixel_noise: 0.1,
        robot_noise: (0.0, 0.0),
        translation_only: false,
        refine_robot_poses: false,
    };
    let cell = run_cell(&regime, trials, SolverBackend::TinySolver, 5.0);
    for (method, s) in METHODS.iter().zip(&cell) {
        println!(
            "±2° at the default 5° filter: {method:?} rejected {}/{}",
            s.rejected, s.trials
        );
    }
}
