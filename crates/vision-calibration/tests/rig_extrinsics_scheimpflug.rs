//! Rig extrinsics with Scheimpflug cameras, through the facade.
//!
//! Each camera's own solve (`step_intrinsics_optimize_all`) must reach the
//! minimum of its planar problem: the reprojection noise floor, with the
//! principal point and tilt at the truth within their uncertainty. The tilt
//! and the principal point are strongly correlated (|ρ| ≈ 0.995 on this
//! scene) but identified through the radial distortion: at σ = 0.1 px the
//! tilted camera's principal point has a standard deviation of about 2.5 px
//! and its tilt about 0.15°.
//!
//! The rig solve refines the tilts with the intrinsics, so its covariance
//! carries them: the Monte-Carlo gate (ignored in the debug run; CI runs it
//! in release with `--ignored`) checks the stated σ of every camera's focal
//! length, principal point and tilt against their scatter. A tilt held at
//! the per-camera estimate leaves the principal point's σ about 20× short.

#![allow(missing_docs)]

mod common;

use common::scatter_over_stated;

use vision_calibration::common::RigIntrinsicsManualInit;
use vision_calibration::core::{
    BrownConrady5, Camera, CameraModelSpec, CameraParams, CorrespondenceView, DistortionKind,
    FxFyCxCy, Iso3, NoMeta, ParameterCovariance, Pinhole, RigDataset, RigView, RigViewObs,
    ScheimpflugParams, SensorKind, SensorParams, Uncertainty,
};
use vision_calibration::optim::SolverBackend;
use vision_calibration::rig_extrinsics::{
    RigExtrinsicsConfig, RigExtrinsicsInput, RigExtrinsicsProblem,
    step_intrinsics_init_all_with_seed, step_intrinsics_optimize_all, step_rig_init,
    step_rig_optimize,
};
use vision_calibration::session::CalibrationSession;
use vision_calibration::synthetic::{noise::UniformPixelNoise, planar, poses};

/// Per-axis pixel noise, px.
const NOISE_PX: f64 = 0.1;
/// Mean reprojection error at the noise floor: the mean norm of the
/// injected noise is about 0.13 px; a solve off the minimum (a held pose or
/// a frozen spurious coefficient) is well above 0.15 px.
const FLOOR_MEAN_PX: f64 = 0.15;
/// Principal-point error bound, px: about 3σ of the tilted camera.
const PP_TOL_PX: f64 = 7.5;
/// Tilt error bound, degrees: about 3σ of the tilted camera.
const TILT_TOL_DEG: f64 = 0.5;

struct TruthCamera {
    k: FxFyCxCy<f64>,
    k1: f64,
    tilt: ScheimpflugParams,
}

/// An untilted and a tilted camera, both with mild barrel distortion.
fn truth() -> [TruthCamera; 2] {
    [
        TruthCamera {
            k: FxFyCxCy {
                fx: 850.0,
                fy: 845.0,
                cx: 640.0,
                cy: 480.0,
            },
            k1: -0.05,
            tilt: ScheimpflugParams::default(),
        },
        TruthCamera {
            k: FxFyCxCy {
                fx: 820.0,
                fy: 815.0,
                cx: 650.0,
                cy: 470.0,
            },
            k1: -0.04,
            tilt: ScheimpflugParams {
                tilt_x: 0.06,
                tilt_y: -0.04,
            },
        },
    ]
}

/// Eight views of a 9 × 7 board (60 mm cells), tilted up to ±0.45 rad, seen
/// by both cameras of a 120 mm baseline rig, with the pixel noise of
/// `seed`.
fn dataset(seed: u64) -> RigExtrinsicsInput {
    let board = planar::grid_points(9, 7, 0.06);
    let rig_se3_target = poses::tilted_board_poses(
        &poses::seeded_tilts(8, 0.45, 7),
        (0.24, 0.18),
        0.55,
        0.2 / 8.0,
    );
    let layout = poses::rig_layout(2, 0.12, 0.0);
    let per_cam: Vec<Vec<CorrespondenceView>> = truth()
        .iter()
        .zip(&layout)
        .enumerate()
        .map(|(c, (cam, cam_se3_rig))| {
            let dist = BrownConrady5 {
                k1: cam.k1,
                ..BrownConrady5::default()
            };
            let camera = Camera::new(Pinhole, dist, cam.tilt.compile(), cam.k);
            let cam_se3_target: Vec<Iso3> =
                rig_se3_target.iter().map(|t| cam_se3_rig * t).collect();
            let noise = UniformPixelNoise {
                seed: seed + c as u64,
                max_abs_px: NOISE_PX * 3.0_f64.sqrt(),
            };
            planar::project_views_noisy(&camera, &board, &cam_se3_target, &noise)
                .expect("every corner projects")
        })
        .collect();
    let views = (0..rig_se3_target.len())
        .map(|v| RigView {
            meta: NoMeta,
            obs: RigViewObs {
                cameras: per_cam.iter().map(|views| Some(views[v].clone())).collect(),
            },
        })
        .collect();
    RigDataset::new(views, 2).expect("rig dataset")
}

/// The session after the per-camera stage on the noise of `noise_seed`, each
/// camera seeded with its nominal mount tilt (about 0.6° off, as a device
/// spec gives it).
fn per_camera_stage(
    noise_seed: u64,
    backend: SolverBackend,
) -> CalibrationSession<RigExtrinsicsProblem> {
    let mut config = RigExtrinsicsConfig::default();
    config.solver.backend = backend;
    config.camera.model =
        CameraModelSpec::pinhole(DistortionKind::BrownConrady5, SensorKind::Scheimpflug);
    let mut seed = RigIntrinsicsManualInit::default();
    seed.per_cam_sensors = Some(vec![
        ScheimpflugParams::default(),
        ScheimpflugParams {
            tilt_x: 0.05,
            tilt_y: -0.03,
        },
    ]);
    let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
    session.set_config(config).expect("config");
    session.set_input(dataset(noise_seed)).expect("input");
    step_intrinsics_init_all_with_seed(&mut session, seed, None).expect("intrinsics init");
    session
}

fn assert_near_truth(stage: &str, cameras: &[CameraParams]) {
    for (i, (camera, truth)) in cameras.iter().zip(truth()).enumerate() {
        let k = camera.intrinsics;
        let pp_err = (k.cx - truth.k.cx).hypot(k.cy - truth.k.cy);
        let SensorParams::Scheimpflug { params: tilt } = camera.sensor else {
            panic!("{stage}: camera {i} lost its Scheimpflug sensor");
        };
        let tilt_err_deg = (tilt.tilt_x - truth.tilt.tilt_x)
            .hypot(tilt.tilt_y - truth.tilt.tilt_y)
            .to_degrees();
        assert!(
            pp_err < PP_TOL_PX,
            "{stage}: camera {i} principal point {pp_err:.2} px off"
        );
        assert!(
            tilt_err_deg < TILT_TOL_DEG,
            "{stage}: camera {i} tilt {tilt_err_deg:.3}° off"
        );
        assert!(
            (k.fx / truth.k.fx - 1.0).abs() < 0.005,
            "{stage}: camera {i} fx {:.2} vs {:.2}",
            k.fx,
            truth.k.fx
        );
    }
}

#[test]
fn per_camera_stage_reaches_the_minimum() {
    let mut session = per_camera_stage(11, SolverBackend::TinySolver);
    let result = step_intrinsics_optimize_all(&mut session, None).expect("intrinsics optimize");
    for (i, err) in result.per_cam_reproj_errors.iter().enumerate() {
        assert!(
            *err < FLOOR_MEAN_PX,
            "camera {i} mean reprojection error {err:.4} px is above the floor"
        );
    }
    assert_near_truth("per-camera stage", &result.cameras);
}

#[test]
fn rig_solve_keeps_the_cameras_at_the_truth() {
    let mut session = per_camera_stage(11, SolverBackend::TinySolver);
    step_intrinsics_optimize_all(&mut session, None).expect("intrinsics optimize");
    step_rig_init(&mut session, None).expect("rig init");
    let result = step_rig_optimize(&mut session, None).expect("rig optimize");
    assert!(
        result.mean_reproj_error < FLOOR_MEAN_PX,
        "rig mean reprojection error {:.4} px is above the floor",
        result.mean_reproj_error
    );
    assert_near_truth("rig solve", &result.cameras);
    let export = session.export().expect("export");
    let Uncertainty::Available(cov) = &export.uncertainty else {
        panic!("no covariance: {:?}", export.uncertainty);
    };
    for label in ["cam1.tilt_x", "cam1.tilt_y"] {
        assert!(
            cov.index(label).is_some(),
            "the rig covariance lacks {label}"
        );
    }
}

/// Monte-Carlo trials of the gate. A scatter over `TRIALS` realizations has
/// a relative standard error of `1/√(2·TRIALS)`, 3.5 % at 400.
const TRIALS: u64 = 400;
/// Each scatter within this fraction of its stated σ: about 3.4 standard
/// errors. Measured: 0.94 to 1.02 on either backend; with the tilts held at
/// the per-camera estimates the principal points scatter about 20× their
/// stated σ.
const SIGMA_TOLERANCE: f64 = 0.12;

fn sigma(cov: &ParameterCovariance, label: &str) -> f64 {
    let i = cov
        .index(label)
        .unwrap_or_else(|| panic!("the covariance lacks {label}"));
    cov.covariance[i][i]
}

/// The error of camera `i`'s labelled parameter.
fn camera_error(camera: &CameraParams, truth: &TruthCamera, parameter: &str) -> f64 {
    let SensorParams::Scheimpflug { params: tilt } = camera.sensor else {
        panic!("the camera lost its Scheimpflug sensor");
    };
    let (k, kt) = (&camera.intrinsics, &truth.k);
    match parameter {
        "fx" => k.fx - kt.fx,
        "fy" => k.fy - kt.fy,
        "cx" => k.cx - kt.cx,
        "cy" => k.cy - kt.cy,
        "tilt_x" => tilt.tilt_x - truth.tilt.tilt_x,
        "tilt_y" => tilt.tilt_y - truth.tilt.tilt_y,
        other => panic!("unexpected parameter {other}"),
    }
}

/// The rig solve's stated σ of every camera's focal length, principal point
/// and tilt matches their scatter over noise realizations, on both solver
/// backends.
#[test]
#[ignore = "Monte-Carlo gate; run in release with --ignored"]
fn rig_covariance_with_tilts_matches_monte_carlo() {
    const PARAMETERS: [&str; 6] = ["fx", "fy", "cx", "cy", "tilt_x", "tilt_y"];
    for backend in [SolverBackend::TinySolver, SolverBackend::Factrs] {
        // [camera][parameter] → (errors, stated variances)
        let mut trials = vec![vec![(Vec::new(), Vec::new()); PARAMETERS.len()]; 2];
        for trial in 0..TRIALS {
            let mut session = per_camera_stage(1000 + 2 * trial, backend);
            step_intrinsics_optimize_all(&mut session, None).expect("intrinsics optimize");
            step_rig_init(&mut session, None).expect("rig init");
            step_rig_optimize(&mut session, None).expect("rig optimize");
            let export = session.export().expect("export");
            let Uncertainty::Available(cov) = &export.uncertainty else {
                panic!("trial {trial}: no covariance: {:?}", export.uncertainty);
            };
            for (c, (camera, truth)) in export.cameras.iter().zip(truth()).enumerate() {
                for (p, parameter) in PARAMETERS.iter().enumerate() {
                    let (errors, variances) = &mut trials[c][p];
                    errors.push(camera_error(camera, &truth, parameter));
                    variances.push(sigma(cov, &format!("cam{c}.{parameter}")));
                }
            }
        }
        for (c, camera) in trials.iter().enumerate() {
            for (parameter, (errors, variances)) in PARAMETERS.iter().zip(camera) {
                let ratio = scatter_over_stated(errors, variances);
                eprintln!("{backend:?} cam{c}.{parameter}: scatter / stated σ {ratio:.3}");
                assert!(
                    (ratio - 1.0).abs() < SIGMA_TOLERANCE,
                    "{backend:?} cam{c}.{parameter}: scatter / stated σ {ratio:.3}"
                );
            }
        }
    }
}
