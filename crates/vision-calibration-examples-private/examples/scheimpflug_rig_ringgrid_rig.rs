//! Full rig + hand-eye calibration on the private `scheimpflug_rig_ringgrid` dataset (a
//! 6-camera Scheimpflug rig captured against a coded **ring-grid** target).
//!
//! Companion to `scheimpflug_rig_ringgrid_intrinsics` (per-camera intrinsics debug loop)
//! and to the puzzleboard `scheimpflug_rig_ref_rig`. Starting only from the images and the
//! robot `tcp2base` poses, it recovers per-camera intrinsics, the rig extrinsics
//! (`cam_se3_rig`), and the hand-eye transform.
//!
//! Pipeline (`RigHandeyeProblem`, Scheimpflug camera model):
//!   intrinsics init → intrinsics BA (+ tilt) → rig init → rig BA →
//!   hand-eye init → hand-eye BA.
//!
//! Intrinsics init is **seeded** by default (coarse focal + nominal mount tilt,
//! the ADR 0011 supported path): Zhang-from-scratch is fragile on Scheimpflug
//! optics. Set `SCHEIMPFLUG_RIG_RINGGRID_FROM_SCRATCH=1` to use Zhang's method instead.
//!
//! The `scheimpflug_rig_ringgrid` dataset ships **no laser images and no oracle** — it is a
//! different physical rig from the puzzleboard `scheimpflug_rig`/`scheimpflug_rig_ref`. So there is
//! no per-parameter ground truth to compare against; the final block reports a
//! **calibration-quality comparison** of ring-grid vs the puzzleboard `scheimpflug_rig_ref`
//! oracle reprojection error (loaded read-only, quality context only — the
//! recovered intrinsics/extrinsics are *not* expected to agree across rigs).
//!
//! Run:
//! `cargo run --release --manifest-path
//! crates/vision-calibration-examples-private/Cargo.toml --example
//! scheimpflug_rig_ringgrid_rig`
//!
//! Env:
//! - `SCHEIMPFLUG_RIG_RINGGRID_DATA_DIR` (default `privatedata/scheimpflug_rig_ringgrid`).
//! - `SCHEIMPFLUG_RIG_RINGGRID_HANDEYE` = `eye_in_hand` (default) | `eye_to_hand`.
//! - `SCHEIMPFLUG_RIG_RINGGRID_FOCAL` — coarse focal seed in px (default `1150`).
//! - `SCHEIMPFLUG_RIG_RINGGRID_TILT_X` — nominal mount-tilt seed in rad (default `-0.087`).
//! - `SCHEIMPFLUG_RIG_RINGGRID_RING_WIDTH_MM` — decode band-width override.
//! - `SCHEIMPFLUG_RIG_RINGGRID_CENTER` — marker centre estimator, `ellipse` (default) |
//!   `photometric`. A cache path given in `SCHEIMPFLUG_RIG_RINGGRID_CACHE` gets a
//!   `.photometric` suffix for the photometric method, so the two never share a cache.
//! - `SCHEIMPFLUG_RIG_RINGGRID_MAXITERS` (default `60`).
//! - `SCHEIMPFLUG_RIG_RINGGRID_FROM_SCRATCH` — set `1` to use Zhang init instead of seeded.
//! - `SCHEIMPFLUG_RIG_PUZZLE_REF_DIR` — puzzleboard oracle dir for the baseline (default
//!   `privatedata/scheimpflug_rig_ref`; skipped if absent).

use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use vision_calibration::common::config::{CameraConfig, RobotPoseConfig};
use vision_calibration::common::{HandeyeSolution, RigIntrinsicsManualInit};
use vision_calibration::{
    rig_handeye::{self as rh, RigHandeyeConfig, RigHandeyeProblem},
    session::CalibrationSession,
};
use vision_calibration_core::{
    CalibratedRig, CameraFix, CameraModelSpec, CorrespondenceView, DistortionKind, FxFyCxCy,
    RigDataset, RigView, RigViewObs, ScheimpflugParams, SensorKind,
};
use vision_calibration_core::{Iso3, PointCovariance, Pt2, RunControl, Uncertainty};
use vision_calibration_detect::RinggridCenterMethod;
use vision_calibration_examples_private::{
    BoardRinggridSpec, camera_parts, detect_ringgrid_all, load_poses, load_ref_artifacts,
    load_ringgrid_board,
};
use vision_calibration_optim::{HandeyeMode, RobotPoseMeta, RobustLoss};
use vision_mvg::{
    triangulation::{RigObservation, triangulate},
    types::Captures,
};

const NUM_CAMERAS: usize = 6;
const TILE_CX: f64 = 360.0;
const TILE_CY: f64 = 270.0;
const DEFAULT_TILT_X: f64 = -0.087;

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() -> Result<()> {
    let data_dir = PathBuf::from(
        std::env::var("SCHEIMPFLUG_RIG_RINGGRID_DATA_DIR")
            .unwrap_or_else(|_| "privatedata/scheimpflug_rig_ringgrid".to_string()),
    );
    let handeye_mode = match std::env::var("SCHEIMPFLUG_RIG_RINGGRID_HANDEYE").as_deref() {
        Ok("eye_to_hand") => HandeyeMode::EyeToHand,
        _ => HandeyeMode::EyeInHand,
    };
    let focal_seed = env_f64("SCHEIMPFLUG_RIG_RINGGRID_FOCAL", 1150.0);
    let tilt_x_seed = env_f64("SCHEIMPFLUG_RIG_RINGGRID_TILT_X", DEFAULT_TILT_X);
    let from_scratch = std::env::var("SCHEIMPFLUG_RIG_RINGGRID_FROM_SCRATCH").as_deref() == Ok("1");

    let board: BoardRinggridSpec = load_ringgrid_board(&data_dir.join("board_ringgrid.json"))
        .context("load ring-grid board manifest")?;
    let ring_width_mm = env_f64(
        "SCHEIMPFLUG_RIG_RINGGRID_RING_WIDTH_MM",
        board.ring_width_mm(),
    );

    println!("data dir = {}", data_dir.display());
    println!("hand-eye mode = {handeye_mode:?}");
    println!(
        "intrinsics init = {}",
        if from_scratch {
            "Zhang (from scratch)".to_string()
        } else {
            format!("seeded (fx=fy={focal_seed:.0}, tilt_x={tilt_x_seed:.3} rad)")
        }
    );
    println!(
        "board: {} rows × {} long-row cols, pitch {:.1} mm, ring width {:.3} mm",
        board.rows, board.long_row_cols, board.pitch_mm, ring_width_mm,
    );

    let poses = load_poses(&data_dir.join("poses.json"))?;
    println!("loaded {} poses", poses.len());

    // ── Detect the ring-grid in every camera tile of every pose ──────────────
    // Detection is the slow stage; `SCHEIMPFLUG_RIG_RINGGRID_CACHE=<path>` memoizes it so
    // solve-only iterations are fast.
    let center_method = match std::env::var("SCHEIMPFLUG_RIG_RINGGRID_CENTER").as_deref() {
        Ok("photometric") => RinggridCenterMethod::Photometric,
        Ok("ellipse") | Err(_) => RinggridCenterMethod::Ellipse,
        Ok(other) => anyhow::bail!(
            "SCHEIMPFLUG_RIG_RINGGRID_CENTER must be `ellipse` or `photometric`, got `{other}`"
        ),
    };
    println!("marker centre method = {center_method:?}");
    // The cache holds detections of one centre method only: the photometric
    // run reads and writes `<path>.photometric`.
    let cache_path = std::env::var("SCHEIMPFLUG_RIG_RINGGRID_CACHE")
        .ok()
        .map(|p| match center_method {
            RinggridCenterMethod::Photometric => PathBuf::from(format!("{p}.photometric")),
            _ => PathBuf::from(p),
        });
    let t_det = Instant::now();
    let per_pose = detect_ringgrid_all(
        &data_dir,
        &poses,
        &board,
        ring_width_mm,
        center_method,
        cache_path.as_deref(),
    )?;
    println!("detect: {:.2?}", t_det.elapsed());

    let mut views: Vec<RigView<RobotPoseMeta>> = Vec::with_capacity(poses.len());
    let mut det_views = [0usize; NUM_CAMERAS];
    let mut det_markers = [0usize; NUM_CAMERAS];
    for (pose, cams) in poses.iter().zip(per_pose) {
        for (c, cam) in cams.iter().enumerate() {
            if let Some(v) = cam {
                det_views[c] += 1;
                det_markers[c] += v.points_2d.len();
            }
        }
        views.push(RigView::new(
            RigViewObs::new(cams),
            RobotPoseMeta::new(pose.base_se3_gripper()),
        ));
    }
    for c in 0..NUM_CAMERAS {
        println!(
            "  cam {c}: {} views, {} markers",
            det_views[c], det_markers[c]
        );
    }

    // ── RigHandeye(Scheimpflug) calibration ──────────────────────────────────
    let observations: Vec<Vec<Option<CorrespondenceView>>> =
        views.iter().map(|v| v.obs.cameras.clone()).collect();
    let dataset = RigDataset::new(views, NUM_CAMERAS)?;
    let mut session =
        CalibrationSession::<RigHandeyeProblem>::with_description("scheimpflug_rig_ringgrid_rig");
    session.set_input(dataset)?;

    let mut cfg = RigHandeyeConfig::default();
    cfg.solver.max_iters = std::env::var("SCHEIMPFLUG_RIG_RINGGRID_MAXITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    cfg.solver.verbosity = 0;
    let huber = RobustLoss::Huber { scale: 1.0 };
    cfg.intrinsics_optimize.reprojection_loss = huber;
    cfg.rig.reprojection_loss = huber;
    cfg.handeye_optimize.reprojection_loss = huber;
    cfg.handeye_init.handeye_mode = handeye_mode;
    // k1,k2 free; k3 + tangential fixed; both Scheimpflug tilts free.
    // The nominal tilt (`tilt_x_seed`) seeds the seeded init below through
    // `per_cam_sensors`. It is used directly (the tilt sweep is skipped); a
    // from-scratch run has no tilt seed and runs the sweep.
    cfg.camera = CameraConfig::new(CameraModelSpec::pinhole(
        DistortionKind::BrownConrady5,
        SensorKind::Scheimpflug,
    ));
    cfg.camera.fix = CameraFix::default();
    cfg.camera.fix.sensor = Some(Vec::new());
    cfg.camera.fix.distortion = Some(["k3", "p1", "p2"].iter().map(|n| n.to_string()).collect());
    cfg.rig.refine_sensor = false;
    cfg.rig.refine_intrinsics = false;
    // Robot-pose refinement on by default; `SCHEIMPFLUG_RIG_RINGGRID_REFINE_ROBOT=0` holds
    // the measured poses fixed (a divergence-isolation knob).
    let refine_robot = std::env::var("SCHEIMPFLUG_RIG_RINGGRID_REFINE_ROBOT").as_deref() != Ok("0");
    if !refine_robot {
        cfg.handeye_optimize.robot_poses = RobotPoseConfig::Exact;
    }
    // Optionally let the final hand-eye BA also refine rig extrinsics / tilts.
    if std::env::var("SCHEIMPFLUG_RIG_RINGGRID_REFINE_EXTRINSICS").as_deref() == Ok("1") {
        cfg.handeye_optimize.refine_cam_se3_rig = true;
    }
    if std::env::var("SCHEIMPFLUG_RIG_RINGGRID_REFINE_TILT").as_deref() == Ok("1") {
        cfg.handeye_optimize.refine_sensor = true;
    }
    println!("refine robot poses = {refine_robot}");
    session.set_config(cfg)?;

    let t0 = Instant::now();
    // Intrinsics init: seeded (default) or Zhang from scratch.
    let s = Instant::now();
    if from_scratch {
        rh::step_intrinsics_init_all(&mut session, None)?;
        println!("  intrinsics init (Zhang): {:.2?}", s.elapsed());
    } else {
        // `RigIntrinsicsManualInit` is #[non_exhaustive] → build via
        // Default and set the public fields.
        let mut seed = RigIntrinsicsManualInit::default();
        seed.per_cam_intrinsics = Some(vec![
            FxFyCxCy {
                fx: focal_seed,
                fy: focal_seed,
                cx: TILE_CX,
                cy: TILE_CY,
            };
            NUM_CAMERAS
        ]);
        seed.per_cam_sensors = Some(vec![
            ScheimpflugParams {
                tilt_x: tilt_x_seed,
                tilt_y: 0.0,
            };
            NUM_CAMERAS
        ]);
        rh::step_intrinsics_init_all_with_seed(&mut session, seed, None)?;
        println!("  intrinsics init (seeded): {:.2?}", s.elapsed());
    }
    let s = Instant::now();
    let intr = rh::step_intrinsics_optimize_all(&mut session, None)?;
    println!(
        "  intrinsics BA: {:.2?}  per-cam reproj={:?}",
        s.elapsed(),
        intr.per_cam_reproj_errors
            .iter()
            .map(|e| (e * 1e4).round() / 1e4)
            .collect::<Vec<_>>()
    );
    println!("── stage: intrinsics BA (own per-camera poses) ──");
    let intr_rig = CalibratedRig::new(
        intr.cameras.clone(),
        vec![Iso3::identity(); NUM_CAMERAS],
        Uncertainty::default(),
    )?;
    stage_report(&intr_rig, &observations, |v, c| {
        intr.cam_se3_target
            .get(v)
            .and_then(|r| r.get(c))
            .copied()
            .flatten()
    });
    let s = Instant::now();
    rh::step_rig_init(&mut session, None)?;
    println!("  rig init: {:.2?}", s.elapsed());
    let s = Instant::now();
    let rig = rh::step_rig_optimize(&mut session, None)?;
    println!(
        "  rig BA: {:.2?}  rig_reproj={:.4}px",
        s.elapsed(),
        rig.mean_reproj_error
    );

    println!("── stage: rig BA ──");
    let rig_model = CalibratedRig::new(
        intr.cameras.clone(),
        rig.cam_se3_rig.clone(),
        Uncertainty::default(),
    )?;
    stage_report(&rig_model, &observations, |v, c| {
        rig.rig_se3_target.get(v).map(|t| rig.cam_se3_rig[c] * t)
    });

    // Hand-eye consistency probe: for eye-in-hand, A·X = X·B with A = robot
    // relative motion and B = rig relative motion, so rotation ANGLES must
    // match pairwise (conjugation preserves angle). A large angle mismatch
    // means the recovered per-view rig poses do not trace the robot motion —
    // the hand-eye stage cannot then fit a single transform.
    vision_calibration_examples_private::handeye_consistency_probe(&poses, &rig.rig_se3_target);
    let s = Instant::now();
    rh::step_handeye_init(&mut session, None)?;
    println!("  hand-eye init: {:.2?}", s.elapsed());
    let s = Instant::now();
    rh::step_handeye_optimize(&mut session, None)?;
    println!("  hand-eye BA: {:.2?}", s.elapsed());
    let export = session.export()?;
    println!(
        "calibrate total: {:.2?}  final_mean_reproj={:.4}px",
        t0.elapsed(),
        export.mean_reproj_error
    );

    report(&export, &data_dir);
    handeye_stage_report(&export);
    cross_camera_check(&export, &observations)?;
    Ok(())
}

fn report(export: &vision_calibration::rig_handeye::RigHandeyeExport, data_dir: &std::path::Path) {
    println!("\n── Recovered ring-grid intrinsics (this rig — no oracle) ──");
    println!(
        "  cam |    fx    |    fy    |   cx   |   cy   |    k1     |    k2     |  tau_x° |  tau_y°"
    );
    for i in 0..NUM_CAMERAS {
        let (k, d, s) = camera_parts(&export.cameras[i]).expect("rig camera model");
        println!(
            "  {i:>3} | {:8.1} | {:8.1} | {:6.1} | {:6.1} | {:+.5} | {:+.5} | {:+.3} | {:+.3}",
            k.fx,
            k.fy,
            k.cx,
            k.cy,
            d.k1,
            d.k2,
            s.tilt_x.to_degrees(),
            s.tilt_y.to_degrees(),
        );
    }

    println!("\n── Recovered rig extrinsics (cam_se3_rig) ──");
    println!("  cam | |t| mm  | (camera 0 is the rig gauge)");
    for i in 0..NUM_CAMERAS {
        let t_mm = export.cam_se3_rig[i].translation.vector.norm() * 1e3;
        println!("  {i:>3} | {t_mm:8.2}");
    }

    println!("\n── Recovered hand-eye ──");
    let (mode, he) = match &export.handeye {
        HandeyeSolution::EyeInHand {
            gripper_se3_rig, ..
        } => ("EyeInHand", gripper_se3_rig),
        HandeyeSolution::EyeToHand { rig_se3_base, .. } => ("EyeToHand", rig_se3_base),
    };
    println!(
        "  {mode}: |t| = {:.2} mm",
        he.translation.vector.norm() * 1e3
    );

    // ── Quality comparison vs puzzleboard scheimpflug_rig_ref (different rig) ───────────
    let puzzle_ref_dir = PathBuf::from(
        std::env::var("SCHEIMPFLUG_RIG_PUZZLE_REF_DIR")
            .unwrap_or_else(|_| "privatedata/scheimpflug_rig_ref".to_string()),
    );
    let puzzle = load_ref_artifacts(&puzzle_ref_dir.join("artifacts.json")).ok();
    let _ = data_dir; // ring-grid dir has no oracle; baseline comes from puzzle_ref_dir

    println!(
        "\n── Calibration quality: ring-grid (this rig) vs puzzleboard scheimpflug_rig_ref ──"
    );
    println!(
        "  (DIFFERENT rigs — compares how well each target calibrates, not parameter agreement)"
    );
    println!("  cam | ringgrid_reproj | puzzle_ref_reproj |    Δ");
    let mut sum_ours = 0.0;
    let mut n_ours = 0.0;
    let mut sum_ref = 0.0;
    let mut n_ref = 0.0;
    let mut ours_vals: Vec<f64> = Vec::new();
    for i in 0..NUM_CAMERAS {
        let ours = export.per_cam_reproj_errors.get(i).copied();
        let refp = puzzle
            .as_ref()
            .and_then(|a| a.intrinsic.get(i))
            .map(|x| x.reprojection_error_pix);
        let ours_s = match ours {
            Some(v) => {
                sum_ours += v;
                n_ours += 1.0;
                ours_vals.push(v);
                format!("{v:15.4}")
            }
            None => format!("{:>15}", "n/a"),
        };
        let ref_s = match refp {
            Some(v) => {
                sum_ref += v;
                n_ref += 1.0;
                format!("{v:17.4}")
            }
            None => format!("{:>17}", "n/a"),
        };
        let delta_s = match (ours, refp) {
            (Some(a), Some(b)) => format!("{:+.4}", a - b),
            _ => "    n/a".to_string(),
        };
        println!("  {i:>3} | {ours_s} | {ref_s} | {delta_s}");
    }
    if n_ours > 0.0 {
        let mean_ours = sum_ours / n_ours;
        ours_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_ours = ours_vals[ours_vals.len() / 2];
        let mean_ref = if n_ref > 0.0 {
            format!("{:.4}", sum_ref / n_ref)
        } else {
            "n/a".to_string()
        };
        println!(
            "  ring-grid: mean {mean_ours:.4} px  median {median_ours:.4} px   puzzleboard: mean {mean_ref} px"
        );
        println!(
            "  (median is the representative ring-grid floor; the mean is skewed by weak cameras)"
        );
        println!(
            "  (the ring-grid floor is expected to sit slightly above puzzleboard: sparser markers,"
        );
        println!(
            "   ellipse-center vs dense-corner localization. Both calibrate the model the same way.)"
        );
    }
}

/// Count, mean, RMS and max of a set of errors.
fn summarize(v: &[f64]) -> String {
    if v.is_empty() {
        return "n=0".to_string();
    }
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let rms = (v.iter().map(|e| e * e).sum::<f64>() / n).sqrt();
    let max = v.iter().cloned().fold(0.0, f64::max);
    format!(
        "n={:>6}  mean={mean:.4}  rms={rms:.4}  max={max:.4}",
        v.len()
    )
}

/// Reprojection statistics of one stage, per camera and pooled: every
/// observation reprojected through its camera model and the stage's
/// `cam_se3_target(view, cam)`.
fn stage_report(
    rig: &CalibratedRig,
    observations: &[Vec<Option<CorrespondenceView>>],
    cam_se3_target: impl Fn(usize, usize) -> Option<Iso3>,
) {
    let mut per_cam: Vec<Vec<f64>> = vec![Vec::new(); NUM_CAMERAS];
    for (v, cams) in observations.iter().enumerate() {
        for (c, obs) in cams.iter().enumerate() {
            let (Some(obs), Some(pose), Some(model)) = (obs, cam_se3_target(v, c), rig.model(c))
            else {
                continue;
            };
            for (p3, p2) in obs.points_3d.iter().zip(&obs.points_2d) {
                if let Some(px) = model.project_point(&(pose * p3)) {
                    per_cam[c].push((px - p2).norm());
                }
            }
        }
    }
    for (c, e) in per_cam.iter().enumerate() {
        println!("  cam {c}: {}", summarize(e));
    }
    let all: Vec<f64> = per_cam.concat();
    println!("  pooled: {}", summarize(&all));
}

/// Hand-eye stage reprojection (the export's per-feature residuals), the
/// rig baselines and the stated uncertainty of the extrinsics and hand-eye.
fn handeye_stage_report(export: &vision_calibration::rig_handeye::RigHandeyeExport) {
    println!("── stage: hand-eye BA (export per-feature residuals) ──");
    let mut per_cam: Vec<Vec<f64>> = vec![Vec::new(); NUM_CAMERAS];
    for r in &export.diagnostics.per_feature_residuals.target {
        if let Some(e) = r.error_px {
            per_cam[r.camera].push(e);
        }
    }
    for (c, e) in per_cam.iter().enumerate() {
        println!("  cam {c}: {}", summarize(e));
    }
    println!("  pooled: {}", summarize(&per_cam.concat()));

    println!("── rig baselines: camera centre distance from cam 0 (mm) ──");
    let centre = |i: usize| export.cam_se3_rig[i].inverse().translation.vector;
    for i in 0..NUM_CAMERAS {
        println!("  cam {i}: {:.3}", (centre(i) - centre(0)).norm() * 1e3);
    }
    let he = match &export.handeye {
        HandeyeSolution::EyeInHand {
            gripper_se3_rig, ..
        } => gripper_se3_rig,
        HandeyeSolution::EyeToHand { rig_se3_base, .. } => rig_se3_base,
    };
    println!(
        "── hand-eye |t| = {:.3} mm ──",
        he.translation.vector.norm() * 1e3
    );

    println!("── stated uncertainty (σ) of rig extrinsics and hand-eye ──");
    match export.uncertainty.covariance() {
        None => println!("  unavailable: {:?}", export.uncertainty),
        Some(cov) => {
            for label in &cov.labels {
                let is_pose = label.contains("rig") || label.contains("handeye");
                if !is_pose {
                    continue;
                }
                let sigma = cov.std_dev(label).unwrap_or(f64::NAN);
                // translations are metres, rotations radians
                if label.ends_with(".tx") || label.ends_with(".ty") || label.ends_with(".tz") {
                    println!("  {label}: {:.4} mm", sigma * 1e3);
                } else {
                    println!("  {label}: {:.4} mrad", sigma * 1e3);
                }
            }
            println!(
                "  noise: {:?}  view_scatter: {:?}",
                cov.noise, cov.view_scatter
            );
        }
    }
}

/// A board point (target frame) and the `(camera, pixel)` observations of it in one view.
type BoardPoint = (vision_calibration_core::Pt3, Vec<(usize, Pt2)>);

/// Cross-camera consistency from the calibrated rig alone (rig frame, no
/// per-view pose involved): every board point seen by two or more cameras
/// is triangulated from all its observations and again leaving out each
/// camera in turn; the left-out camera's observation is then compared with
/// the prediction. Neighbouring points' distances are checked against the
/// board pitch.
fn cross_camera_check(
    export: &vision_calibration::rig_handeye::RigHandeyeExport,
    observations: &[Vec<Option<CorrespondenceView>>],
) -> Result<()> {
    println!("── cross-camera consistency (unit pixel noise, no covariance) ──");
    let rig = CalibratedRig::new(
        export.cameras.clone(),
        export.cam_se3_rig.clone(),
        Uncertainty::default(),
    )?;
    let noise = PointCovariance::isotropic(1.0)?;
    let key =
        |p: &vision_calibration_core::Pt3| ((p.x * 1e7).round() as i64, (p.y * 1e7).round() as i64);
    let run = |tracks: &[Vec<RigObservation>]| {
        triangulate(&rig, Captures::Rig, tracks, &RunControl::default())
    };

    let mut diff3d: Vec<Vec<f64>> = vec![Vec::new(); NUM_CAMERAS];
    let mut loo_px: Vec<Vec<f64>> = vec![Vec::new(); NUM_CAMERAS];
    // Distance errors by nominal distance class: nearest neighbours (hex
    // lattice, 8 mm * sqrt(3)), then up to 30 mm and up to 60 mm.
    let mut metric: [Vec<f64>; 3] = Default::default();
    let mut n_points_2plus = 0usize;
    let mut n_points_3plus = 0usize;
    let mut n_pairs_2 = 0usize;
    let mut failed_calls = 0usize;

    for cams in observations {
        // key -> board xy, observations
        let mut pts: BTreeMap<(i64, i64), BoardPoint> = BTreeMap::new();
        for (c, obs) in cams.iter().enumerate() {
            let Some(obs) = obs else { continue };
            for (p3, p2) in obs.points_3d.iter().zip(&obs.points_2d) {
                pts.entry(key(p3))
                    .or_insert((*p3, Vec::new()))
                    .1
                    .push((c, *p2));
            }
        }
        let multi: Vec<_> = pts.iter().filter(|(_, (_, o))| o.len() >= 2).collect();
        n_points_2plus += multi.len();
        n_points_3plus += multi.iter().filter(|(_, (_, o))| o.len() >= 3).count();
        n_pairs_2 += multi.iter().filter(|(_, (_, o))| o.len() == 2).count();
        if multi.is_empty() {
            continue;
        }
        let track = |o: &[(usize, Pt2)], skip: Option<usize>| -> Vec<RigObservation> {
            o.iter()
                .filter(|(c, _)| Some(*c) != skip)
                .map(|(c, px)| RigObservation::new(0, *c, *px, noise))
                .collect()
        };
        let all_tracks: Vec<_> = multi.iter().map(|(_, (_, o))| track(o, None)).collect();
        let Ok(all) = run(&all_tracks) else {
            failed_calls += 1;
            continue;
        };
        let all_pt: BTreeMap<(i64, i64), (vision_calibration_core::Pt3, bool)> = multi
            .iter()
            .zip(&all.points)
            .map(|((k, _), t)| (**k, (t.point, t.in_front)))
            .collect();

        // metric: neighbouring points
        let keys: Vec<_> = multi.iter().map(|(k, _)| **k).collect();
        for (a, ka) in keys.iter().enumerate() {
            for kb in &keys[a + 1..] {
                let (na, nb) = (pts[ka].0, pts[kb].0);
                let nominal = (na - nb).norm();
                let class = [15e-3, 30e-3, 60e-3].iter().position(|&m| nominal < m);
                if let Some(class) = class {
                    let (pa, fa) = all_pt[ka];
                    let (pb, fb) = all_pt[kb];
                    if fa && fb {
                        metric[class].push((pa - pb).norm() - nominal);
                    }
                }
            }
        }

        for c in 0..NUM_CAMERAS {
            let sel: Vec<_> = multi
                .iter()
                .filter(|(_, (_, o))| o.len() >= 3 && o.iter().any(|(cc, _)| *cc == c))
                .collect();
            if sel.is_empty() {
                continue;
            }
            let tracks: Vec<_> = sel.iter().map(|(_, (_, o))| track(o, Some(c))).collect();
            let Ok(loo) = run(&tracks) else {
                failed_calls += 1;
                continue;
            };
            let model = rig.model(c).expect("camera index in range");
            for ((k, (_, o)), t) in sel.iter().zip(&loo.points) {
                if !t.in_front {
                    continue;
                }
                let (full, full_front) = all_pt[*k];
                if full_front {
                    diff3d[c].push((full - t.point).norm());
                }
                let obs = o.iter().find(|(cc, _)| *cc == c).expect("selected").1;
                if let Some(px) = model.project_point(&(export.cam_se3_rig[c] * t.point)) {
                    loo_px[c].push((px - obs).norm());
                }
            }
        }
    }

    println!(
        "  points seen by >=2 cams: {n_points_2plus} (exactly 2: {n_pairs_2}, >=3: {n_points_3plus}); failed triangulation calls: {failed_calls}"
    );
    let rms = |v: &[f64]| (v.iter().map(|e| e * e).sum::<f64>() / v.len().max(1) as f64).sqrt();
    println!("  (a) 3D difference, all vs leave-camera-out, RMS in um (per left-out camera):");
    for (c, d) in diff3d.iter().enumerate() {
        println!("    cam {c}: n={:>5}  rms={:.1}", d.len(), rms(d) * 1e6);
    }
    let all3: Vec<f64> = diff3d.concat();
    println!(
        "    pooled: n={:>5}  rms={:.1}",
        all3.len(),
        rms(&all3) * 1e6
    );
    println!("  (b) leave-camera-out reprojection in the left-out camera, px:");
    for (c, e) in loo_px.iter().enumerate() {
        println!("    cam {c}: {}", summarize(e));
    }
    println!("    pooled: {}", summarize(&loo_px.concat()));
    println!("  (c) triangulated distance vs nominal board distance (um):");
    for (m, label) in metric
        .iter()
        .zip(["<15 mm (nearest)", "15-30 mm", "30-60 mm"])
    {
        let n = m.len().max(1) as f64;
        println!(
            "    {label:>16}: n={:>5}  mean signed={:+.1}  mean abs={:.1}  rms={:.1}",
            m.len(),
            m.iter().sum::<f64>() / n * 1e6,
            m.iter().map(|e| e.abs()).sum::<f64>() / n * 1e6,
            rms(m) * 1e6
        );
    }
    Ok(())
}
