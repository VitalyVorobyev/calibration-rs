//! Full rig + hand-eye calibration on the private `rtv3d_ringgrid` dataset (a
//! 6-camera Scheimpflug rig captured against a coded **ring-grid** target).
//!
//! Companion to `rtv3d_ringgrid_intrinsics` (per-camera intrinsics debug loop)
//! and to the puzzleboard `rtv3d_ref_rig`. Starting only from the images and the
//! robot `tcp2base` poses, it recovers per-camera intrinsics, the rig extrinsics
//! (`cam_se3_rig`), and the hand-eye transform.
//!
//! Pipeline (`RigHandeyeProblem`, `SensorMode::Scheimpflug`):
//!   intrinsics init → intrinsics BA (+ tilt) → rig init → rig BA →
//!   hand-eye init → hand-eye BA.
//!
//! Intrinsics init is **seeded** by default (coarse focal + nominal mount tilt,
//! the ADR 0022 supported path): Zhang-from-scratch is fragile on Scheimpflug
//! optics. Set `RTV3D_RINGGRID_FROM_SCRATCH=1` to use Zhang's method instead.
//!
//! The `rtv3d_ringgrid` dataset ships **no laser images and no oracle** — it is a
//! different physical rig from the puzzleboard `rtv3d`/`rtv3d_ref`. So there is
//! no per-parameter ground truth to compare against; the final block reports a
//! **calibration-quality comparison** of ring-grid vs the puzzleboard `rtv3d_ref`
//! oracle reprojection error (loaded read-only, quality context only — the
//! recovered intrinsics/extrinsics are *not* expected to agree across rigs).
//!
//! Run:
//! `cargo run --release --manifest-path
//! crates/vision-calibration-examples-private/Cargo.toml --example
//! rtv3d_ringgrid_rig`
//!
//! Env:
//! - `RTV3D_RINGGRID_DATA_DIR` (default `privatedata/rtv3d_ringgrid`).
//! - `RTV3D_RINGGRID_HANDEYE` = `eye_in_hand` (default) | `eye_to_hand`.
//! - `RTV3D_RINGGRID_FOCAL` — coarse focal seed in px (default `1150`).
//! - `RTV3D_RINGGRID_TILT_X` — nominal mount-tilt seed in rad (default `-0.087`).
//! - `RTV3D_RINGGRID_RING_WIDTH_MM` — decode band-width override.
//! - `RTV3D_RINGGRID_MAXITERS` (default `60`).
//! - `RTV3D_RINGGRID_FROM_SCRATCH` — set `1` to use Zhang init instead of seeded.
//! - `RTV3D_PUZZLE_REF_DIR` — puzzleboard oracle dir for the baseline (default
//!   `privatedata/rtv3d_ref`; skipped if absent).

use anyhow::{Context, Result};
use std::path::PathBuf;
use std::time::Instant;

use vision_calibration::{
    rig_handeye::{
        self as rh, RigHandeyeConfig, RigHandeyeIntrinsicsManualInit, RigHandeyeProblem, SensorMode,
    },
    session::CalibrationSession,
};
use vision_calibration_core::{
    DistortionFixMask, FxFyCxCySkew, RigDataset, RigView, RigViewObs, ScheimpflugParams,
};
use vision_calibration_examples_private::{
    BoardRinggridSpec, detect_ringgrid_all, load_poses, load_ref_artifacts, load_ringgrid_board,
};
use vision_calibration_optim::{HandEyeMode, RobotPoseMeta, RobustLoss, ScheimpflugFixMask};

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
        std::env::var("RTV3D_RINGGRID_DATA_DIR")
            .unwrap_or_else(|_| "privatedata/rtv3d_ringgrid".to_string()),
    );
    let handeye_mode = match std::env::var("RTV3D_RINGGRID_HANDEYE").as_deref() {
        Ok("eye_to_hand") => HandEyeMode::EyeToHand,
        _ => HandEyeMode::EyeInHand,
    };
    let focal_seed = env_f64("RTV3D_RINGGRID_FOCAL", 1150.0);
    let tilt_x_seed = env_f64("RTV3D_RINGGRID_TILT_X", DEFAULT_TILT_X);
    let from_scratch = std::env::var("RTV3D_RINGGRID_FROM_SCRATCH").as_deref() == Ok("1");

    let board: BoardRinggridSpec = load_ringgrid_board(&data_dir.join("board_ringgrid.json"))
        .context("load ring-grid board manifest")?;
    let ring_width_mm = env_f64("RTV3D_RINGGRID_RING_WIDTH_MM", board.ring_width_mm());

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
    // Detection is the slow stage; `RTV3D_RINGGRID_CACHE=<path>` memoizes it so
    // solve-only iterations are fast.
    let cache_path = std::env::var("RTV3D_RINGGRID_CACHE")
        .ok()
        .map(PathBuf::from);
    let t_det = Instant::now();
    let per_pose = detect_ringgrid_all(
        &data_dir,
        &poses,
        &board,
        ring_width_mm,
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
        views.push(RigView {
            meta: RobotPoseMeta {
                base_se3_gripper: pose.base_se3_gripper(),
            },
            obs: RigViewObs { cameras: cams },
        });
    }
    for c in 0..NUM_CAMERAS {
        println!(
            "  cam {c}: {} views, {} markers",
            det_views[c], det_markers[c]
        );
    }

    // ── RigHandeye(Scheimpflug) calibration ──────────────────────────────────
    let dataset = RigDataset::new(views, NUM_CAMERAS)?;
    let mut session =
        CalibrationSession::<RigHandeyeProblem>::with_description("rtv3d_ringgrid_rig");
    session.set_input(dataset)?;

    let mut cfg = RigHandeyeConfig::default();
    cfg.solver.max_iters = std::env::var("RTV3D_RINGGRID_MAXITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    cfg.solver.verbosity = 0;
    cfg.solver.robust_loss = RobustLoss::Huber { scale: 1.0 };
    cfg.handeye_init.handeye_mode = handeye_mode;
    // k1,k2 free; k3 + tangential fixed; both Scheimpflug tilts free.
    cfg.sensor = SensorMode::Scheimpflug {
        init_tilt_x: tilt_x_seed,
        init_tilt_y: 0.0,
        fix_scheimpflug_in_intrinsics: ScheimpflugFixMask {
            tilt_x: false,
            tilt_y: false,
        },
        distortion_mask_in_percam_ba: DistortionFixMask {
            k1: false,
            k2: false,
            k3: true,
            p1: true,
            p2: true,
        },
        refine_scheimpflug_in_rig_ba: false,
    };
    cfg.rig.refine_intrinsics_in_rig_ba = false;
    // Robot-pose refinement on by default; `RTV3D_RINGGRID_REFINE_ROBOT=0` holds
    // the measured poses fixed (a divergence-isolation knob).
    let refine_robot = std::env::var("RTV3D_RINGGRID_REFINE_ROBOT").as_deref() != Ok("0");
    cfg.handeye_ba.refine_robot_poses = refine_robot;
    // Optionally let the final hand-eye BA also refine rig extrinsics / tilts.
    if std::env::var("RTV3D_RINGGRID_REFINE_EXTRINSICS").as_deref() == Ok("1") {
        cfg.handeye_ba.refine_cam_se3_rig_in_handeye_ba = true;
    }
    if std::env::var("RTV3D_RINGGRID_REFINE_TILT").as_deref() == Ok("1") {
        cfg.handeye_ba.refine_scheimpflug_in_handeye_ba = true;
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
        // `RigHandeyeIntrinsicsManualInit` is #[non_exhaustive] → build via
        // Default and set the public fields.
        let mut seed = RigHandeyeIntrinsicsManualInit::default();
        seed.per_cam_intrinsics = Some(vec![
            FxFyCxCySkew {
                fx: focal_seed,
                fy: focal_seed,
                cx: TILE_CX,
                cy: TILE_CY,
                skew: 0.0,
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
    let s = Instant::now();
    rh::step_rig_init(&mut session)?;
    println!("  rig init: {:.2?}", s.elapsed());
    let s = Instant::now();
    let rig = rh::step_rig_optimize(&mut session, None)?;
    println!(
        "  rig BA: {:.2?}  rig_reproj={:.4}px",
        s.elapsed(),
        rig.mean_reproj_error
    );

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

    report(&export, handeye_mode, &data_dir);
    Ok(())
}

fn report(
    export: &vision_calibration::rig_handeye::RigHandeyeExport,
    handeye_mode: HandEyeMode,
    data_dir: &std::path::Path,
) {
    println!("\n── Recovered ring-grid intrinsics (this rig — no oracle) ──");
    println!(
        "  cam |    fx    |    fy    |   cx   |   cy   |    k1     |    k2     |  tau_x° |  tau_y°"
    );
    let sensors = export.sensors.as_ref();
    for i in 0..NUM_CAMERAS {
        let c = &export.cameras[i];
        let s = sensors.map(|s| s[i]).unwrap_or_default();
        println!(
            "  {i:>3} | {:8.1} | {:8.1} | {:6.1} | {:6.1} | {:+.5} | {:+.5} | {:+.3} | {:+.3}",
            c.k.fx,
            c.k.fy,
            c.k.cx,
            c.k.cy,
            c.dist.k1,
            c.dist.k2,
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
    let he = match handeye_mode {
        HandEyeMode::EyeInHand => export.gripper_se3_rig,
        HandEyeMode::EyeToHand => export.rig_se3_base,
    };
    match he {
        Some(he) => println!(
            "  {handeye_mode:?}: |t| = {:.2} mm",
            he.translation.vector.norm() * 1e3
        ),
        None => println!("  no hand-eye of the expected topology in the export"),
    }

    // ── Quality comparison vs puzzleboard rtv3d_ref (different rig) ───────────
    let puzzle_ref_dir = PathBuf::from(
        std::env::var("RTV3D_PUZZLE_REF_DIR")
            .unwrap_or_else(|_| "privatedata/rtv3d_ref".to_string()),
    );
    let puzzle = load_ref_artifacts(&puzzle_ref_dir.join("artifacts.json")).ok();
    let _ = data_dir; // ring-grid dir has no oracle; baseline comes from puzzle_ref_dir

    println!("\n── Calibration quality: ring-grid (this rig) vs puzzleboard rtv3d_ref ──");
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
